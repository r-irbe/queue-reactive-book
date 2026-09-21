#pragma once

#include "event_trace.hpp"   // EventSeq
#include "export_schema.hpp"  // ExportColumn, ExportSchemaMeta, enums
#include "types.hpp"          // Side, Tick, Qty, OrderId, TimeNs, is_buy

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace sim {

// --- Queue-state / per-order lifecycle observability (Tier-2 PROTOTYPE) -------
//
// PROTOTYPE, GATED. This is a cold-path OBSERVER surface: it reconstructs a
// best-estimate per-order queue position and lifecycle from observations the
// caller supplies (resting qty ahead at submit, aggregate level trades/cancels,
// own fills/cancels). It NEVER holds a `FlatLadderBook` handle, never mutates
// matching, and is replay-hash invariant by construction -- driving it changes
// no book state or matching output (demonstrated by an A/B fixture).
//
// Honesty boundary (roadmap T3): `queue_ahead_qty` is EXACT only while
// `queue_ahead_censored == false`. Real market data rarely exposes true order
// rank, so any ambiguous event (unknown-position cancel, an aggregate trade that
// reaches/crosses the order, or an own fill that proves the queue-ahead estimate
// was stale) sets `queue_ahead_censored`, after which the field is a lower-
// confidence estimate, not ground-truth rank. Aggregate level trades update
// queue-ahead ONLY; per-order fill/lifecycle is driven solely by `on_own_fill`/
// `on_own_cancel` so aggregate prints can never fabricate own executions.
//
// FIFO assumption: trades consume the front of a price-time queue, so a level
// trade reduces every tracked order's queue-ahead by the consumed amount. This
// holds for the discrete-tick FIFO sim; it does NOT hold under pro-rata,
// scope for this prototype and, where observable, should set the censor flag.

enum class QueueLifecycle : std::uint8_t {
    Resting = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,  // partial-fill-then-cancel is Cancelled with filled_qty > 0
};

inline constexpr bool is_terminal(QueueLifecycle s) noexcept {
    return s == QueueLifecycle::Filled || s == QueueLifecycle::Cancelled;
}

struct QueueStateSample {
    OrderId order_id = 0;
    Side side = Side::Buy;
    Tick price = 0;
    EventSeq submit_seq = 0;
    TimeNs submit_ts = 0;
    EventSeq age_events = 0;  // current_seq - submit_seq (clamped at 0)
    TimeNs age_ns = 0;        // current_ts - submit_ts (clamped at 0)
    // Best estimate of resting qty ahead of this order. Exact only while
    // `queue_ahead_censored` is false; afterwards a lower-confidence estimate.
    Qty queue_ahead_qty = 0;
    bool queue_ahead_censored = false;
    Qty original_qty = 0;
    Qty filled_qty = 0;
    Qty remaining_qty = 0;
    // Same-side same-price cancels observed while this order was alive. Split so
    // a consumer never reads the total as a queue-ahead cancellation count.
    std::uint32_t cancels_total = 0;
    std::uint32_t cancels_unknown_position = 0;  // position ahead/behind unknown
    Qty cancels_ahead_qty = 0;                    // qty of cancels known to be ahead
    QueueLifecycle lifecycle = QueueLifecycle::Resting;
    // Observation ended without a terminal (Filled/Cancelled): a censored label
    // for survival use, NOT a fabricated outcome.
    bool right_censored = false;
};

inline constexpr std::size_t kQueueStateExportCount = 17;
inline constexpr std::array<ExportColumn, kQueueStateExportCount>
    kQueueStateExportSchema = {{{"order_id"},
                                {"side"},
                                {"price"},
                                {"submit_seq"},
                                {"submit_ts"},
                                {"age_events"},
                                {"age_ns"},
                                {"queue_ahead_qty"},
                                {"queue_ahead_censored"},
                                {"original_qty"},
                                {"filled_qty"},
                                {"remaining_qty"},
                                {"cancels_total"},
                                {"cancels_unknown_position"},
                                {"cancels_ahead_qty"},
                                {"lifecycle"},
                                {"right_censored"}}};

// The exported frame carries a censorable estimate column, so the missing policy
// is SentinelFlagged (the `queue_ahead_censored` column is the flag).
inline ExportSchemaMeta queue_state_schema_meta() noexcept {
    ExportSchemaMeta meta{};
    meta.schema_name = "queue_state";
    meta.timestamp_basis = TimestampBasis::EventTime;
    meta.price_unit = QuantityUnit::Ticks;
    meta.quantity_unit = QuantityUnit::Lots;
    meta.label_availability = LabelAvailability::NotApplicable;
    meta.missing_policy = MissingPolicy::SentinelFlagged;
    return meta;
}

inline std::array<double, kQueueStateExportCount>
export_queue_state_sample(const QueueStateSample& s) noexcept {
    return {static_cast<double>(s.order_id),
            is_buy(s.side) ? 1.0 : -1.0,
            static_cast<double>(s.price),
            static_cast<double>(s.submit_seq),
            static_cast<double>(s.submit_ts),
            static_cast<double>(s.age_events),
            static_cast<double>(s.age_ns),
            static_cast<double>(s.queue_ahead_qty),
            s.queue_ahead_censored ? 1.0 : 0.0,
            static_cast<double>(s.original_qty),
            static_cast<double>(s.filled_qty),
            static_cast<double>(s.remaining_qty),
            static_cast<double>(s.cancels_total),
            static_cast<double>(s.cancels_unknown_position),
            static_cast<double>(s.cancels_ahead_qty),
            static_cast<double>(s.lifecycle),
            s.right_censored ? 1.0 : 0.0};
}

class QueueStateObserver {
public:
    using Map = std::unordered_map<OrderId, QueueStateSample>;

    // Register a newly-rested order. `observed_qty_ahead` is the resting qty
    // already at this (side, price) at submit time -- in a price-time FIFO book
    // everything resting is ahead of us, so it seeds the queue-ahead estimate.
    void on_submit(OrderId id, Side side, Tick price, Qty qty, EventSeq seq,
                   TimeNs ts, Qty observed_qty_ahead) {
        QueueStateSample s{};
        s.order_id = id;
        s.side = side;
        s.price = price;
        s.submit_seq = seq;
        s.submit_ts = ts;
        s.original_qty = qty;
        s.remaining_qty = qty;
        s.queue_ahead_qty = observed_qty_ahead > 0 ? observed_qty_ahead : 0;
        s.lifecycle = QueueLifecycle::Resting;
        orders_[id] = s;
    }

    // Aggregate trade that consumed `traded_qty` from the front of (side, price).
    // Updates queue-ahead ONLY; never touches fill/remaining (that is own-fill's
    // job). If the trade reaches or crosses an order's queue-ahead, the estimate
    // becomes uncertain and is censored.
    void on_level_trade(Side side, Tick price, Qty traded_qty, EventSeq /*seq*/,
                        TimeNs /*ts*/) {
        if (traded_qty <= 0) return;
        for (auto& [id, s] : orders_) {
            (void)id;
            if (is_terminal(s.lifecycle)) continue;
            if (s.side != side || s.price != price) continue;
            if (traded_qty >= s.queue_ahead_qty) {
                if (traded_qty > s.queue_ahead_qty) s.queue_ahead_censored = true;
                s.queue_ahead_qty = 0;
            } else {
                s.queue_ahead_qty = static_cast<Qty>(s.queue_ahead_qty - traded_qty);
            }
        }
    }

    // Aggregate cancel of `cancelled_qty` at (side, price). With unknown position
    // (the default for real market data) we cannot tell whether it was ahead or
    // behind, so we censor the queue-ahead estimate rather than silently advance
    // it. Only a caller that genuinely knows the cancel was ahead may reduce
    // queue-ahead.
    void on_level_cancel(Side side, Tick price, Qty cancelled_qty,
                         EventSeq /*seq*/, TimeNs /*ts*/, bool position_known = false,
                         bool ahead = false) {
        for (auto& [id, s] : orders_) {
            (void)id;
            if (is_terminal(s.lifecycle)) continue;
            if (s.side != side || s.price != price) continue;
            ++s.cancels_total;
            if (!position_known) {
                ++s.cancels_unknown_position;
                s.queue_ahead_censored = true;
            } else if (ahead && cancelled_qty > 0) {
                s.cancels_ahead_qty = static_cast<Qty>(s.cancels_ahead_qty + cancelled_qty);
                const Qty take = cancelled_qty < s.queue_ahead_qty ? cancelled_qty
                                                                   : s.queue_ahead_qty;
                s.queue_ahead_qty = static_cast<Qty>(s.queue_ahead_qty - take);
            }
            // A cancel known to be behind us changes no queue-ahead estimate.
        }
    }

    // Contemporaneous own fill (not lookahead). Proves the order reached the
    // front, so queue-ahead is 0; if the prior estimate was nonzero we missed
    // some depletion, so the estimate is flagged.
    void on_own_fill(OrderId id, Qty filled) {
        const auto it = orders_.find(id);
        if (it == orders_.end() || filled <= 0) return;
        QueueStateSample& s = it->second;
        if (s.queue_ahead_qty > 0) s.queue_ahead_censored = true;
        s.queue_ahead_qty = 0;
        s.filled_qty = static_cast<Qty>(s.filled_qty + filled);
        if (s.filled_qty > s.original_qty) s.filled_qty = s.original_qty;
        s.remaining_qty = static_cast<Qty>(s.original_qty - s.filled_qty);
        s.lifecycle = s.remaining_qty == 0 ? QueueLifecycle::Filled
                                           : QueueLifecycle::PartiallyFilled;
    }

    void on_own_cancel(OrderId id) {
        const auto it = orders_.find(id);
        if (it == orders_.end()) return;
        it->second.lifecycle = QueueLifecycle::Cancelled;
    }

    // Read-only snapshot with age computed against `current_*`. Ages clamp at 0
    // (an order is never observed before it is submitted).
    QueueStateSample sample(OrderId id, EventSeq current_seq,
                            TimeNs current_ts) const {
        const auto it = orders_.find(id);
        if (it == orders_.end()) return QueueStateSample{};
        QueueStateSample s = it->second;
        s.age_events = current_seq > s.submit_seq
                           ? static_cast<EventSeq>(current_seq - s.submit_seq)
                           : 0;
        s.age_ns = current_ts > s.submit_ts
                       ? static_cast<TimeNs>(current_ts - s.submit_ts)
                       : 0;
        return s;
    }

    // Flag every non-terminal order as right-censored at end of observation. No
    // terminal outcome is fabricated; this is the survival-analysis censoring
    // label.
    void finalize_right_censored() {
        for (auto& [id, s] : orders_) {
            (void)id;
            if (!is_terminal(s.lifecycle)) s.right_censored = true;
        }
    }

    const Map& orders() const noexcept { return orders_; }
    std::size_t size() const noexcept { return orders_.size(); }
    const QueueStateSample* find(OrderId id) const {
        const auto it = orders_.find(id);
        return it == orders_.end() ? nullptr : &it->second;
    }

private:
    Map orders_{};
};

} // namespace sim
