#pragma once

#include "event_trace.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace sim {

enum class StrategyIntentKind : std::uint8_t {
    SubmitLimit = 0,
    SubmitMarket = 1,
    Cancel = 2,
};

struct StrategyIntent {
    EventSeq seq = 0;
    TimeNs ts = 0;
    StrategyIntentKind kind = StrategyIntentKind::SubmitLimit;
    OrderId order_id = 0;
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;
    Side side = Side::Buy;
    Tick price = 0;
    Qty qty = 0;
};

struct RiskLimits {
    Qty max_abs_inventory = 0;
    Qty max_order_qty = 0;
    std::int64_t max_order_notional = 0;
    Tick min_price = 0;
    Tick max_price = 0;
    std::uint32_t max_messages = 0;
    TimeNs message_window_ns = 0;
    std::int64_t max_loss = 0;
    bool kill_switch = false;
};

struct RiskState {
    Qty inventory = 0;
    std::int64_t realized_loss = 0;
};

enum class RiskRejectReason : std::uint8_t {
    None = 0,
    KillSwitch,
    NonPositiveQuantity,
    OrderTooLarge,
    NotionalTooLarge,
    PriceBelowCollar,
    PriceAboveCollar,
    InventoryLimit,
    MessageRateLimit,
    LossLimit,
};

// Number of RiskRejectReason enumerators (None .. LossLimit), for fixed-size
// per-reason tallies in the audit log.
inline constexpr std::size_t kRiskRejectReasonCount = 10;

inline constexpr std::string_view risk_reject_reason_name(RiskRejectReason reason) noexcept {
    switch (reason) {
    case RiskRejectReason::None: return "none";
    case RiskRejectReason::KillSwitch: return "kill_switch";
    case RiskRejectReason::NonPositiveQuantity: return "non_positive_quantity";
    case RiskRejectReason::OrderTooLarge: return "order_too_large";
    case RiskRejectReason::NotionalTooLarge: return "notional_too_large";
    case RiskRejectReason::PriceBelowCollar: return "price_below_collar";
    case RiskRejectReason::PriceAboveCollar: return "price_above_collar";
    case RiskRejectReason::InventoryLimit: return "inventory_limit";
    case RiskRejectReason::MessageRateLimit: return "message_rate_limit";
    case RiskRejectReason::LossLimit: return "loss_limit";
    }
    return "unknown";
}

struct RiskDecision {
    bool accepted = false;
    RiskRejectReason reject_reason = RiskRejectReason::None;
    TraceEvent event{};
};

inline std::int64_t abs_i64(std::int64_t value) noexcept {
    return value < 0 ? -value : value;
}

inline Qty signed_inventory_delta(Side side, Qty qty) noexcept {
    return is_buy(side) ? qty : static_cast<Qty>(-qty);
}

inline TraceEvent to_trace_event(const StrategyIntent& intent) noexcept {
    TraceEventKind kind = TraceEventKind::SubmitLimit;
    if (intent.kind == StrategyIntentKind::SubmitMarket) {
        kind = TraceEventKind::SubmitMarket;
    } else if (intent.kind == StrategyIntentKind::Cancel) {
        kind = TraceEventKind::Cancel;
    }

    return TraceEvent{.seq = intent.seq,
                      .ts = intent.ts,
                      .kind = kind,
                      .order_id = intent.order_id,
                      .maker_id = 0,
                      .taker_id = 0,
                      .parent_id = intent.parent_id,
                      .agent = intent.agent,
                      .counterparty = 0,
                      .source = intent.source,
                      .side = intent.side,
                      .price = intent.price,
                      .qty = intent.qty,
                      .liquidity = LiquidityFlag::Unknown};
}

class RiskGuard {
public:
    explicit RiskGuard(RiskLimits limits) noexcept : limits_(limits) {}

    RiskDecision evaluate(const StrategyIntent& intent, const RiskState& state) {
        if (limits_.kill_switch) return reject(RiskRejectReason::KillSwitch);
        if (limits_.max_loss > 0 && state.realized_loss > limits_.max_loss) {
            return reject(RiskRejectReason::LossLimit);
        }
        if (!message_allowed(intent.ts)) {
            return reject(RiskRejectReason::MessageRateLimit);
        }

        if (intent.kind != StrategyIntentKind::Cancel) {
            const auto submit_check = validate_submit(intent, state);
            if (!submit_check.accepted) return submit_check;
        }

        record_message(intent.ts);
        return RiskDecision{.accepted = true,
                            .reject_reason = RiskRejectReason::None,
                            .event = to_trace_event(intent)};
    }

    std::uint32_t messages_in_window() const noexcept { return messages_in_window_; }
    TimeNs window_start() const noexcept { return window_start_; }
    const RiskLimits& limits() const noexcept { return limits_; }

private:
    RiskLimits limits_{};
    TimeNs window_start_ = 0;
    std::uint32_t messages_in_window_ = 0;

    static RiskDecision reject(RiskRejectReason reason) noexcept {
        return RiskDecision{.accepted = false, .reject_reason = reason, .event = {}};
    }

    RiskDecision validate_submit(const StrategyIntent& intent,
                                 const RiskState& state) const noexcept {
        if (intent.qty <= 0) return reject(RiskRejectReason::NonPositiveQuantity);
        if (limits_.max_order_qty > 0 && intent.qty > limits_.max_order_qty) {
            return reject(RiskRejectReason::OrderTooLarge);
        }

        if (intent.kind == StrategyIntentKind::SubmitLimit) {
            if (limits_.min_price != limits_.max_price && intent.price < limits_.min_price) {
                return reject(RiskRejectReason::PriceBelowCollar);
            }
            if (limits_.min_price != limits_.max_price && intent.price > limits_.max_price) {
                return reject(RiskRejectReason::PriceAboveCollar);
            }
            const std::int64_t notional =
                abs_i64(static_cast<std::int64_t>(intent.price)) *
                static_cast<std::int64_t>(intent.qty);
            if (limits_.max_order_notional > 0 && notional > limits_.max_order_notional) {
                return reject(RiskRejectReason::NotionalTooLarge);
            }
        }

        const Qty projected = static_cast<Qty>(
            state.inventory + signed_inventory_delta(intent.side, intent.qty));
        if (limits_.max_abs_inventory > 0 &&
            abs_i64(static_cast<std::int64_t>(projected)) > limits_.max_abs_inventory) {
            return reject(RiskRejectReason::InventoryLimit);
        }
        return RiskDecision{.accepted = true};
    }

    bool message_allowed(TimeNs ts) const noexcept {
        if (limits_.max_messages == 0 || limits_.message_window_ns <= 0) return true;
        if (messages_in_window_ == 0) return true;
        if (ts - window_start_ >= limits_.message_window_ns) return true;
        return messages_in_window_ < limits_.max_messages;
    }

    void record_message(TimeNs ts) noexcept {
        if (limits_.max_messages == 0 || limits_.message_window_ns <= 0) return;
        if (messages_in_window_ == 0 || ts - window_start_ >= limits_.message_window_ns) {
            window_start_ = ts;
            messages_in_window_ = 1;
            return;
        }
        ++messages_in_window_;
    }
};

// --- Deterministic risk accept/reject audit (Tier-1, non-kernel) --------------
// Opt-in observer that records every RiskGuard decision so a replayed strategy
// produces a stable, inspectable accept/reject trail with a per-reason
// decomposition. It changes no RiskGuard logic and never feeds the matching
// kernel; it is the closure evidence for stateful-risk replay scenarios.
struct RiskAuditRecord {
    EventSeq seq = 0;
    TimeNs ts = 0;
    StrategyIntentKind kind = StrategyIntentKind::SubmitLimit;
    AgentId agent = 0;
    ParentOrderId parent_id = 0;
    Side side = Side::Buy;
    Qty qty = 0;
    bool accepted = false;
    RiskRejectReason reject_reason = RiskRejectReason::None;
};

inline RiskAuditRecord to_audit_record(const StrategyIntent& intent,
                                       const RiskDecision& decision) noexcept {
    return RiskAuditRecord{.seq = intent.seq,
                           .ts = intent.ts,
                           .kind = intent.kind,
                           .agent = intent.agent,
                           .parent_id = intent.parent_id,
                           .side = intent.side,
                           .qty = intent.qty,
                           .accepted = decision.accepted,
                           .reject_reason = decision.reject_reason};
}

class RiskAuditLog {
public:
    void record(const StrategyIntent& intent, const RiskDecision& decision) {
        ++total_;
        if (decision.accepted) {
            ++accepted_;
        } else {
            ++rejected_;
            reject_counts_[static_cast<std::size_t>(decision.reject_reason)]++;
        }
        records_.push_back(to_audit_record(intent, decision));
    }

    std::uint64_t total() const noexcept { return total_; }
    std::uint64_t accepted() const noexcept { return accepted_; }
    std::uint64_t rejected() const noexcept { return rejected_; }
    std::uint64_t reject_count(RiskRejectReason reason) const noexcept {
        return reject_counts_[static_cast<std::size_t>(reason)];
    }
    const std::vector<RiskAuditRecord>& records() const noexcept { return records_; }

private:
    std::array<std::uint64_t, kRiskRejectReasonCount> reject_counts_{};
    std::vector<RiskAuditRecord> records_{};
    std::uint64_t total_ = 0;
    std::uint64_t accepted_ = 0;
    std::uint64_t rejected_ = 0;
};

} // namespace sim
