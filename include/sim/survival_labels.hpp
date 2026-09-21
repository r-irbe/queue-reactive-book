#pragma once

#include "event_trace.hpp"   // EventSeq (kernel-free: pulls only types.hpp)
#include "types.hpp"         // Qty, OrderId, TimeNs

#include <cstdint>

namespace sim {

// --- Resting-order fill / survival LABELS (Tier-3 item 14, offline labels) -----
//
// Roadmap item 14 asks for "fill / survival labels" for resting limit orders.
// This scaffold produces a no-look-ahead survival label for ONE resting order
// from the events that touch it, in event-sequence order. It is header-only,
// cold-path, and never feeds the frozen matching kernel (it only LABELS the
// kernel's already-produced event stream).
//
// Terminal states (a resting order resolves to exactly one):
//   Filled            cumulative fills reached the ORIGINAL resting qty within
//                     the horizon window. Partial fills alone do NOT count.
//   CensoredCancel    the order was cancelled (possibly after partial fills)
//                     before being fully filled and before the horizon.
//   CensoredHorizon   the horizon window [start, start+H) closed before a full
//                     fill (event-horizon and/or time-horizon). A fill observed
//                     AT or AFTER the boundary is too late -> CensoredHorizon.
//   CensoredEndOfData the data stream ended while the order was still Pending and
//                     no horizon had been reached. This is finalized EXPLICITLY
//                     and is NOT conflated with CensoredHorizon (data simply ran
//                     out; the horizon was never actually reached).
//
// No look-ahead: the label is `Pending` until a terminal condition is OBSERVED.
// Determinism: at the same event seq, feed fills BEFORE cancels (a same-seq fill
// that completes the order wins, and the trailing cancel is ignored).
//
// The horizon window is HALF-OPEN [start, start+H): an event whose offset
// (seq-start_seq for events, ts-start_ts for time) is >= H is OUTSIDE the window.

enum class SurvivalState : std::uint8_t {
    Pending = 0,
    Filled,
    CensoredCancel,
    CensoredHorizon,
    CensoredEndOfData,
};

struct SurvivalConfig {
    EventSeq start_seq = 0;
    TimeNs start_ts = 0;
    Qty original_qty = 0;           // resting size at submission (precondition: > 0)
    EventSeq horizon_events = 0;    // 0 disables the event-count horizon
    TimeNs horizon_ns = 0;          // 0 disables the time horizon
};

inline bool survival_is_terminal(SurvivalState s) noexcept {
    return s != SurvivalState::Pending;
}

// Single-order, single-pass survival labeller. Feed events in non-decreasing seq
// order; the tracker never inspects future events (no look-ahead).
class SurvivalTracker {
public:
    explicit SurvivalTracker(const SurvivalConfig& cfg) noexcept : cfg_(cfg) {}

    SurvivalState state() const noexcept { return state_; }
    bool terminal() const noexcept { return survival_is_terminal(state_); }
    std::int64_t filled_qty() const noexcept { return filled_; }
    std::int64_t remaining_qty() const noexcept {
        const std::int64_t rem = static_cast<std::int64_t>(cfg_.original_qty) - filled_;
        return rem > 0 ? rem : 0;
    }
    EventSeq resolve_seq() const noexcept { return resolve_seq_; }
    TimeNs resolve_ts() const noexcept { return resolve_ts_; }

    // Apply horizon censoring given the clock has advanced to (seq, ts). Safe to
    // call standalone (e.g. on a heartbeat) or implicitly before fills/cancels.
    void advance(EventSeq seq, TimeNs ts) noexcept {
        if (terminal()) return;
        if (cfg_.horizon_events != 0) {
            const std::uint64_t off = seq >= cfg_.start_seq ? seq - cfg_.start_seq : 0;
            if (off >= cfg_.horizon_events) {
                censor_horizon(cfg_.start_seq + cfg_.horizon_events, ts);
                return;
            }
        }
        if (cfg_.horizon_ns != 0) {
            const std::int64_t off = ts - cfg_.start_ts;  // int64 exact
            if (off >= cfg_.horizon_ns) {
                censor_horizon(seq, cfg_.start_ts + cfg_.horizon_ns);
                return;
            }
        }
    }

    // A (partial or full) fill of `qty` at (seq, ts). Late fills (at/after the
    // horizon boundary) are rejected as CensoredHorizon, never counted.
    void on_fill(Qty qty, EventSeq seq, TimeNs ts) noexcept {
        advance(seq, ts);
        if (terminal()) return;            // censored before this fill landed
        filled_ += static_cast<std::int64_t>(qty);
        if (filled_ >= static_cast<std::int64_t>(cfg_.original_qty)) {
            state_ = SurvivalState::Filled;
            resolve_seq_ = seq;
            resolve_ts_ = ts;
        }
    }

    // A cancel of the resting order at (seq, ts).
    void on_cancel(EventSeq seq, TimeNs ts) noexcept {
        advance(seq, ts);
        if (terminal()) return;            // already filled / horizon-censored
        state_ = SurvivalState::CensoredCancel;
        resolve_seq_ = seq;
        resolve_ts_ = ts;
    }

    // Stream ended. A still-Pending order is censored as end-of-data -- NOT as a
    // horizon censor, because the horizon was never actually reached.
    void finalize_end_of_data(EventSeq last_seq, TimeNs last_ts) noexcept {
        if (terminal()) return;
        state_ = SurvivalState::CensoredEndOfData;
        resolve_seq_ = last_seq;
        resolve_ts_ = last_ts;
    }

private:
    void censor_horizon(EventSeq seq, TimeNs ts) noexcept {
        state_ = SurvivalState::CensoredHorizon;
        resolve_seq_ = seq;
        resolve_ts_ = ts;
    }

    SurvivalConfig cfg_{};
    std::int64_t filled_ = 0;
    SurvivalState state_ = SurvivalState::Pending;
    EventSeq resolve_seq_ = 0;
    TimeNs resolve_ts_ = 0;
};

} // namespace sim
