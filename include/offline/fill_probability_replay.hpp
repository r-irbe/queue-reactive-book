#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Offline fill-probability / queue-position REPLAY + censoring labeller.
//
// (Wave-I re-council controlling RED gate, 2026-06-19).
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and has NO dependency on
//   `FlatLadderBook`, the matching kernel, or ANY `sim::` symbol -- it includes
//   nothing from `sim/`. It is a PURE CONSUMER of an already-recorded
//   event/queue trace: data flows ONE WAY, OUT of matching and into derived
//   diagnostics. The labels it computes (queue-position-ahead, fill / right-
//   censoring outcome) are observability/calibration targets and may NEVER be
//   fed back as a matching input. This file is the deterministic proof of that
//   one-way contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   Fill probability, queue position, and queue-time are DERIVED/OFFLINE
//   estimates. Given a recorded trace of (submit, aggregate level trade/cancel,
//   own fill/own cancel) events -- already produced by the frozen kernel -- this
//   header reconstructs, with no look-ahead and no kernel handle:
//     * queue-ahead qty per resting order (exact only while not censored);
//     * a RIGHT-CENSORED survival outcome per order (Filled vs censored), where
//       an order that never reaches its original qty is RIGHT-CENSORED, NOT
//       counted as a fill;
//     * a deterministic, order-sensitive FNV-1a-64 replay hash over the labels.
//
// SEMANTIC CONTRACT (the replay-equivalence guarantee):
//   replay_fill_labels(trace) is a PURE function of the trace: replaying an
//   identical trace yields IDENTICAL labels and an IDENTICAL hash; reordering or
//   tampering with the trace changes the hash (order-sensitive integrity). The
//   computation takes the trace by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// FIFO assumption (mirrors the discrete-tick sim, kept self-contained here):
//   aggregate trades consume the FRONT of a price-time queue, so a level trade
//   reduces every tracked order's queue-ahead by the consumed amount; an
//   ambiguous event (unknown-position cancel, a trade that reaches/crosses the
//   order, or an own fill proving the estimate was stale) sets the censor flag.
//   Pro-rata / hidden liquidity / priority-changing replaces are out of scope
//   and, where observable, set the censor flag rather than silently advancing.
// =============================================================================

namespace offline {

// Kinds of recorded trace events. All are OBSERVED outputs of a completed
// matching replay; none is a matching input.
enum class FillEventKind : std::uint8_t {
    Submit = 1,       // own order rests; carries resting qty ahead at submit
    LevelTrade = 2,   // aggregate trade consuming the front of (side, price)
    LevelCancel = 3,  // aggregate cancel at (side, price)
    OwnFill = 4,      // contemporaneous own fill (proves we reached the front)
    OwnCancel = 5,    // own cancel (terminal, possibly after partial fills)
};

enum class FillSide : std::uint8_t { Bid = 0, Ask = 1 };

// One recorded trace event. Pure integer/enum data: no pointer, no engine
// handle, no vtable. (`Submit` uses qty as the resting size and qty_ahead as the
// observed resting qty ahead; `LevelTrade`/`LevelCancel`/`OwnFill` use qty as
// traded/cancelled/filled amount.)
struct FillTraceEvent {
    std::uint64_t seq = 0;
    FillEventKind kind = FillEventKind::Submit;
    FillSide side = FillSide::Bid;
    std::uint64_t order_id = 0;     // Submit / OwnFill / OwnCancel
    std::int64_t price = 0;         // ticks
    std::int64_t qty = 0;           // submit size / traded / cancelled / filled
    std::int64_t qty_ahead = 0;     // Submit only: resting qty ahead at submit
    std::uint64_t ts_ns = 0;        // event-time, ns
    bool position_known = false;    // LevelCancel: is the cancel position known?
    bool ahead = false;             // LevelCancel: known to be ahead of us?
};

// Right-censored survival outcome. A resting order resolves to exactly one.
enum class FillOutcome : std::uint8_t {
    Pending = 0,                 // not yet terminal (transient, never emitted)
    Filled = 1,                  // cumulative fills reached ORIGINAL qty
    RightCensoredCancel = 2,     // cancelled before a full fill
    RightCensoredEndOfData = 3,  // trace ended while still resting (no fill)
};

inline constexpr bool fill_is_terminal(FillOutcome o) noexcept {
    return o != FillOutcome::Pending;
}

// Per-order derived label. Pure data; carries the queue-ahead estimate and the
// censoring-aware survival outcome.
struct FillLabel {
    std::uint64_t order_id = 0;
    FillSide side = FillSide::Bid;
    std::int64_t price = 0;
    std::uint64_t submit_seq = 0;
    std::uint64_t submit_ts = 0;
    std::int64_t original_qty = 0;
    std::int64_t filled_qty = 0;
    std::int64_t queue_ahead_qty = 0;   // exact only while !queue_ahead_censored
    bool queue_ahead_censored = false;
    FillOutcome outcome = FillOutcome::Pending;

    // A fill is a FULL fill within the trace; everything else is right-censored.
    bool is_fill() const noexcept { return outcome == FillOutcome::Filled; }
    bool is_right_censored() const noexcept {
        return outcome == FillOutcome::RightCensoredCancel ||
               outcome == FillOutcome::RightCensoredEndOfData;
    }
};

using ReplayHash = std::uint64_t;

namespace detail {

inline constexpr ReplayHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr ReplayHash kFnvPrime = 1099511628211ULL;

inline constexpr ReplayHash fnv1a_byte(ReplayHash h, std::uint8_t b) noexcept {
    h ^= static_cast<ReplayHash>(b);
    h *= kFnvPrime;
    return h;
}

// Fold a fixed-width value in canonical little-endian order, so the hash does
// not depend on host endianness.
inline constexpr ReplayHash fnv1a_u64(ReplayHash h, std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) {
        h = fnv1a_byte(h, static_cast<std::uint8_t>(v & 0xFFULL));
        v >>= 8;
    }
    return h;
}

inline constexpr ReplayHash fnv1a_i64(ReplayHash h, std::int64_t v) noexcept {
    return fnv1a_u64(h, static_cast<std::uint64_t>(v));
}

}  // namespace detail

// Deterministic, order-sensitive FNV-1a-64 hash over the derived labels. Pure,
// read-only: folds each label's ORDINAL with its field bytes so reordering two
// distinct labels changes the hash; folds the count so truncation/extension
// cannot collide with a prefix.
inline ReplayHash fill_labels_hash(const std::vector<FillLabel>& labels) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    std::uint64_t ordinal = 0;
    for (const FillLabel& l : labels) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, l.order_id);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(l.side));
        h = detail::fnv1a_i64(h, l.price);
        h = detail::fnv1a_u64(h, l.submit_seq);
        h = detail::fnv1a_u64(h, l.submit_ts);
        h = detail::fnv1a_i64(h, l.original_qty);
        h = detail::fnv1a_i64(h, l.filled_qty);
        h = detail::fnv1a_i64(h, l.queue_ahead_qty);
        h = detail::fnv1a_byte(h, l.queue_ahead_censored ? 1u : 0u);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(l.outcome));
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the count
    return h;
}

// Result of a replay: the per-order labels (in submit order, so the layout is
// deterministic and not hash-map-iteration-dependent) plus the replay hash.
struct FillReplayResult {
    std::vector<FillLabel> labels;
    ReplayHash hash = 0;

    std::size_t fill_count() const noexcept {
        std::size_t n = 0;
        for (const FillLabel& l : labels)
            if (l.is_fill()) ++n;
        return n;
    }
    std::size_t right_censored_count() const noexcept {
        std::size_t n = 0;
        for (const FillLabel& l : labels)
            if (l.is_right_censored()) ++n;
        return n;
    }
};

// Replay a recorded trace into right-censored fill/queue-position labels. PURE,
// READ-ONLY, OFFLINE: the trace is taken by const reference and never modified;
// the result is a plain value. No look-ahead -- an order stays `Pending` until a
// terminal condition is OBSERVED, then is finalized as right-censored if the
// trace ends while it is still resting. This cannot mutate, and is not consumed
// by, any matching state.
inline FillReplayResult replay_fill_labels(const std::vector<FillTraceEvent>& trace) {
    FillReplayResult result;
    std::vector<FillLabel>& labels = result.labels;

    // Linear find keeps order deterministic (submit order); fixtures are small.
    auto find_index = [&labels](std::uint64_t id) -> std::size_t {
        for (std::size_t i = 0; i < labels.size(); ++i)
            if (labels[i].order_id == id) return i;
        return labels.size();
    };

    for (const FillTraceEvent& ev : trace) {
        switch (ev.kind) {
            case FillEventKind::Submit: {
                if (find_index(ev.order_id) != labels.size()) break;  // dup submit ignored
                FillLabel l;
                l.order_id = ev.order_id;
                l.side = ev.side;
                l.price = ev.price;
                l.submit_seq = ev.seq;
                l.submit_ts = ev.ts_ns;
                l.original_qty = ev.qty > 0 ? ev.qty : 0;
                l.filled_qty = 0;
                l.queue_ahead_qty = ev.qty_ahead > 0 ? ev.qty_ahead : 0;
                l.queue_ahead_censored = false;
                l.outcome = FillOutcome::Pending;
                labels.push_back(l);
                break;
            }
            case FillEventKind::LevelTrade: {
                if (ev.qty <= 0) break;
                for (FillLabel& l : labels) {
                    if (fill_is_terminal(l.outcome)) continue;
                    if (l.side != ev.side || l.price != ev.price) continue;
                    if (ev.qty >= l.queue_ahead_qty) {
                        if (ev.qty > l.queue_ahead_qty) l.queue_ahead_censored = true;
                        l.queue_ahead_qty = 0;
                    } else {
                        l.queue_ahead_qty -= ev.qty;
                    }
                }
                break;
            }
            case FillEventKind::LevelCancel: {
                for (FillLabel& l : labels) {
                    if (fill_is_terminal(l.outcome)) continue;
                    if (l.side != ev.side || l.price != ev.price) continue;
                    if (!ev.position_known) {
                        // Unknown position (default for real data): censor rather
                        // than silently advance the queue-ahead estimate.
                        l.queue_ahead_censored = true;
                    } else if (ev.ahead && ev.qty > 0) {
                        const std::int64_t take =
                            ev.qty < l.queue_ahead_qty ? ev.qty : l.queue_ahead_qty;
                        l.queue_ahead_qty -= take;
                    }
                    // A cancel known to be behind us changes no estimate.
                }
                break;
            }
            case FillEventKind::OwnFill: {
                const std::size_t i = find_index(ev.order_id);
                if (i == labels.size() || ev.qty <= 0) break;
                FillLabel& l = labels[i];
                if (fill_is_terminal(l.outcome)) break;
                if (l.queue_ahead_qty > 0) l.queue_ahead_censored = true;
                l.queue_ahead_qty = 0;
                l.filled_qty += ev.qty;
                if (l.filled_qty > l.original_qty) l.filled_qty = l.original_qty;
                if (l.filled_qty >= l.original_qty && l.original_qty > 0)
                    l.outcome = FillOutcome::Filled;  // FULL fill only
                break;
            }
            case FillEventKind::OwnCancel: {
                const std::size_t i = find_index(ev.order_id);
                if (i == labels.size()) break;
                FillLabel& l = labels[i];
                if (fill_is_terminal(l.outcome)) break;  // a completed fill wins
                l.outcome = FillOutcome::RightCensoredCancel;  // never a fill
                break;
            }
        }
    }

    // Finalize: any order still resting when the trace ends is RIGHT-CENSORED at
    // end-of-data. No terminal fill outcome is ever fabricated.
    for (FillLabel& l : labels)
        if (!fill_is_terminal(l.outcome))
            l.outcome = FillOutcome::RightCensoredEndOfData;

    result.hash = fill_labels_hash(labels);
    return result;
}

// Read-only convenience wrapper. Holds a const view of an externally owned trace
// and exposes ONLY the replay; it has no mutating member and no matching-engine
// coupling, making the offline / one-way-data-flow boundary explicit at the call
// site. The estimates it returns can be read but never written back.
class FillProbabilityReplay {
public:
    explicit FillProbabilityReplay(const std::vector<FillTraceEvent>& trace) noexcept
        : trace_(&trace) {}

    FillReplayResult replay() const { return replay_fill_labels(*trace_); }
    ReplayHash hash() const { return replay_fill_labels(*trace_).hash; }
    std::size_t size() const noexcept { return trace_->size(); }

private:
    const std::vector<FillTraceEvent>* trace_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline label/trace types do NOT,
// which is the type-level proof that an estimate cannot be wired back into the
// kernel through these structs. A test supplies a positive control (a type that
// DOES expose such a member) to prove the trait actually discriminates.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// Static proof, evaluated at header-include time, that the offline types are
// pure data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => estimates are one-way OUT.
static_assert(std::is_trivially_copyable_v<FillTraceEvent>,
              "FillTraceEvent must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<FillLabel>,
              "FillLabel must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<FillTraceEvent>,
              "FillTraceEvent must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<FillLabel>,
              "FillLabel must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<FillLabel>,
              "FillLabel must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FillTraceEvent>,
              "FillTraceEvent must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FillReplayResult>,
              "FillReplayResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FillProbabilityReplay>,
              "FillProbabilityReplay must not expose a matching-feedback member (one-way data flow)");

}  // namespace offline
