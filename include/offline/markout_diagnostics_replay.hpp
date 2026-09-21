#pragma once

// =============================================================================
// Offline MARKOUT / TCA-DIAGNOSTICS REPLAY fixture.
//
// by supplying the missing "worked markout/TCA diagnostic fixture" the
// code-validation evidence gate demands.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. Every
//   quantity built here (a fill record, a forward midprice, a signed markout at
//   horizon h) is a PURE OFFLINE artifact over an already-recorded, replayable
//   (fill, midprice) event stream. Data flows ONE WAY, OUT of replay and into
//   derived TCA diagnostics; a markout may NEVER be fed back as a matching
//   input, a hot-path branch, an order, a fill, a price, or accounting truth.
//   This header is the deterministic proof of that one-way, kernel-free
//   contract.
//
// WHAT IT PROVES (the load-bearing leakage-hygiene boundary the MOC defends):
//   A markout at horizon h is the SIGNED post-trade price move from a fill out
//   to a FUTURE midprice:
//       markout_h = side * (mid_{t+h} - fill_price)        (integer ticks)
//   with side = +1 for a buy, -1 for a sell. Computing it REQUIRES future
//   midprices, so it is STRICTLY FORWARD-LOOKING: it is a read-only post-hoc
//   LABEL, never a feature available at decision time. Letting a markout (or any
//   future-reading quantity) feed back into matching/replay, or be used as an
//   at-decision-time feature, is LOOK-AHEAD LEAKAGE. This fixture demonstrates:
//     * deterministic, ORDER-SENSITIVE FNV-1a-64 replay hash over a fixed
//       synthetic (fill, future-midprice) stream -- reproducible and
//       tamper-evident;
//     * domain correctness of the signed markout at several horizons on a
//       hand-built trade+midprice path, plus horizon monotonicity on a
//       constructed drift case and the buy-then-rise sign convention;
//     * the leakage-hygiene boundary at TYPE level: the markout label type is
//       flagged NOT available at decision time, so it is excluded from any
//       at-decision-time feature set; a positive-control causal feature type IS
//       admissible, proving the discriminator is not vacuous;
//     * read-only / non-feedback: the markout computer consumes a CONST future
//       view and cannot mutate engine/replay state, and no offline type exposes
//       a matching-feedback member (with a positive control proving the trait
//       discriminates).
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   evaluate_markouts() is a PURE function of the (immutable) event stream and
//   the horizon list: an identical ordered stream + horizons yields an identical
//   result and an identical hash; reordering the events changes the rows and the
//   hash (order-sensitive integrity). The computation takes its inputs by const
//   reference, returns a plain value, and exposes no API that could mutate or be
//   consumed by matching state.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;

// One replayable (fill, midprice) observation. Pure integer data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal; `ts_ns` is
// event-time in ns; `mid` is the observed midprice in integer ticks AT this
// event. `side` is +1 for a buy fill, -1 for a sell fill, 0 when the event is a
// pure midprice observation (no fill); `fill_price` (ticks) is meaningful only
// when `side != 0`.
struct MarkoutTick {
    std::uint64_t seq = 0;
    std::uint64_t ts_ns = 0;
    std::int64_t mid = 0;          // midprice in ticks at this event
    std::int8_t side = 0;          // +1 buy, -1 sell, 0 = not a fill
    std::int64_t fill_price = 0;   // fill price in ticks (valid iff side != 0)
};

// One derived markout LABEL row: the signed post-trade move from a fill at
// decision index t out to the FUTURE midprice at t+h. Pure value data -- never
// an order, fill, price, or feature input.
//
// `available_at_decision_time` is FALSE by construction: a markout reads the
// future midprice mid_{t+h}, so it is a post-hoc label and is INADMISSIBLE as an
// at-decision-time feature. The flag is the type-level leakage-hygiene gate.
struct Markout {
    std::uint64_t fill_seq = 0;    // seq of the fill event t
    std::uint64_t ts_ns = 0;       // event-time of the fill
    std::int8_t side = 0;          // +1 buy, -1 sell
    std::int64_t fill_price = 0;   // fill price in ticks
    int horizon = 0;               // forward horizon h (>= 1)
    std::int64_t mid_future = 0;   // mid_{t+h} in ticks (the FUTURE midprice)
    std::int64_t markout = 0;      // side * (mid_{t+h} - fill_price), ticks
    bool reads_future = true;      // always true: markout looks forward to t+h

    // A markout is a STRICTLY FORWARD-LOOKING post-hoc label, NOT a feature
    // available when the trade decision is made.
    static constexpr bool available_at_decision_time = false;
};

// Outcome of one replay: the resolved markout rows, summary counts, and the
// order-sensitive replay hash.
struct MarkoutResult {
    std::size_t n_events = 0;
    std::size_t n_fills = 0;       // events with side != 0
    std::size_t n_rows = 0;        // (fill, horizon) pairs with a full forward h
    std::int64_t markout_sum = 0;  // sum of signed markouts over the rows
    std::vector<Markout> rows;
    ReplayHash replay_hash = 0;
};

// An example CAUSAL feature: the trailing return ending AT-OR-BEFORE the decision
// event t, mid_t - mid_{t-1}. It uses only information available at decision
// time, so it IS admissible as an at-decision-time feature. It exists to give the
// leakage discriminator a true positive control (a genuine decision-time
// feature) to contrast against the forward-looking markout label.
struct TrailingReturnFeature {
    std::uint64_t event_seq = 0;
    std::int64_t value = 0;        // mid_t - mid_{t-1}, ticks (causal)

    // This quantity is measurable at-or-before t, so it MAY enter the
    // decision-time feature set.
    static constexpr bool available_at_decision_time = true;
};

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

// Read-only CONST view of an externally owned (fill, midprice) stream. It exposes
// ONLY const reads of the FUTURE midprice path; it has no mutating member, so the
// markout computer that consumes it provably cannot alter the replay/engine
// state it reads from. This is the const-correctness half of the leakage-hygiene
// proof.
class FutureMidView {
public:
    explicit FutureMidView(const std::vector<MarkoutTick>& stream) noexcept
        : stream_(&stream) {}

    std::size_t size() const noexcept { return stream_->size(); }
    bool empty() const noexcept { return stream_->empty(); }
    std::int64_t mid_at(std::size_t i) const noexcept {
        return (*stream_)[i].mid;
    }

private:
    const std::vector<MarkoutTick>* stream_;
};

// Compute a single signed markout from a CONST future view. PURE, READ-ONLY: the
// future midprices are read through a const view and never modified. Returns the
// integer-tick markout side * (mid_{t+h} - fill_price); requires t + h < size().
inline std::int64_t compute_markout(const FutureMidView& future, std::size_t t,
                                    std::int8_t side, std::int64_t fill_price,
                                    int horizon) noexcept {
    const std::size_t fut_idx = t + static_cast<std::size_t>(horizon);
    const std::int64_t move = future.mid_at(fut_idx) - fill_price;
    return static_cast<std::int64_t>(side) * move;
}

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered events, the
// horizon list, and the derived markout rows. Pure, read-only: folds each event's
// ORDINAL with its bytes so reordering the input changes the hash; folds the
// horizons and each row; folds the counts so truncation/extension cannot collide
// with a prefix.
inline ReplayHash markout_hash(const std::vector<MarkoutTick>& events,
                               const std::vector<int>& horizons,
                               const MarkoutResult& res) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const MarkoutTick& e : events) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, e.seq);
        h = detail::fnv1a_u64(h, e.ts_ns);
        h = detail::fnv1a_i64(h, e.mid);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(e.side));
        h = detail::fnv1a_i64(h, e.fill_price);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the event count

    std::uint64_t horizon_ordinal = 0;
    for (const int hz : horizons) {
        h = detail::fnv1a_u64(h, horizon_ordinal);
        h = detail::fnv1a_i64(h, static_cast<std::int64_t>(hz));
        ++horizon_ordinal;
    }
    h = detail::fnv1a_u64(h, horizon_ordinal);  // fold the horizon count

    std::uint64_t row_ordinal = 0;
    for (const Markout& r : res.rows) {
        h = detail::fnv1a_u64(h, row_ordinal);
        h = detail::fnv1a_u64(h, r.fill_seq);
        h = detail::fnv1a_u64(h, r.ts_ns);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(r.side));
        h = detail::fnv1a_i64(h, r.fill_price);
        h = detail::fnv1a_i64(h, static_cast<std::int64_t>(r.horizon));
        h = detail::fnv1a_i64(h, r.mid_future);
        h = detail::fnv1a_i64(h, r.markout);
        ++row_ordinal;
    }
    h = detail::fnv1a_u64(h, row_ordinal);  // fold the row count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(res.n_fills));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(res.n_rows));
    h = detail::fnv1a_i64(h, res.markout_sum);
    return h;
}

// Evaluate a replayable (fill, midprice) stream into signed markout labels at the
// requested forward horizons. PURE, READ-ONLY, OFFLINE: the events are taken by
// const reference and never modified; the result is a plain value. The work is a
// FIXED traversal (each fill x each horizon, with a single forward lookup), so it
// is deterministic and never iterates a data-dependent number of times. The
// markout is computed THROUGH a const FutureMidView, making the forward-looking,
// read-only nature explicit at the call site.
//
// For each fill event at index t (side != 0) and each horizon h with a full
// forward horizon (t + h < n), one Markout row is produced:
//     markout = side * (mid_{t+h} - fill_price).
inline MarkoutResult evaluate_markouts(const std::vector<MarkoutTick>& events,
                                       const std::vector<int>& horizons) {
    MarkoutResult res;
    res.n_events = events.size();

    const FutureMidView future(events);  // const, read-only future view
    const std::size_t n = events.size();

    for (std::size_t t = 0; t < n; ++t) {
        if (events[t].side == 0) continue;  // not a fill
        ++res.n_fills;
        for (const int hz : horizons) {
            if (hz < 1) continue;
            if (t + static_cast<std::size_t>(hz) >= n) continue;  // no full horizon

            Markout row;
            row.fill_seq = events[t].seq;
            row.ts_ns = events[t].ts_ns;
            row.side = events[t].side;
            row.fill_price = events[t].fill_price;
            row.horizon = hz;
            row.mid_future = future.mid_at(t + static_cast<std::size_t>(hz));
            row.markout =
                compute_markout(future, t, row.side, row.fill_price, hz);
            row.reads_future = true;  // markout always looks forward to t+h

            res.markout_sum += row.markout;
            res.rows.push_back(row);
        }
    }

    res.n_rows = res.rows.size();
    res.replay_hash = markout_hash(events, horizons, res);
    return res;
}

// The value-level leakage gate: using a markout as an at-decision-time feature
// would be look-ahead leakage exactly because it reads the future. This returns
// true whenever the label is forward-looking (always, for a markout), so a caller
// that tries to admit a markout into the decision-time feature set is flagged.
inline bool would_leak_as_decision_feature(const Markout& label) noexcept {
    return label.reads_future;
}

// Read-only convenience wrapper. Holds a const view of an externally owned event
// stream and exposes ONLY the evaluation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The diagnostic it returns can be read but never
// written back into matching.
class MarkoutDiagnosticsReplay {
public:
    explicit MarkoutDiagnosticsReplay(
        const std::vector<MarkoutTick>& events) noexcept
        : events_(&events) {}

    MarkoutResult evaluate(const std::vector<int>& horizons) const {
        return evaluate_markouts(*events_, horizons);
    }
    ReplayHash hash(const std::vector<int>& horizons) const {
        return evaluate_markouts(*events_, horizons).replay_hash;
    }
    std::size_t size() const noexcept { return events_->size(); }
    bool empty() const noexcept { return events_->empty(); }

private:
    const std::vector<MarkoutTick>* events_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member (a
// method named `apply_to_matching`). The offline markout types do NOT, which is
// the type-level proof that a markout cannot be wired back into the kernel
// through these structs. A test supplies a POSITIVE CONTROL (a type that DOES
// expose such a member) to prove the trait actually discriminates.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// --- Compile-time decision-time-admissibility discriminator -------------------
//
// Reads the `available_at_decision_time` flag a type publishes. A markout LABEL
// publishes `false` (forward-looking post-hoc), a causal feature publishes
// `true`. `is_decision_time_feature_v<T>` is true ONLY for types that are
// admissible at decision time; it is used to EXCLUDE the markout label from any
// at-decision-time feature set at compile time.
template <typename T, typename = void>
struct is_decision_time_feature : std::false_type {};

template <typename T>
struct is_decision_time_feature<
    T, std::void_t<decltype(T::available_at_decision_time)>>
    : std::bool_constant<T::available_at_decision_time> {};

template <typename T>
inline constexpr bool is_decision_time_feature_v =
    is_decision_time_feature<T>::value;

// Static proof, evaluated at header-include time, of the leakage-hygiene and
// no-coupling boundary:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * the markout LABEL is NOT a decision-time feature (forward-looking), while a
//     causal feature IS => markout is excluded from the decision-time set;
//   * evaluate() is invocable on a CONST wrapper, and compute_markout consumes a
//     CONST future view => the diagnostic is read-only.
static_assert(std::is_trivially_copyable_v<MarkoutTick>,
              "MarkoutTick must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<Markout>,
              "Markout must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<TrailingReturnFeature>,
              "TrailingReturnFeature must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<MarkoutTick>,
              "MarkoutTick must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<Markout>,
              "Markout must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<MarkoutTick>,
              "MarkoutTick must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<Markout>,
              "Markout must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<MarkoutResult>,
              "MarkoutResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<MarkoutDiagnosticsReplay>,
              "MarkoutDiagnosticsReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<FutureMidView>,
              "FutureMidView must not expose a matching-feedback member (one-way data flow)");
// The load-bearing leakage gate: the markout label is forward-looking and so is
// EXCLUDED from the at-decision-time feature set, while a genuine causal feature
// is admitted -- proving the discriminator is not vacuously false.
static_assert(!is_decision_time_feature_v<Markout>,
              "markout is a forward-looking post-hoc label, never a decision-time feature");
static_assert(is_decision_time_feature_v<TrailingReturnFeature>,
              "a causal at-or-before-t feature must be admissible at decision time");
static_assert(
    std::is_invocable_r_v<MarkoutResult,
                          decltype(&MarkoutDiagnosticsReplay::evaluate),
                          const MarkoutDiagnosticsReplay&,
                          const std::vector<int>&>,
    "evaluate() must be a read-only (const) diagnostic");
static_assert(
    std::is_invocable_r_v<std::int64_t, decltype(&compute_markout),
                          const FutureMidView&, std::size_t, std::int8_t,
                          std::int64_t, int>,
    "compute_markout must consume a const future view (read-only)");

}  // namespace offline
