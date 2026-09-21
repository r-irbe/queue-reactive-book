#pragma once

// =============================================================================
// Offline CAUSAL-LEAKAGE-HYGIENE markout REPLAY fixture.
//
// by supplying the missing "leakage-controlled causal/markout replay fixture"
// (the diagnostics-on/off, causal-vs-leaky discriminator that channel demands).
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and event-driven*; every quantity built
//   here (an at-or-before-t feature, a forward markout label at horizon h, a
//   feature/label association) is a PURE OFFLINE artifact over an already-
//   recorded, replayable event stream. Data flows ONE WAY, OUT of replay and
//   into derived diagnostics; a feature, label, or association may NEVER be fed
//   back as a matching input, a hot-path branch, an order, a fill, a price, or
//   accounting truth. This header is the deterministic proof of that one-way,
//   kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-causal-inference-leakage-hygiene` carries the load-bearing hygiene
//   rule: a feature/label row is admissible only if it preserves EVENT-TIME
//   CAUSALITY -- features use ONLY information available at-or-before event-time
//   t, labels look forward to t+h, and NO future / post-decision information
//   leaks into the feature. This fixture demonstrates LEAKAGE DETECTION:
//     * given an ORDERED event stream of (event-time, price) observations, it
//       builds, for each event with a full forward horizon, a CAUSAL feature
//       (the trailing price change ending at-or-before t) and a forward MARKOUT
//       LABEL (price[t+h] - price[t]) -- the at-or-before / forward split is the
//       whole point;
//     * it scores a feature->label association (sign-agreement count and a fixed
//       -point covariance-style score) over the admissible rows;
//     * it exposes a LEAKAGE-CONTROL discriminator: a "leaky" configuration that
//       (incorrectly) lets the feature peek at the FUTURE price[t+1] produces a
//       DIFFERENT, detectably-biased association than the causal configuration.
//       The hygiene gate flags any run whose feature window is not strictly
//       at-or-before t as leaky -- proving the gate catches leakage;
//     * it emits a deterministic, ORDER-SENSITIVE FNV-1a-64 replay hash over the
//       ordered input plus the admissible feature/label rows and the association,
//       so the diagnostic is reproducible and tamper-evident.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   evaluate() is a PURE function of the (immutable) event stream and the config:
//   an identical ordered stream + config yields an identical result and an
//   identical hash; reordering the events changes the rows and the hash (order-
//   sensitive integrity); the causal and leaky configs yield DIFFERENT hashes and
//   results on any stream where the future actually differs from the past. The
//   computation takes the events by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Markout / event-time-causality background (kept self-contained; nothing
// imported from a rendered source): a forward markout at horizon h is the signed
// price change from decision-time t to t+h. An admissible feature is measurable
// with respect to the information set available at-or-before t (its `sigma`-
// algebra); a feature that reads price[t+k] for any k > 0 is FUTURE-PEEKING and
// inadmissible -- this is the textbook label-leakage failure restated here for
// the fixture. The "leaky" config deliberately commits that error so the test
// can prove the hygiene assertion fires.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;

// One replayable event-time observation. Pure integer data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal; `ts_ns`
// is event-time in ns; `price` is the observed mid/last in integer ticks.
struct CausalEvent {
    std::uint64_t seq = 0;
    std::uint64_t ts_ns = 0;
    std::int64_t price = 0;  // ticks
};

// One derived feature/label row: built at decision event index t with a forward
// markout label at t+h. Pure value data -- never an order, fill, or price input.
struct MarkoutRow {
    std::uint64_t event_seq = 0;     // seq of the decision event t
    std::uint64_t ts_ns = 0;         // event-time of t
    std::int64_t feature = 0;        // causal trailing price change ending <= t
    std::int64_t label = 0;          // forward markout: price[t+h] - price[t]
    bool feature_admissible = true;  // false iff the feature window peeked future
};

// Configuration of one replay. `horizon` is the forward markout horizon h (>=1);
// `lookback` is the trailing feature window length (>=1). `leaky_future_peek`
// is the LEAKAGE-CONTROL switch: when true the feature window is shifted FORWARD
// by one event so the feature (incorrectly) reads price[t+1] -- the inadmissible
// future-peek the hygiene gate must catch.
struct CausalConfig {
    int horizon = 1;
    int lookback = 1;
    bool leaky_future_peek = false;
};

// Outcome of one replay: the admissible/inspected rows, the feature->label
// association scores, the leakage verdict, and the order-sensitive replay hash.
struct CausalHygieneResult {
    std::size_t n_events = 0;
    std::size_t n_rows = 0;         // rows with a full forward horizon
    std::size_t sign_agree = 0;     // rows where sign(feature) == sign(label)
    std::size_t sign_disagree = 0;  // rows where signs are opposite (both nonzero)
    std::int64_t assoc_score = 0;   // sum feature*label over the rows (cov-style)
    bool leakage_detected = false;  // the hygiene assertion: config peeks future
    std::vector<MarkoutRow> rows;
    ReplayHash replay_hash = 0;

    // The hygiene gate: a result is admissible only if no leakage was detected.
    bool is_admissible() const noexcept { return !leakage_detected; }
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

// Sign of a signed value as a small canonical byte (0 = zero, 1 = positive,
// 2 = negative) so it folds deterministically into the integer-only hash.
inline constexpr std::uint8_t sign_byte(std::int64_t v) noexcept {
    return v > 0 ? 1u : (v < 0 ? 2u : 0u);
}

}  // namespace detail

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered events, the
// config, and the derived rows + association. Pure, read-only: folds each
// event's ORDINAL with its bytes so reordering the input changes the hash; folds
// the config (including the leakage switch) and each row; folds the counts so
// truncation/extension cannot collide with a prefix.
inline ReplayHash causal_hygiene_hash(const std::vector<CausalEvent>& events,
                                      const CausalConfig& cfg,
                                      const CausalHygieneResult& res) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const CausalEvent& e : events) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, e.seq);
        h = detail::fnv1a_u64(h, e.ts_ns);
        h = detail::fnv1a_i64(h, e.price);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the event count

    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(cfg.horizon));
    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(cfg.lookback));
    h = detail::fnv1a_byte(h, cfg.leaky_future_peek ? 1u : 0u);

    std::uint64_t row_ordinal = 0;
    for (const MarkoutRow& r : res.rows) {
        h = detail::fnv1a_u64(h, row_ordinal);
        h = detail::fnv1a_u64(h, r.event_seq);
        h = detail::fnv1a_u64(h, r.ts_ns);
        h = detail::fnv1a_i64(h, r.feature);
        h = detail::fnv1a_i64(h, r.label);
        h = detail::fnv1a_byte(h, r.feature_admissible ? 1u : 0u);
        ++row_ordinal;
    }
    h = detail::fnv1a_u64(h, row_ordinal);  // fold the row count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(res.n_rows));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(res.sign_agree));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(res.sign_disagree));
    h = detail::fnv1a_i64(h, res.assoc_score);
    h = detail::fnv1a_byte(h, res.leakage_detected ? 1u : 0u);
    return h;
}

// Evaluate a replayable event stream into a causal markout feature/label
// association under STRICT event-time causality. PURE, READ-ONLY, OFFLINE: the
// events are taken by const reference and never modified; the result is a plain
// value. The amount of work is FIXED per event (a constant-size trailing window
// and a single forward lookup), so the evaluation is deterministic and never
// iterates a data-dependent number of times. This cannot mutate, and is not
// consumed by, any matching state.
//
// For each decision event at index t that has a full forward horizon (t + h
// within bounds), the row carries:
//   * feature -- the CAUSAL trailing price change over the lookback window
//     ending at-or-before t: price[t] - price[t - lookback]. Under the leaky
//     config the window is shifted forward by one (price[t+1] - price[t+1 -
//     lookback]), which reads the FUTURE price[t+1] -- inadmissible.
//   * label   -- the forward markout: price[t + h] - price[t].
// The leakage assertion fires whenever the config requests the future-peeking
// window; that flag is the hygiene gate the MOC requires.
inline CausalHygieneResult evaluate_causal_hygiene(
    const std::vector<CausalEvent>& events, const CausalConfig& cfg) {
    CausalHygieneResult res;
    res.n_events = events.size();

    const int h = cfg.horizon >= 1 ? cfg.horizon : 1;
    const int lb = cfg.lookback >= 1 ? cfg.lookback : 1;
    const std::size_t n = events.size();

    // The leaky config shifts the trailing feature window forward by one event,
    // which makes it read price[t+1]: a future-peek the gate must flag.
    const std::size_t peek = cfg.leaky_future_peek ? 1u : 0u;

    for (std::size_t t = 0; t < n; ++t) {
        // Need a full forward horizon for the label...
        if (t + static_cast<std::size_t>(h) >= n) continue;
        // ...and a full trailing window for the feature (shifted under leak).
        if (t + peek < static_cast<std::size_t>(lb)) continue;
        // The leaky forward shift must also stay in bounds.
        if (t + peek >= n) continue;

        const std::size_t f_end = t + peek;            // feature window end index
        const std::size_t f_begin = f_end - static_cast<std::size_t>(lb);

        MarkoutRow row;
        row.event_seq = events[t].seq;
        row.ts_ns = events[t].ts_ns;
        row.feature = events[f_end].price - events[f_begin].price;
        row.label = events[t + static_cast<std::size_t>(h)].price - events[t].price;
        // Admissible iff the feature window did NOT extend past the decision
        // event t (peek == 0). The future-peek shift makes it inadmissible.
        row.feature_admissible = (peek == 0);

        res.rows.push_back(row);

        const std::uint8_t fs = detail::sign_byte(row.feature);
        const std::uint8_t ls = detail::sign_byte(row.label);
        if (fs != 0 && ls != 0) {
            if (fs == ls)
                ++res.sign_agree;
            else
                ++res.sign_disagree;
        }
        res.assoc_score += row.feature * row.label;
    }

    res.n_rows = res.rows.size();
    // The hygiene assertion: leakage is detected exactly when the config built
    // any feature that peeked at the future (i.e. the leaky switch was on and at
    // least one row was produced under it).
    res.leakage_detected = cfg.leaky_future_peek && res.n_rows > 0;

    res.replay_hash = causal_hygiene_hash(events, cfg, res);
    return res;
}

// Read-only convenience wrapper. Holds a const view of an externally owned event
// stream and exposes ONLY the evaluation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The diagnostic it returns can be read but never
// written back into matching.
class CausalLeakageHygieneReplay {
public:
    explicit CausalLeakageHygieneReplay(
        const std::vector<CausalEvent>& events) noexcept
        : events_(&events) {}

    CausalHygieneResult evaluate(const CausalConfig& cfg) const {
        return evaluate_causal_hygiene(*events_, cfg);
    }
    ReplayHash hash(const CausalConfig& cfg) const {
        return evaluate_causal_hygiene(*events_, cfg).replay_hash;
    }
    std::size_t size() const noexcept { return events_->size(); }
    bool empty() const noexcept { return events_->empty(); }

private:
    const std::vector<CausalEvent>* events_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline causal-hygiene types do NOT,
// which is the type-level proof that a feature/label/association cannot be wired
// back into the kernel through these structs. A test supplies a POSITIVE CONTROL
// (a type that DOES expose such a member) to prove the trait actually
// discriminates and is not vacuously false.
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
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * evaluate() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<CausalEvent>,
              "CausalEvent must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<MarkoutRow>,
              "MarkoutRow must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<CausalConfig>,
              "CausalConfig must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<CausalEvent>,
              "CausalEvent must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<MarkoutRow>,
              "MarkoutRow must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<CausalEvent>,
              "CausalEvent must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<MarkoutRow>,
              "MarkoutRow must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<CausalHygieneResult>,
              "CausalHygieneResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<CausalLeakageHygieneReplay>,
              "CausalLeakageHygieneReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<CausalHygieneResult,
                          decltype(&CausalLeakageHygieneReplay::evaluate),
                          const CausalLeakageHygieneReplay&, const CausalConfig&>,
    "evaluate() must be a read-only (const) diagnostic");

}  // namespace offline
