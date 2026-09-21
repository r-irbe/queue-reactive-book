#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Offline arrival / point-process / inter-arrival-time REPLAY fixture.
//
// (the MOC routes Poisson/Hawkes intensities, conditional durations,
// Markov-renewal inter-event times and changepoint timing, all under one
// shared disposition: every such model is an OFFLINE feature / calibration /
// diagnostic over replayed traces, NEVER a matching-kernel arrival generator).
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel applies explicit orders under deterministic price-time
//   priority only and CONSUMES a replayed event stream; it never SYNTHESIZES
//   arrivals. Every quantity computed here (inter-arrival duration, counting
//   process N(t), empirical intensity, and the self-exciting Hawkes intensity
//   recursion) is a PURE OFFLINE DIAGNOSTIC over an already-recorded, replayable
//   event stream. Data flows ONE WAY, OUT of replay and into derived statistics;
//   NO inter-arrival gap, intensity value, excitation state, or counting index
//   may EVER be fed back as a matching input, a hot-path branch, an arrival
//   generator, a fill/priority/accounting decision, or a live trading signal.
//   This header is the deterministic proof of that one-way, kernel-free
//   contract. (`FlatLadderBook` is named here only as a prose boundary; it is
//   never an import target.)
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-offline-arrival-point-process-modelling` holds that arrival-intensity /
//   inter-event-time / point-process models are offline features, calibration
//   targets, or diagnostics -- "never a matching-kernel arrival generator". This
//   fixture demonstrates that the discipline is CONSTRUCTIBLE and REPLAY-NEUTRAL
//   with zero kernel coupling: given a replayable, deterministic stream of events
//   carrying integer event-times (ticks), it
//     * reconstructs the INTER-ARRIVAL DURATIONS (gaps between consecutive event
//       times) and the running COUNTING PROCESS N(t) in exact integer/event-time
//       arithmetic, plus min/max/total/mean gap summaries;
//     * computes the EMPIRICAL INTENSITY (counting-process rate N(T)/T over the
//       observed window);
//     * runs Ogata's EXACT recursion for a self-exciting HAWKES intensity with an
//       exponential kernel whose decay is a per-tick dyadic halving
//       (kernel = 2^{-decay_shift * dt}, i.e. beta = decay_shift * ln 2), so the
//       recursion is bit-exact and the intensity at each event is a hand-checkable
//       dyadic rational lambda_i = mu + alpha * R_i with
//       R_1 = 0, R_i = 2^{-decay_shift*dt_i} * (R_{i-1} + 1);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED event stream, the parameters, and the derived per-event statistics
//       and aggregates, so the diagnostic is reproducible and order-sensitive.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   analyze_arrivals() is a PURE function of the (immutable) event stream and the
//   fixed Hawkes parameters: an identical ordered stream yields identical
//   statistics and an identical hash; reordering the events changes the
//   inter-arrival reconstruction, the intensity path, and the hash (order-
//   sensitive integrity). The computation takes its inputs by const reference,
//   returns a plain value, and exposes no API that could mutate or be consumed by
//   matching state.
//
// Hawkes background (kept self-contained; no equation imported):
//   A self-exciting (Hawkes) point process has conditional intensity
//   lambda(t) = mu + alpha * sum_{t_j < t} exp(-beta (t - t_j)); each event raises
//   the intensity by alpha and the excitation decays exponentially with rate
//   beta, so events cluster. For the exponential kernel the prior-event sum admits
//   the EXACT recursion R_i = exp(-beta * dt_i) * (R_{i-1} + 1) evaluated only at
//   event times (Ogata). Here exp(-beta) is fixed to a dyadic 2^{-decay_shift} so
//   R_i and lambda_i are exactly representable and hand-checkable. NOTE on the
//   boundary: a Hawkes process is, in its native theory, a GENERATIVE arrival
//   mechanism (it produces event times); here it is deliberately re-scoped to an
//   OFFLINE intensity DIAGNOSTIC that only reads a replayed stream, matching the
//   MOC's project-imposed offline-only invariant -- the kernel never generates
//   arrivals.
// =============================================================================

namespace offline {

using ReplayHash = std::uint64_t;
using EventTime = std::int64_t;  // integer event-time / tick

// Which fixed point-process model produced the intensity path. Pure label.
enum class PointProcessKind : std::uint8_t {
    SelfExcitingHawkes = 1,  // exact exponential-kernel Hawkes recursion
};

// One replayable arrival event. Pure integer data: no pointer, no engine handle,
// no vtable. `seq` is the deterministic event ordinal, `time` the integer arrival
// time (tick), `mark` an optional integer type/mark carried for the hash only.
struct ArrivalEvent {
    std::uint64_t seq = 0;
    EventTime time = 0;
    std::int32_t mark = 0;
};

// Fixed Hawkes parameters. They are supplied (not adaptively fitted) so the
// recursion runs deterministically; the analyzer never modifies them. mu is the
// baseline intensity, alpha the self-excitation jump (alpha >= 0), and
// decay_shift the per-tick exponential decay expressed as a dyadic halving count:
// the kernel over a gap dt is 2^{-decay_shift*dt} (beta = decay_shift * ln 2).
struct HawkesParams {
    std::int64_t base_mu = 0;        // mu baseline intensity (integer units)
    std::int64_t excite_alpha = 0;   // alpha self-excitation jump (>= 0)
    std::uint32_t decay_shift = 1;   // beta = decay_shift * ln 2, kernel = 2^{-decay_shift*dt}
};

// Per-event reconstructed statistic. Pure data.
struct EventStat {
    std::size_t index = 0;         // position in the ordered stream
    EventTime time = 0;            // arrival time (tick)
    EventTime interarrival = 0;    // gap from previous event (0 for the first)
    std::size_t count_through = 0; // counting process N at this event = index + 1
    double excitation = 0.0;       // R_i recursion state (sum of prior kernels)
    double intensity = 0.0;        // lambda_i = mu + alpha * R_i
};

// Aggregate offline diagnostics plus the order-sensitive replay hash. Pure value
// type: copyable, holds no reference to any external (matching) state.
struct ArrivalDiagnostics {
    PointProcessKind process = PointProcessKind::SelfExcitingHawkes;
    std::size_t n = 0;                  // event count

    EventTime first_time = 0;
    EventTime last_time = 0;
    EventTime total_span = 0;           // last_time - first_time (integer)

    std::int64_t total_interarrival = 0;  // sum of gaps (== total_span for ordered)
    EventTime min_interarrival = 0;     // smallest gap (0 if < 2 events)
    EventTime max_interarrival = 0;     // largest gap (0 if < 2 events)
    double mean_interarrival = 0.0;     // total_span / (n - 1) (0 if < 2 events)

    double empirical_intensity = 0.0;   // counting-process rate N(T)/T (0 if span <= 0)

    double max_intensity = 0.0;         // peak Hawkes lambda over events
    double final_intensity = 0.0;       // Hawkes lambda at the last event

    ReplayHash replay_hash = 0;
};

namespace detail {

inline constexpr ReplayHash kFnvOffsetBasis = 14695981039346656037ULL;
inline constexpr ReplayHash kFnvPrime = 1099511628211ULL;

inline constexpr ReplayHash fnv1a_byte(ReplayHash h, std::uint8_t b) noexcept {
    h ^= static_cast<ReplayHash>(b);
    h *= kFnvPrime;
    return h;
}

// Fold a fixed-width value in canonical little-endian order, so the hash does not
// depend on host endianness.
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

// Quantize a real metric to a fixed-point integer so it can be folded into the
// integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// Exact dyadic exponential kernel exp(-beta * dt) with beta = decay_shift * ln 2,
// i.e. 2^{-decay_shift*dt}. ldexp by an integer power of two is exact; for an
// out-of-order (negative) gap the exponent is positive (kernel > 1), which is
// fine for the order-sensitivity demonstration and still bit-exact.
inline double hawkes_kernel(std::uint32_t decay_shift, EventTime dt) noexcept {
    const long long exponent =
        -static_cast<long long>(decay_shift) * static_cast<long long>(dt);
    return std::ldexp(1.0, static_cast<int>(exponent));
}

}  // namespace detail

// Result of analyzing an ordered event stream: the per-event reconstructed
// statistics (deterministic, ascending index order) held alongside the aggregate
// diagnostics so a caller can audit the metric.
struct ArrivalReplayResult {
    std::vector<EventStat> events;
    ArrivalDiagnostics diagnostics;
};

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered event stream,
// the fixed Hawkes parameters, the derived per-event statistics, and the
// aggregate metrics. Pure, read-only: folds each event's ORDINAL with its bytes
// so reordering the input changes the hash; folds the count so truncation/
// extension cannot collide with a prefix.
inline ReplayHash arrival_diagnostics_hash(const std::vector<ArrivalEvent>& stream,
                                           const HawkesParams& params,
                                           const std::vector<EventStat>& stats,
                                           const ArrivalDiagnostics& diag) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const ArrivalEvent& e : stream) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, e.seq);
        h = detail::fnv1a_i64(h, e.time);
        h = detail::fnv1a_i64(h, static_cast<std::int64_t>(e.mark));
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the event count

    h = detail::fnv1a_i64(h, params.base_mu);
    h = detail::fnv1a_i64(h, params.excite_alpha);
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(params.decay_shift));

    for (const EventStat& s : stats) {
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(s.index));
        h = detail::fnv1a_i64(h, s.time);
        h = detail::fnv1a_i64(h, s.interarrival);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(s.count_through));
        h = detail::fnv1a_i64(h, detail::quantize(s.excitation));
        h = detail::fnv1a_i64(h, detail::quantize(s.intensity));
    }
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(stats.size()));

    h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(diag.process));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.n));
    h = detail::fnv1a_i64(h, diag.first_time);
    h = detail::fnv1a_i64(h, diag.last_time);
    h = detail::fnv1a_i64(h, diag.total_span);
    h = detail::fnv1a_i64(h, diag.total_interarrival);
    h = detail::fnv1a_i64(h, diag.min_interarrival);
    h = detail::fnv1a_i64(h, diag.max_interarrival);
    h = detail::fnv1a_i64(h, detail::quantize(diag.mean_interarrival));
    h = detail::fnv1a_i64(h, detail::quantize(diag.empirical_intensity));
    h = detail::fnv1a_i64(h, detail::quantize(diag.max_intensity));
    h = detail::fnv1a_i64(h, detail::quantize(diag.final_intensity));
    return h;
}

// Counting process N(t): the number of events in the stream whose arrival time is
// <= the query time `t`. PURE, READ-ONLY: a single linear pass over a const view.
// The stream is assumed ordered (non-decreasing time) for a meaningful counting
// process, but the count is well-defined for any input.
inline std::size_t count_through_time(const std::vector<ArrivalEvent>& stream,
                                      EventTime t) noexcept {
    std::size_t count = 0;
    for (const ArrivalEvent& e : stream) {
        if (e.time <= t) ++count;
    }
    return count;
}

// Analyze a replayable event stream under the fixed Hawkes model. PURE,
// READ-ONLY, OFFLINE: every input is taken by const reference and never modified;
// the result is a plain value. The work is a single linear pass (gap
// reconstruction + counting process + exact Hawkes recursion), so the analysis is
// deterministic. This cannot mutate, and is not consumed by, any matching state.
//
// Inter-arrival convention: gap_i = time_i - time_{i-1} (0 for the first event).
// The stream is taken in REPLAY ORDER; reordering changes the reconstructed gaps,
// the intensity path, and the hash. The Hawkes recursion uses the exact dyadic
// kernel R_1 = 0, R_i = 2^{-decay_shift*gap_i} * (R_{i-1} + 1),
// lambda_i = mu + alpha * R_i.
inline ArrivalReplayResult analyze_arrivals(const std::vector<ArrivalEvent>& stream,
                                            const HawkesParams& params) {
    ArrivalReplayResult result;
    const std::size_t n = stream.size();

    ArrivalDiagnostics& diag = result.diagnostics;
    diag.process = PointProcessKind::SelfExcitingHawkes;
    diag.n = n;

    if (n == 0) {
        diag.replay_hash =
            arrival_diagnostics_hash(stream, params, result.events, diag);
        return result;
    }

    result.events.reserve(n);
    diag.first_time = stream.front().time;
    diag.last_time = stream.back().time;
    diag.total_span = diag.last_time - diag.first_time;

    const double mu = static_cast<double>(params.base_mu);
    const double alpha = static_cast<double>(params.excite_alpha);

    double excitation = 0.0;  // R_i recursion state
    bool gap_seen = false;
    for (std::size_t i = 0; i < n; ++i) {
        EventStat s;
        s.index = i;
        s.time = stream[i].time;
        s.count_through = i + 1;  // counting process N at event i

        if (i == 0) {
            s.interarrival = 0;
            excitation = 0.0;  // R_1 = 0: no prior events
        } else {
            const EventTime gap = stream[i].time - stream[i - 1].time;
            s.interarrival = gap;
            // Exact Ogata recursion: decay the running excitation across the gap
            // and add the previous event's unit jump.
            excitation = detail::hawkes_kernel(params.decay_shift, gap) *
                         (excitation + 1.0);

            diag.total_interarrival += gap;
            if (!gap_seen) {
                diag.min_interarrival = gap;
                diag.max_interarrival = gap;
                gap_seen = true;
            } else {
                if (gap < diag.min_interarrival) diag.min_interarrival = gap;
                if (gap > diag.max_interarrival) diag.max_interarrival = gap;
            }
        }

        s.excitation = excitation;
        s.intensity = mu + alpha * excitation;

        if (i == 0 || s.intensity > diag.max_intensity) {
            diag.max_intensity = s.intensity;
        }
        diag.final_intensity = s.intensity;

        result.events.push_back(s);
    }

    if (n >= 2) {
        diag.mean_interarrival = static_cast<double>(diag.total_span) /
                                 static_cast<double>(n - 1);
    }
    if (diag.total_span > 0) {
        // Counting-process rate N(T)/T over the observed window.
        diag.empirical_intensity =
            static_cast<double>(n) / static_cast<double>(diag.total_span);
    }

    diag.replay_hash =
        arrival_diagnostics_hash(stream, params, result.events, diag);
    return result;
}

// Read-only convenience wrapper. Holds a const view of an externally owned event
// stream with the fixed Hawkes parameters, and exposes ONLY the analysis; it has
// no mutating member and no matching-engine coupling, making the offline /
// one-way-data-flow boundary explicit at the call site. The diagnostics it
// returns can be read but never written back into matching.
class ArrivalPointProcessReplay {
public:
    ArrivalPointProcessReplay(const std::vector<ArrivalEvent>& stream,
                              const HawkesParams& params) noexcept
        : stream_(&stream), params_(params) {}

    ArrivalReplayResult analyze() const {
        return analyze_arrivals(*stream_, params_);
    }
    ArrivalDiagnostics diagnostics() const { return analyze().diagnostics; }
    ReplayHash hash() const { return analyze().diagnostics.replay_hash; }
    std::size_t count_through(EventTime t) const noexcept {
        return count_through_time(*stream_, t);
    }
    std::size_t size() const noexcept { return stream_->size(); }
    bool empty() const noexcept { return stream_->empty(); }

private:
    const std::vector<ArrivalEvent>* stream_;
    HawkesParams params_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline diagnostic types do NOT,
// which is the type-level proof that an intensity / inter-arrival statistic
// cannot be wired back into the kernel through these structs. A test supplies a
// POSITIVE CONTROL (a type that DOES expose such a member) to prove the trait
// actually discriminates and is not vacuously false.
template <typename T, typename = void>
struct has_matching_feedback : std::false_type {};

template <typename T>
struct has_matching_feedback<
    T, std::void_t<decltype(std::declval<T&>().apply_to_matching())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_matching_feedback_v = has_matching_feedback<T>::value;

// Static proof, evaluated at header-include time, that the offline types are pure
// data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * analyze() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<ArrivalEvent>,
              "ArrivalEvent must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<EventStat>,
              "EventStat must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ArrivalDiagnostics>,
              "ArrivalDiagnostics must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<HawkesParams>,
              "HawkesParams must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<ArrivalEvent>,
              "ArrivalEvent must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<ArrivalDiagnostics>,
              "ArrivalDiagnostics must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<ArrivalEvent>,
              "ArrivalEvent must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<EventStat>,
              "EventStat must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ArrivalDiagnostics>,
              "ArrivalDiagnostics must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ArrivalReplayResult>,
              "ArrivalReplayResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ArrivalPointProcessReplay>,
              "ArrivalPointProcessReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<ArrivalReplayResult,
                          decltype(&ArrivalPointProcessReplay::analyze),
                          const ArrivalPointProcessReplay&>,
    "analyze() must be a read-only (const) diagnostic");

}  // namespace offline
