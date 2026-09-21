#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Offline regime-detection / changepoint VALIDATION-METRIC REPLAY fixture.
//
// (the controlling open code channel; the MOC records "No offline regime/
// changepoint diagnostic fixture exists ... no local replay fixture", 2026-06-20).
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel applies explicit orders under deterministic price-time
//   priority only; every quantity computed here (CUSUM statistic, alarm index,
//   detection latency, false-alarm count, missed-detection count) is a PURE
//   OFFLINE DIAGNOSTIC that SCORES a detector against ground-truth labels on an
//   already-recorded, replayable series. Data flows ONE WAY, OUT of replay and
//   into derived validation metrics; NO regime label, changepoint flag, CUSUM
//   stopping rule, or latency/false-alarm score may EVER be fed back as a
//   matching input, a hot-path branch, a fill/priority/accounting decision, or a
//   live trading signal. This header is the deterministic proof of that one-way,
//   kernel-free contract. (`FlatLadderBook` is named here only as a prose
//   boundary; it is never an import target.)
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-regime-detection-changepoint-diagnostics` holds that regime /
//   changepoint / structural-break labels are model-conditioned OFFLINE
//   diagnostics that "require ... a detection-latency metric measured against
//   causal event-time labels" and "must never mutate deterministic replay,
//   matching, fills, accounting, inventory, fees, or any hot-path state". This
//   fixture closes that gate by implementing the detection-latency / false-alarm
//   metric itself: given a replayable, deterministic series of integer samples
//   over event-time with KNOWN injected changepoints, it
//     * runs a simple, FIXED-parameter, two-sided CUSUM mean-shift detector
//       (reference mean mu0, slack k, threshold h) in exact integer arithmetic
//       so the alarm sequence is deterministic and reproducible;
//     * scores the alarms against the ground-truth changepoints to compute the
//       VALIDATION METRICS the MOC routes -- per-changepoint detection LATENCY
//       (samples from the true changepoint to the first alarm in its segment),
//       MISSED-DETECTION count (a changepoint segment with no alarm), and
//       FALSE-ALARM count/rate (alarms in the in-control baseline segment, plus
//       redundant repeat alarms after a segment is already detected);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED series, the detector parameters, and the derived scores, so the
//       diagnostic is reproducible and order-sensitive.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   score_regime_detection() is a PURE function of the (immutable) series, the
//   ground-truth changepoints, and the fixed detector parameters: an identical
//   ordered series yields identical scores and an identical hash; reordering the
//   samples changes the alarm timing and the hash (order-sensitive integrity).
//   The computation takes its inputs by const reference, returns a plain value,
//   and exposes no API that could mutate or be consumed by matching state.
//
// CUSUM background (kept self-contained; no equation imported):
//   The two-sided cumulative-sum detector tracks an upper statistic
//   S_hi <- max(0, S_hi + (x - mu0) - k) and a lower statistic
//   S_lo <- max(0, S_lo + (mu0 - x) - k); the slack k makes the in-control walk
//   drift toward zero while a sustained mean shift accumulates past the decision
//   interval h, at which point an alarm fires and the statistics reset. This is
//   the textbook page/CUSUM recursion re-stated for the fixture (integer form),
//   not reused from a rendered source image. NOTE on the boundary: the optimal-
//   stopping pedigree of CUSUM (quickest detection) is an ONLINE action rule in
//   its native theory; here it is deliberately re-scoped to an OFFLINE scorer,
//   matching the MOC's project-imposed offline-only invariant.
// =============================================================================

namespace offline {

using ReplayHash = std::uint64_t;
using SampleValue = std::int64_t;  // integer event-time observation value

// Which fixed detector produced the alarms. Pure label; never a matching rule.
enum class RegimeDetectorKind : std::uint8_t {
    TwoSidedCusum = 1,  // fixed (mu0, k, h) cumulative-sum mean-shift detector
};

// One replayable observation. Pure integer data: no pointer, no engine handle,
// no vtable. `seq` is the deterministic event ordinal/time; `value` is the
// observed quantity in integer units.
struct RegimeSample {
    std::uint64_t seq = 0;
    SampleValue value = 0;
};

// Fixed detector parameters. They are supplied (not adaptively tuned) so the
// detector runs deterministically; the scorer never modifies them. mu0 is the
// in-control reference mean, slack the allowable per-sample drift (k >= 0), and
// threshold the decision interval (h > 0) past which an alarm fires.
struct CusumParams {
    SampleValue target_mean = 0;  // mu0
    SampleValue slack = 0;        // k (>= 0)
    SampleValue threshold = 1;    // h (> 0)
};

// Which CUSUM arm raised an alarm. Pure label.
enum class CusumArm : std::uint8_t {
    None = 0,
    Upper = 1,  // S_hi crossed h (upward mean shift)
    Lower = 2,  // S_lo crossed h (downward mean shift)
};

// One detector alarm at a sample index. Pure data.
struct RegimeAlarm {
    std::size_t index = 0;   // sample index at which the alarm fired
    std::uint64_t seq = 0;   // event-time ordinal of that sample
    CusumArm arm = CusumArm::None;
};

// Per-changepoint score against ground truth. Pure data.
struct ChangepointScore {
    std::size_t cp_index = 0;      // ground-truth changepoint sample index
    bool detected = false;         // an alarm fired within this changepoint's segment
    std::size_t alarm_index = 0;   // detecting alarm index (valid only if detected)
    std::int64_t latency = 0;      // samples from cp_index to alarm_index (>= 0)
};

// Aggregate offline diagnostics plus the order-sensitive replay hash. Pure value
// type: copyable, holds no reference to any external (matching) state.
struct RegimeDiagnostics {
    RegimeDetectorKind detector = RegimeDetectorKind::TwoSidedCusum;
    std::size_t n = 0;                 // sample count
    std::size_t num_changepoints = 0;  // ground-truth changepoint count (valid)
    std::size_t num_alarms = 0;        // total alarms raised

    std::size_t detected_count = 0;    // changepoints with an in-segment alarm
    std::size_t missed_count = 0;      // changepoints with NO in-segment alarm
    std::size_t false_alarm_count = 0; // baseline-segment + redundant-repeat alarms

    std::int64_t total_latency = 0;    // sum of per-changepoint detection latencies
    double mean_detection_latency = 0.0;  // total_latency / detected_count (0 if none)
    double detection_rate = 0.0;       // detected_count / num_changepoints
    double false_alarm_rate = 0.0;     // false_alarm_count / n (per-sample rate)

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

// Quantize a real metric to a fixed-point integer so it can be folded into the
// integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

}  // namespace detail

// Result of running the fixed CUSUM detector over a series: the ordered alarms
// (deterministic, ascending index order) and the per-changepoint scores. Held
// alongside the aggregate diagnostics so a caller can audit the metric.
struct RegimeReplayResult {
    std::vector<RegimeAlarm> alarms;
    std::vector<ChangepointScore> scores;
    RegimeDiagnostics diagnostics;
};

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered series, the
// fixed detector parameters, the derived alarms/scores, and the aggregate
// metrics. Pure, read-only: folds each sample's ORDINAL with its bytes so
// reordering the input changes the hash; folds the count so truncation/extension
// cannot collide with a prefix.
inline ReplayHash regime_diagnostics_hash(const std::vector<RegimeSample>& series,
                                          const std::vector<std::size_t>& changepoints,
                                          const CusumParams& params,
                                          const std::vector<RegimeAlarm>& alarms,
                                          const std::vector<ChangepointScore>& scores,
                                          const RegimeDiagnostics& diag) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const RegimeSample& s : series) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, s.seq);
        h = detail::fnv1a_i64(h, s.value);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the sample count

    ordinal = 0;
    for (std::size_t cp : changepoints) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(cp));
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the changepoint count

    h = detail::fnv1a_i64(h, params.target_mean);
    h = detail::fnv1a_i64(h, params.slack);
    h = detail::fnv1a_i64(h, params.threshold);

    for (const RegimeAlarm& a : alarms) {
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(a.index));
        h = detail::fnv1a_u64(h, a.seq);
        h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(a.arm));
    }
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(alarms.size()));

    for (const ChangepointScore& c : scores) {
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(c.cp_index));
        h = detail::fnv1a_byte(h, c.detected ? 1u : 0u);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(c.alarm_index));
        h = detail::fnv1a_i64(h, c.latency);
    }
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(scores.size()));

    h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(diag.detector));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.n));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.num_changepoints));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.num_alarms));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.detected_count));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.missed_count));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.false_alarm_count));
    h = detail::fnv1a_i64(h, diag.total_latency);
    h = detail::fnv1a_i64(h, detail::quantize(diag.mean_detection_latency));
    h = detail::fnv1a_i64(h, detail::quantize(diag.detection_rate));
    h = detail::fnv1a_i64(h, detail::quantize(diag.false_alarm_rate));
    return h;
}

namespace detail {

// Run the fixed two-sided CUSUM over the series, appending alarms (ascending
// index) to `out`. Exact integer arithmetic; the statistic resets after each
// alarm (standard signalling CUSUM). Pure: reads the series, writes only `out`.
inline void run_cusum(const std::vector<RegimeSample>& series, const CusumParams& params,
                      std::vector<RegimeAlarm>& out) {
    std::int64_t s_hi = 0;
    std::int64_t s_lo = 0;
    const std::int64_t k = params.slack;
    const std::int64_t h = params.threshold;
    for (std::size_t i = 0; i < series.size(); ++i) {
        const std::int64_t dev = series[i].value - params.target_mean;
        s_hi += dev - k;
        if (s_hi < 0) s_hi = 0;
        s_lo += -dev - k;
        if (s_lo < 0) s_lo = 0;
        if (s_hi > h || s_lo > h) {
            RegimeAlarm a;
            a.index = i;
            a.seq = series[i].seq;
            // Tie-break deterministically: upper arm reported when it (also)
            // crossed; otherwise the lower arm.
            a.arm = (s_hi > h) ? CusumArm::Upper : CusumArm::Lower;
            out.push_back(a);
            s_hi = 0;
            s_lo = 0;
        }
    }
}

// Keep only the valid, strictly increasing, in-range ground-truth changepoints
// (each in (0, n)). Deterministic single pass; invalid/duplicate entries are
// dropped so the score is well-defined regardless of caller hygiene.
inline std::vector<std::size_t> sanitize_changepoints(
    const std::vector<std::size_t>& changepoints, std::size_t n) {
    std::vector<std::size_t> out;
    out.reserve(changepoints.size());
    std::size_t last = 0;  // segment 0 starts at 0; first change must exceed it
    for (std::size_t cp : changepoints) {
        if (cp == 0 || cp >= n) continue;       // baseline / out of range
        if (!out.empty() && cp <= last) continue;  // must be strictly increasing
        out.push_back(cp);
        last = cp;
    }
    return out;
}

}  // namespace detail

// Score a replayable series with KNOWN injected changepoints under the fixed
// CUSUM detector. PURE, READ-ONLY, OFFLINE: every input is taken by const
// reference and never modified; the result is a plain value. The work is a fixed
// number of linear passes (CUSUM run, then alarm/changepoint merge), so the
// scoring is deterministic. This cannot mutate, and is not consumed by, any
// matching state.
//
// Scoring convention (causal, against ground-truth event-time labels):
//   * Segment 0 = [0, cp_1) is the in-control BASELINE; it contains no change to
//     detect, so EVERY alarm in it is a FALSE ALARM.
//   * Segment j (j >= 1) begins exactly at changepoint cp_j and ends at cp_{j+1}
//     (or n). The FIRST alarm in [cp_j, end_j) DETECTS cp_j with latency
//     alarm_index - cp_j; a segment with no alarm is a MISSED detection.
//   * Any alarm that is not the first in a post-change segment is a redundant
//     repeat and counts as a FALSE ALARM.
inline RegimeReplayResult score_regime_detection(
    const std::vector<RegimeSample>& series,
    const std::vector<std::size_t>& changepoints, const CusumParams& params) {
    RegimeReplayResult result;
    const std::size_t n = series.size();

    detail::run_cusum(series, params, result.alarms);

    const std::vector<std::size_t> cps = detail::sanitize_changepoints(changepoints, n);

    // Segment starts: [0, cp_1, ..., cp_m]; segment j spans [start_j, start_{j+1})
    // with the implicit final end n. Segment 0 is the baseline.
    std::vector<std::size_t> starts;
    starts.reserve(cps.size() + 1);
    starts.push_back(0);
    for (std::size_t cp : cps) starts.push_back(cp);

    // Per-changepoint scores (one per post-change segment, i.e. per cp).
    result.scores.reserve(cps.size());
    for (std::size_t cp : cps) {
        ChangepointScore s;
        s.cp_index = cp;
        result.scores.push_back(s);
    }

    RegimeDiagnostics& diag = result.diagnostics;
    diag.detector = RegimeDetectorKind::TwoSidedCusum;
    diag.n = n;
    diag.num_changepoints = cps.size();
    diag.num_alarms = result.alarms.size();

    // Walk alarms in ascending index order, assigning each to its segment.
    std::size_t seg = 0;  // current segment for the alarm under consideration
    for (const RegimeAlarm& a : result.alarms) {
        // Advance to the segment containing this alarm index. starts is sorted;
        // segment seg spans [starts[seg], starts[seg+1]).
        while (seg + 1 < starts.size() && a.index >= starts[seg + 1]) ++seg;

        if (seg == 0) {
            // Baseline segment: no changepoint to detect here.
            ++diag.false_alarm_count;
            continue;
        }
        ChangepointScore& s = result.scores[seg - 1];  // cp for this segment
        if (!s.detected) {
            s.detected = true;
            s.alarm_index = a.index;
            s.latency = static_cast<std::int64_t>(a.index - s.cp_index);
        } else {
            ++diag.false_alarm_count;  // redundant repeat alarm
        }
    }

    for (const ChangepointScore& s : result.scores) {
        if (s.detected) {
            ++diag.detected_count;
            diag.total_latency += s.latency;
        } else {
            ++diag.missed_count;
        }
    }

    if (diag.detected_count > 0) {
        diag.mean_detection_latency =
            static_cast<double>(diag.total_latency) /
            static_cast<double>(diag.detected_count);
    }
    if (diag.num_changepoints > 0) {
        diag.detection_rate = static_cast<double>(diag.detected_count) /
                              static_cast<double>(diag.num_changepoints);
    }
    if (n > 0) {
        diag.false_alarm_rate =
            static_cast<double>(diag.false_alarm_count) / static_cast<double>(n);
    }

    diag.replay_hash = regime_diagnostics_hash(series, cps, params, result.alarms,
                                               result.scores, diag);
    return result;
}

// Read-only convenience wrapper. Holds const views of an externally owned series
// and its ground-truth changepoints, with the fixed detector parameters, and
// exposes ONLY the scoring; it has no mutating member and no matching-engine
// coupling, making the offline / one-way-data-flow boundary explicit at the call
// site. The diagnostics it returns can be read but never written back into
// matching.
class RegimeDetectionDiagnosticsReplay {
public:
    RegimeDetectionDiagnosticsReplay(const std::vector<RegimeSample>& series,
                                     const std::vector<std::size_t>& changepoints,
                                     const CusumParams& params) noexcept
        : series_(&series), changepoints_(&changepoints), params_(params) {}

    RegimeReplayResult score() const {
        return score_regime_detection(*series_, *changepoints_, params_);
    }
    RegimeDiagnostics diagnostics() const { return score().diagnostics; }
    ReplayHash hash() const { return score().diagnostics.replay_hash; }
    std::size_t size() const noexcept { return series_->size(); }
    bool empty() const noexcept { return series_->empty(); }

private:
    const std::vector<RegimeSample>* series_;
    const std::vector<std::size_t>* changepoints_;
    CusumParams params_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline diagnostic types do NOT,
// which is the type-level proof that a regime score cannot be wired back into
// the kernel through these structs. A test supplies a POSITIVE CONTROL (a type
// that DOES expose such a member) to prove the trait actually discriminates and
// is not vacuously false.
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
//   * score() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<RegimeSample>,
              "RegimeSample must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<RegimeAlarm>,
              "RegimeAlarm must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<ChangepointScore>,
              "ChangepointScore must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<RegimeDiagnostics>,
              "RegimeDiagnostics must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<CusumParams>,
              "CusumParams must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<RegimeSample>,
              "RegimeSample must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<RegimeDiagnostics>,
              "RegimeDiagnostics must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<RegimeSample>,
              "RegimeSample must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<RegimeAlarm>,
              "RegimeAlarm must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<ChangepointScore>,
              "ChangepointScore must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<RegimeDiagnostics>,
              "RegimeDiagnostics must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<RegimeReplayResult>,
              "RegimeReplayResult must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<RegimeDetectionDiagnosticsReplay>,
              "RegimeDetectionDiagnosticsReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<RegimeReplayResult,
                          decltype(&RegimeDetectionDiagnosticsReplay::score),
                          const RegimeDetectionDiagnosticsReplay&>,
    "score() must be a read-only (const) diagnostic");

}  // namespace offline
