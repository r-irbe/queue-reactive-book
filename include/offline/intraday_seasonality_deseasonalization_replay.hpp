#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Offline INTRADAY-SEASONALITY DESEASONALIZATION (declared-time-base) REPLAY
// fixture.
//
// which records that "a MOC-level OFFLINE time-base fixture (scenario hash
// changes when the declared time scale / session policy changes ...) is NOT yet
// implemented". This fixture supplies exactly that worked time-scale /
// session-policy scenario-hash falsifier through a concrete, distinct vehicle:
// intraday-seasonality (time-of-day) deseasonalization under a DECLARED diurnal
// time-base. (The MOC's separate exit-ts lifetime-filtration sub-hook is NOT
// claimed by this fixture and remains a distinct, separately routed requirement.)
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel applies explicit orders under deterministic price-time
//   priority only; every quantity computed here (diurnal-bin assignment, per-bin
//   seasonal factor, deseasonalized value, seasonality-strength range, scenario
//   replay hash) is a PURE OFFLINE DIAGNOSTIC over an already-recorded,
//   replayable series. Data flows ONE WAY, OUT of replay and into derived
//   diagnostics; NO seasonal factor, deseasonalized value, bin assignment, or
//   time-base declaration may EVER be fed back as a matching input, a hot-path
//   branch, a fill/priority/accounting decision, or a live trading signal. This
//   header is the deterministic proof of that one-way, kernel-free contract.
//   (`FlatLadderBook` is named only as a prose boundary; never an import target.)
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-time-base-normalization` holds that "every time value must declare its
//   time-base" -- the timestamp CLASS and the session/DST/overnight policy and
//   the time SCALE (physical vs trading-session vs event) -- and that "a
//   normalized nanosecond timestamp is NOT self-sufficient". A diurnal
//   deseasonalization makes this falsifiable in code: the seasonal profile, the
//   deseasonalized series, and the scenario identity are all FUNCTIONS OF THE
//   DECLARED TIME-BASE. This fixture
//     * bins each observation's raw within-session time tick into one of a
//       DECLARED number of time-of-day bins (the declared time SCALE) after a
//       DECLARED session-open phase offset and under a DECLARED clock class,
//       using exact integer arithmetic so the binning is deterministic;
//     * estimates the multiplicative diurnal seasonal profile (per-bin mean over
//       grand mean) and deseasonalizes the series (value / seasonal factor);
//     * exposes a deterministic, order-sensitive FNV-1a-64 SCENARIO hash over the
//       ordered series, the DECLARED time-base policy, the profile, the
//       deseasonalized points, and the diagnostics -- so that CHANGING the
//       declared time scale (bins-per-session / session length) OR the session
//       policy (session-open phase / clock class) changes the scenario hash even
//       on byte-identical raw samples. That is the MOC's falsifier
//       `scenario_hash_changes_when_time_scale_changes`.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   deseasonalize_intraday() is a PURE function of the (immutable) observations
//   and the DECLARED time-base policy: an identical ordered series under an
//   identical declared policy yields an identical profile, identical
//   deseasonalized points, and an identical scenario hash. Reordering the
//   observations leaves the (set-statistic) seasonal PROFILE invariant but
//   changes the ordered deseasonalized sequence and therefore the scenario hash
//   (order-sensitive integrity). Changing ANY declared time-base field changes
//   the scenario hash. Inputs are taken by const reference, the result is a plain
//   value, and no API can mutate or be consumed by matching state.
//
// Deseasonalization background (kept self-contained; no equation imported):
//   Intraday financial activity (volume, |return|, spread) has a pronounced
//   time-of-day shape. A multiplicative seasonal model writes an observation as
//   season(bin) * residual; deseasonalizing divides by the estimated per-bin
//   factor (bin mean / grand mean) to expose the residual. The seasonal profile
//   is a BATCH (offline, full-sample) statistic, NOT a causal live feature; this
//   header neither claims nor relies on event-time no-look-ahead causality -- its
//   load-bearing claim is the declared-time-base sensitivity of the scenario
//   identity, exactly the MOC's project-imposed normalization discipline.
// =============================================================================

namespace offline {

using ReplayHash = std::uint64_t;
using ObsValue = std::int64_t;  // integer observed magnitude (volume, |return|...)
using TimeTick = std::int64_t;  // integer within-day time tick (e.g. ns-of-session)

// The DECLARED timestamp class. Pure metadata: it is committed into the scenario
// hash so two otherwise-identical replays under a different declared clock are
// distinct scenarios. It is never a matching rule.
enum class TimeScaleClass : std::uint8_t {
    SessionClock = 1,  // trading-session-relative time base
    WallClock = 2,     // physical/wall-clock time base
    EventClock = 3,    // transaction/event-count time base
};

// The DECLARED time-base policy. Every field is a declaration the reader must
// make explicit; the binning, profile, deseasonalized series, and scenario hash
// are all functions of it. The diagnostic never adaptively tunes it.
struct TimeBasePolicy {
    TimeScaleClass clock = TimeScaleClass::SessionClock;  // declared class
    TimeTick session_length_ticks = 1;  // declared session span (sanitized to >= 1)
    TimeTick session_open_ticks = 0;    // declared session-open phase offset
    std::size_t bins_per_session = 1;   // declared diurnal grid (sanitized to >= 1)
};

// One replayable observation. Pure integer data: no pointer, no engine handle,
// no vtable. `seq` is the deterministic event ordinal; `tod_tick` is the raw,
// un-normalized within-day time tick; `value` is the observed magnitude.
struct TodObservation {
    std::uint64_t seq = 0;
    TimeTick tod_tick = 0;
    ObsValue value = 0;
};

// Per-bin seasonal profile entry. Pure data.
struct SeasonalBin {
    std::size_t bin = 0;          // diurnal bin index in [0, bins_per_session)
    std::size_t count = 0;        // observations that fell in this bin
    ObsValue sum = 0;             // integer sum of values in this bin
    double bin_mean = 0.0;        // sum / count (0 if empty)
    double seasonal_factor = 1.0; // bin_mean / grand_mean (1 if grand_mean == 0)
};

// One deseasonalized point, emitted in INPUT ORDER. Pure data.
struct DeseasonalizedPoint {
    std::uint64_t seq = 0;
    std::size_t bin = 0;
    ObsValue value = 0;          // original observed value
    double seasonal_factor = 1.0;
    double deseasonalized = 0.0; // value / seasonal_factor (value if factor == 0)
};

// Aggregate offline diagnostics plus the order-sensitive scenario hash. Pure
// value type: copyable, holds no reference to any external (matching) state.
struct DeseasonalizationDiagnostics {
    TimeScaleClass clock = TimeScaleClass::SessionClock;
    std::size_t n = 0;                  // observation count
    std::size_t bins_per_session = 0;   // declared diurnal grid resolution
    std::size_t num_nonempty_bins = 0;  // bins that received at least one obs
    std::size_t total_count = 0;        // total observations binned (== n)
    ObsValue total_sum = 0;             // integer grand total of values
    double grand_mean = 0.0;            // total_sum / total_count (0 if empty)

    double min_factor = 0.0;            // smallest seasonal factor over nonempty bins
    double max_factor = 0.0;            // largest seasonal factor over nonempty bins
    double seasonality_strength = 0.0;  // max_factor - min_factor (0 => no seasonality)

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

// Apply the DECLARED time-base policy with safe, deterministic bounds. The
// declared span and grid are forced to be at least 1 so the binning is always
// well-defined regardless of caller hygiene.
inline TimeTick sane_session_length(const TimeBasePolicy& p) noexcept {
    return p.session_length_ticks > 0 ? p.session_length_ticks : 1;
}
inline std::size_t sane_bins(const TimeBasePolicy& p) noexcept {
    return p.bins_per_session > 0 ? p.bins_per_session : 1;
}

// Map a raw within-day tick to a declared diurnal bin: subtract the declared
// session-open phase, reduce modulo the declared session span with EUCLIDEAN
// modulo (so negative offsets are deterministic and in range), then scale to the
// declared bin grid. The result is always in [0, bins).
inline std::size_t bin_of(const TimeBasePolicy& p, TimeTick tod_tick) noexcept {
    const TimeTick len = sane_session_length(p);
    const std::size_t bins = sane_bins(p);
    TimeTick rel = tod_tick - p.session_open_ticks;
    TimeTick phase = rel % len;
    if (phase < 0) phase += len;  // euclidean: phase in [0, len)
    // floor((phase / len) * bins) in exact integer arithmetic.
    std::size_t bin =
        static_cast<std::size_t>((phase * static_cast<TimeTick>(bins)) / len);
    if (bin >= bins) bin = bins - 1;  // guard the len-boundary rounding case
    return bin;
}

}  // namespace detail

// Result of deseasonalizing a series under a declared time-base: the per-bin
// seasonal profile, the deseasonalized points (input order), and the aggregate
// diagnostics. Held together so a caller can audit the diagnostic.
struct DeseasonalizationReplayResult {
    std::vector<SeasonalBin> profile;            // one entry per declared bin
    std::vector<DeseasonalizedPoint> points;     // one per observation, input order
    DeseasonalizationDiagnostics diagnostics;
};

// Deterministic, order-sensitive FNV-1a-64 SCENARIO hash over the ordered
// observations, the DECLARED time-base policy, the seasonal profile, the
// deseasonalized points, and the aggregate diagnostics. Pure, read-only: folds
// each observation's ORDINAL so reordering changes the hash, folds the count so
// truncation/extension cannot collide with a prefix, and folds EVERY declared
// time-base field so changing the declared time scale or session policy changes
// the scenario hash on byte-identical samples.
inline ReplayHash deseasonalization_scenario_hash(
    const std::vector<TodObservation>& obs, const TimeBasePolicy& policy,
    const std::vector<SeasonalBin>& profile,
    const std::vector<DeseasonalizedPoint>& points,
    const DeseasonalizationDiagnostics& diag) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const TodObservation& o : obs) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, o.seq);
        h = detail::fnv1a_i64(h, o.tod_tick);
        h = detail::fnv1a_i64(h, o.value);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the observation count

    // The DECLARED time-base policy: every field is committed to the scenario.
    h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(policy.clock));
    h = detail::fnv1a_i64(h, policy.session_length_ticks);
    h = detail::fnv1a_i64(h, policy.session_open_ticks);
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(policy.bins_per_session));

    for (const SeasonalBin& b : profile) {
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(b.bin));
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(b.count));
        h = detail::fnv1a_i64(h, b.sum);
        h = detail::fnv1a_i64(h, detail::quantize(b.bin_mean));
        h = detail::fnv1a_i64(h, detail::quantize(b.seasonal_factor));
    }
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(profile.size()));

    for (const DeseasonalizedPoint& p : points) {
        h = detail::fnv1a_u64(h, p.seq);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(p.bin));
        h = detail::fnv1a_i64(h, p.value);
        h = detail::fnv1a_i64(h, detail::quantize(p.seasonal_factor));
        h = detail::fnv1a_i64(h, detail::quantize(p.deseasonalized));
    }
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(points.size()));

    h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(diag.clock));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.n));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.bins_per_session));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.num_nonempty_bins));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.total_count));
    h = detail::fnv1a_i64(h, diag.total_sum);
    h = detail::fnv1a_i64(h, detail::quantize(diag.grand_mean));
    h = detail::fnv1a_i64(h, detail::quantize(diag.min_factor));
    h = detail::fnv1a_i64(h, detail::quantize(diag.max_factor));
    h = detail::fnv1a_i64(h, detail::quantize(diag.seasonality_strength));
    return h;
}

// Deseasonalize a replayable intraday series under a DECLARED time-base policy.
// PURE, READ-ONLY, OFFLINE: every input is taken by const reference and never
// modified; the result is a plain value. The work is a fixed number of linear
// passes (bin, accumulate profile, emit deseasonalized points), so the diagnostic
// is deterministic. This cannot mutate, and is not consumed by, any matching
// state.
inline DeseasonalizationReplayResult deseasonalize_intraday(
    const std::vector<TodObservation>& obs, const TimeBasePolicy& policy) {
    DeseasonalizationReplayResult result;
    const std::size_t bins = detail::sane_bins(policy);

    // Profile slots, one per DECLARED bin (so the profile shape is itself a
    // function of the declared time scale).
    result.profile.resize(bins);
    for (std::size_t i = 0; i < bins; ++i) {
        result.profile[i].bin = i;
    }

    DeseasonalizationDiagnostics& diag = result.diagnostics;
    diag.clock = policy.clock;
    diag.n = obs.size();
    diag.bins_per_session = bins;

    // Pass 1: bin every observation and accumulate the per-bin and grand totals.
    for (const TodObservation& o : obs) {
        const std::size_t b = detail::bin_of(policy, o.tod_tick);
        SeasonalBin& slot = result.profile[b];
        slot.count += 1;
        slot.sum += o.value;
        diag.total_count += 1;
        diag.total_sum += o.value;
    }

    const double grand_mean =
        diag.total_count > 0
            ? static_cast<double>(diag.total_sum) /
                  static_cast<double>(diag.total_count)
            : 0.0;
    diag.grand_mean = grand_mean;

    // Per-bin mean and multiplicative seasonal factor (bin mean / grand mean).
    bool any_nonempty = false;
    for (SeasonalBin& slot : result.profile) {
        if (slot.count == 0) {
            slot.bin_mean = 0.0;
            slot.seasonal_factor = 1.0;  // an empty bin contributes no seasonality
            continue;
        }
        slot.bin_mean =
            static_cast<double>(slot.sum) / static_cast<double>(slot.count);
        slot.seasonal_factor =
            grand_mean != 0.0 ? slot.bin_mean / grand_mean : 1.0;
        diag.num_nonempty_bins += 1;
        if (!any_nonempty) {
            diag.min_factor = slot.seasonal_factor;
            diag.max_factor = slot.seasonal_factor;
            any_nonempty = true;
        } else {
            if (slot.seasonal_factor < diag.min_factor)
                diag.min_factor = slot.seasonal_factor;
            if (slot.seasonal_factor > diag.max_factor)
                diag.max_factor = slot.seasonal_factor;
        }
    }
    diag.seasonality_strength = any_nonempty ? diag.max_factor - diag.min_factor : 0.0;

    // Pass 2: emit deseasonalized points in INPUT ORDER (so the ordered output is
    // order-sensitive even though the set-statistic profile is not).
    result.points.reserve(obs.size());
    for (const TodObservation& o : obs) {
        const std::size_t b = detail::bin_of(policy, o.tod_tick);
        const double factor = result.profile[b].seasonal_factor;
        DeseasonalizedPoint p;
        p.seq = o.seq;
        p.bin = b;
        p.value = o.value;
        p.seasonal_factor = factor;
        p.deseasonalized =
            factor != 0.0 ? static_cast<double>(o.value) / factor
                          : static_cast<double>(o.value);
        result.points.push_back(p);
    }

    diag.replay_hash = deseasonalization_scenario_hash(obs, policy, result.profile,
                                                       result.points, diag);
    return result;
}

// Read-only convenience wrapper. Holds a const view of an externally owned series
// with the DECLARED time-base policy, and exposes ONLY the deseasonalization; it
// has no mutating member and no matching-engine coupling, making the offline /
// one-way-data-flow boundary explicit at the call site. The diagnostics it
// returns can be read but never written back into matching.
class IntradaySeasonalityDeseasonalizationReplay {
public:
    IntradaySeasonalityDeseasonalizationReplay(
        const std::vector<TodObservation>& obs, const TimeBasePolicy& policy) noexcept
        : obs_(&obs), policy_(policy) {}

    DeseasonalizationReplayResult run() const {
        return deseasonalize_intraday(*obs_, policy_);
    }
    DeseasonalizationDiagnostics diagnostics() const { return run().diagnostics; }
    ReplayHash hash() const { return run().diagnostics.replay_hash; }
    std::size_t size() const noexcept { return obs_->size(); }
    bool empty() const noexcept { return obs_->empty(); }
    const TimeBasePolicy& policy() const noexcept { return policy_; }

private:
    const std::vector<TodObservation>* obs_;
    TimeBasePolicy policy_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline diagnostic types do NOT,
// which is the type-level proof that a deseasonalized value or seasonal factor
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

// Static proof, evaluated at header-include time, that the offline data types are
// pure data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the diagnostic is one-way OUT;
//   * run() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<TodObservation>,
              "TodObservation must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<TimeBasePolicy>,
              "TimeBasePolicy must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<SeasonalBin>,
              "SeasonalBin must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<DeseasonalizedPoint>,
              "DeseasonalizedPoint must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<DeseasonalizationDiagnostics>,
              "DeseasonalizationDiagnostics must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<TodObservation>,
              "TodObservation must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<DeseasonalizationDiagnostics>,
              "DeseasonalizationDiagnostics must be standard-layout (no vtable / coupling)");
static_assert(!has_matching_feedback_v<TodObservation>,
              "TodObservation must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<TimeBasePolicy>,
              "TimeBasePolicy must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<SeasonalBin>,
              "SeasonalBin must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<DeseasonalizedPoint>,
              "DeseasonalizedPoint must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<DeseasonalizationDiagnostics>,
              "DeseasonalizationDiagnostics must not expose a matching-feedback member (one-way)");
static_assert(!has_matching_feedback_v<DeseasonalizationReplayResult>,
              "DeseasonalizationReplayResult must not expose a matching-feedback member (one-way)");
static_assert(!has_matching_feedback_v<IntradaySeasonalityDeseasonalizationReplay>,
              "IntradaySeasonalityDeseasonalizationReplay must not expose a matching-feedback member");
static_assert(
    std::is_invocable_r_v<DeseasonalizationReplayResult,
                          decltype(&IntradaySeasonalityDeseasonalizationReplay::run),
                          const IntradaySeasonalityDeseasonalizationReplay&>,
    "run() must be a read-only (const) diagnostic");

}  // namespace offline
