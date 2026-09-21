#pragma once

// =============================================================================
// Offline volatility / return-statistics REPLAY fixture.
//
// by providing the missing deterministic offline diagnostic fixture: a
// deseasonalized-volatility + fixed-GARCH feature-export and a PUCK-style
// variance-scaling / mean-reversion calibration artifact, all carried by a
// diagnostics-on/off-stable replay hash.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and event-driven*; every volatility /
//   return-statistics object computed here (realized variance, deseasonalized
//   realized vol, fixed GARCH(1,1) conditional-variance path, variance-ratio
//   mean-reversion signature) is a PURE OFFLINE feature/label/calibration target
//   over an already-recorded, replayable return series. Data flows ONE WAY, OUT
//   of replay and into derived statistics; a fitted/estimated value may NEVER be
//   fed back as a matching input, a hot-path branch, a price-process mechanic, or
//   production state. This header is the deterministic proof of that one-way,
//   kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-offline-volatility-return-statistics` keeps intraday conditional
//   volatility (GARCH/EGARCH), price-change-duration volatility proxies,
//   stationarity/ergodicity return assumptions, and moving-potential (PUCK)
//   mean-reversion strictly in the offline feature/label/calibration layer. Its
//   open `needs-code-cross-validation` channel asked for a deseasonalized-GARCH
//   feature-export fixture, a variance-scaling PUCK calibration artifact, and a
//   diagnostics-on/off replay-hash artifact. This fixture supplies all three:
//   given a replayable, deterministic ORDERED return series (integer-tick,
//   event-time), it
//     * computes the exact integer raw-moment sums (S1,S2) and the mean,
//       population variance, realized variance (RV = sum r^2) and realized vol
//       -- order-INVARIANT moments;
//     * DESEASONALIZES the realized variance against a FIXED, data-independent
//       diurnal variance profile (divide each bin's squared return by its fixed
//       diurnal factor), emitting deseasonalized RV and vol -- the intraday
//       seasonality feature/label;
//     * runs a FIXED-parameter GARCH(1,1) conditional-variance recursion
//       sigma^2_t = omega + alpha*r_{t-1}^2 + beta*sigma^2_{t-1} with FIXED
//       omega/alpha/beta and a FIXED initial variance, emitting the terminal and
//       path-mean conditional variance -- the order-SENSITIVE volatility forecast
//       feature;
//     * scores a PUCK-style variance-scaling statistic: the k=2 non-overlapping
//       variance ratio VR = Var(2-period) / (2*Var(1-period)), which a random
//       walk leaves at 1, mean reversion pushes below 1, and trending/persistence
//       above 1 -- the order-SENSITIVE mean-reversion signature;
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED series plus the FIXED parameters and the derived statistics, so
//       the diagnostic is reproducible and order-sensitive.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered series and the FIXED
//   parameters: an identical ordered series + parameters yields an identical
//   result and an identical hash; reordering the SAME multiset of returns leaves
//   the marginal moments and the deseasonalized RV unchanged, yet changes the
//   GARCH path, the variance ratio, and the hash (order-sensitive integrity). The
//   computation takes the series by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Volatility / return-statistics background (kept self-contained; no equation
// imported from any rendered source image -- these are the textbook relations
// re-stated here for the fixture):
//   * Realized variance over a window is the sum of squared returns; realized vol
//     is its square root.
//   * Intraday deseasonalization divides each observation's squared return by a
//     fixed diurnal variance factor for its time-of-day bin, removing the U-shape
//     so cross-bin volatilities are comparable.
//   * GARCH(1,1) conditional variance is the affine recursion above; with FIXED
//     coefficients it is a deterministic filter of the ordered return path.
//   * The Lo-MacKinlay variance ratio compares the variance of k-period returns
//     to k times the single-period variance; departures from 1 are the classic
//     variance-scaling / mean-reversion (sub-/super-diffusion) signature.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price change / event-time return

// One replayable return observation. Pure integer/enum data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal/time, `bin`
// is the intraday seasonality bin index (time-of-day bucket), and `ret` is the
// price change in ticks over this event interval (may be negative).
struct ReturnObservation {
    std::uint64_t seq = 0;
    std::uint32_t bin = 0;
    Tick ret = 0;
};

// FIXED, data-independent GARCH(1,1) parameters. Chosen OFFLINE; never derived
// from the very series being filtered in a feedback loop. Pure value type.
struct GarchParams {
    double omega = 0.0;             // unconditional variance floor
    double alpha = 0.0;             // ARCH (squared-return) weight
    double beta = 0.0;              // GARCH (persistence) weight
    double initial_variance = 0.0;  // sigma^2_0 seed (forecast for r_0)
};

// Derived volatility / return statistics plus the order-sensitive replay hash.
// Pure value type: copyable, holds no reference to any external (matching) state.
struct VolatilityReturnStats {
    std::size_t n = 0;

    // Exact integer raw-moment sums over the returns (order-insensitive, exact).
    std::int64_t sum1 = 0;  // sum r
    std::int64_t sum2 = 0;  // sum r^2  (== realized variance in ticks^2)

    // Central moments / realized volatility (deterministic double arithmetic).
    double mean = 0.0;
    double variance = 0.0;           // population variance of returns
    double realized_variance = 0.0;  // RV = sum r^2 (double view of sum2)
    double realized_vol = 0.0;       // sqrt(RV)

    // (1) Intraday-seasonality DESEASONALIZED realized variance / vol: each
    // squared return divided by its bin's FIXED diurnal variance factor, summed.
    // Order-INVARIANT (a pure per-observation rescale-and-sum).
    double deseasonalized_realized_variance = 0.0;
    double deseasonalized_realized_vol = 0.0;

    // (2) FIXED-parameter GARCH(1,1) conditional-variance path. Order-SENSITIVE.
    double garch_terminal_variance = 0.0;  // forecast for the next, unseen return
    double garch_mean_variance = 0.0;      // mean conditional variance over path

    // (3) PUCK-style variance-scaling statistic: k=2 non-overlapping variance
    // ratio VR = Var(2-period) / (2 * Var(1-period)). Order-SENSITIVE.
    bool variance_ratio_valid = false;
    double variance_ratio = 0.0;
    int mean_reversion_sign = 0;  // -1 reverting (VR<1), +1 trending (VR>1), 0 RW/NA

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

// Quantize a real parameter to a fixed-point integer so it can be folded into
// the integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// Deterministic square root via a FIXED Newton iteration count, keeping the
// fixture free of any <cmath> dependency and deterministic regardless of input
// magnitude. Non-positive input maps to 0.
inline double det_sqrt(double v) noexcept {
    if (v <= 0.0) return 0.0;
    double g = (v < 1.0) ? 1.0 : v;  // bounded start
    for (int i = 0; i < 60; ++i) {   // fixed iterations
        g = 0.5 * (g + v / g);
    }
    return g;
}

}  // namespace detail

// Analytic reference relations, used to validate the statistics on closed-form
// cases. Textbook relations, computed here so the fixture is self-checking and
// self-contained.
namespace analytic {

// One FIXED GARCH(1,1) step: next conditional variance from the current one and
// the just-observed return. sigma^2_{t+1} = omega + alpha*r_t^2 + beta*sigma^2_t.
inline double garch_step(const GarchParams& p, double sigma2, double ret) noexcept {
    return p.omega + p.alpha * (ret * ret) + p.beta * sigma2;
}

// Deseasonalized squared return: squared return divided by the FIXED diurnal
// variance factor for its bin (a non-positive factor is treated as 1.0).
inline double deseasonalized_sq_return(double ret, double diurnal_factor) noexcept {
    const double f = (diurnal_factor > 0.0) ? diurnal_factor : 1.0;
    return (ret * ret) / f;
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered observations,
// the FIXED parameters, and the derived statistics. Pure, read-only: folds each
// observation's ORDINAL with its bytes so reordering the input changes the hash;
// folds the FIXED diurnal profile and GARCH parameters so a parameter change is
// detectable; folds the integer moment sums and the quantized derived statistics;
// folds the counts so truncation/extension cannot collide with a prefix.
inline ReplayHash volatility_stats_hash(const std::vector<ReturnObservation>& series,
                                        const std::vector<double>& diurnal_profile,
                                        const GarchParams& garch,
                                        const VolatilityReturnStats& stats) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const ReturnObservation& o : series) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, o.seq);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(o.bin));
        h = detail::fnv1a_i64(h, o.ret);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the series count

    std::uint64_t bin_ordinal = 0;
    for (double f : diurnal_profile) {
        h = detail::fnv1a_u64(h, bin_ordinal);
        h = detail::fnv1a_i64(h, detail::quantize(f));
        ++bin_ordinal;
    }
    h = detail::fnv1a_u64(h, bin_ordinal);  // fold the profile count

    h = detail::fnv1a_i64(h, detail::quantize(garch.omega));
    h = detail::fnv1a_i64(h, detail::quantize(garch.alpha));
    h = detail::fnv1a_i64(h, detail::quantize(garch.beta));
    h = detail::fnv1a_i64(h, detail::quantize(garch.initial_variance));

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(stats.n));
    h = detail::fnv1a_i64(h, stats.sum1);
    h = detail::fnv1a_i64(h, stats.sum2);
    h = detail::fnv1a_i64(h, detail::quantize(stats.mean));
    h = detail::fnv1a_i64(h, detail::quantize(stats.variance));
    h = detail::fnv1a_i64(h, detail::quantize(stats.realized_variance));
    h = detail::fnv1a_i64(h, detail::quantize(stats.realized_vol));
    h = detail::fnv1a_i64(h, detail::quantize(stats.deseasonalized_realized_variance));
    h = detail::fnv1a_i64(h, detail::quantize(stats.deseasonalized_realized_vol));
    h = detail::fnv1a_i64(h, detail::quantize(stats.garch_terminal_variance));
    h = detail::fnv1a_i64(h, detail::quantize(stats.garch_mean_variance));
    h = detail::fnv1a_byte(h, stats.variance_ratio_valid ? 1u : 0u);
    h = detail::fnv1a_i64(h, detail::quantize(stats.variance_ratio));
    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(stats.mean_reversion_sign));
    return h;
}

// Compute the offline volatility / return statistics for a replayable ordered
// return series under a FIXED diurnal profile and FIXED GARCH parameters. PURE,
// READ-ONLY, OFFLINE: the series is taken by const reference and never modified;
// the result is a plain value. The amount of work is FIXED per observation (a few
// linear passes plus constant-size closed forms), so the computation is
// deterministic and never iterates a data-dependent number of times. This cannot
// mutate, and is not consumed by, any matching state.
inline VolatilityReturnStats compute_volatility_return_statistics(
    const std::vector<ReturnObservation>& series,
    const std::vector<double>& diurnal_profile, const GarchParams& garch) {
    VolatilityReturnStats stats;
    stats.n = series.size();

    // Pass 1: exact integer raw-moment sums and the deseasonalized RV.
    for (const ReturnObservation& o : series) {
        const std::int64_t x = o.ret;
        stats.sum1 += x;
        stats.sum2 += x * x;
        const double factor =
            (o.bin < diurnal_profile.size()) ? diurnal_profile[o.bin] : 1.0;
        stats.deseasonalized_realized_variance +=
            analytic::deseasonalized_sq_return(static_cast<double>(x), factor);
    }
    stats.realized_variance = static_cast<double>(stats.sum2);
    stats.realized_vol = detail::det_sqrt(stats.realized_variance);
    stats.deseasonalized_realized_vol =
        detail::det_sqrt(stats.deseasonalized_realized_variance);

    if (stats.n >= 1) {
        const double n = static_cast<double>(stats.n);
        stats.mean = static_cast<double>(stats.sum1) / n;
        const double raw_m2 = static_cast<double>(stats.sum2) / n;
        stats.variance = raw_m2 - stats.mean * stats.mean;
        if (stats.variance < 0.0) stats.variance = 0.0;  // guard FP noise
    }

    // (2) FIXED-parameter GARCH(1,1) conditional-variance recursion. The entering
    // conditional variance `cv` is the forecast for the current return; after
    // observing the return it advances to the forecast for the next one. The
    // terminal value is the forecast for the next, unseen return. ORDER-SENSITIVE.
    {
        double cv = garch.initial_variance;
        double sum_cv = 0.0;
        for (const ReturnObservation& o : series) {
            sum_cv += cv;  // forecast in force for this observation
            cv = analytic::garch_step(garch, cv, static_cast<double>(o.ret));
        }
        stats.garch_terminal_variance = cv;
        stats.garch_mean_variance = (stats.n >= 1)
                                        ? sum_cv / static_cast<double>(stats.n)
                                        : garch.initial_variance;
    }

    // (3) PUCK-style variance-scaling statistic: k=2 non-overlapping variance
    // ratio. Requires an even count >= 4 and non-degenerate single-period
    // variance. ORDER-SENSITIVE (aggregating consecutive returns).
    if (stats.n >= 4 && (stats.n % 2 == 0) && stats.variance > 0.0) {
        const double mean1 = stats.mean;
        double ss1 = 0.0;
        for (const ReturnObservation& o : series) {
            const double d = static_cast<double>(o.ret) - mean1;
            ss1 += d * d;
        }

        const std::size_t m = stats.n / 2;
        double sum_two = 0.0;
        std::vector<double> two_period;
        two_period.reserve(m);
        for (std::size_t j = 0; j < m; ++j) {
            const double y = static_cast<double>(series[2 * j].ret) +
                             static_cast<double>(series[2 * j + 1].ret);
            two_period.push_back(y);
            sum_two += y;
        }
        const double mean2 = sum_two / static_cast<double>(m);
        double ss2 = 0.0;
        for (double y : two_period) {
            const double d = y - mean2;
            ss2 += d * d;
        }

        const double var1 = ss1 / static_cast<double>(stats.n);
        const double var2 = ss2 / static_cast<double>(m);
        if (var1 > 0.0) {
            stats.variance_ratio = var2 / (2.0 * var1);
            stats.variance_ratio_valid = true;
            constexpr double kVrTol = 1e-9;
            if (stats.variance_ratio < 1.0 - kVrTol) {
                stats.mean_reversion_sign = -1;  // sub-diffusive / mean-reverting
            } else if (stats.variance_ratio > 1.0 + kVrTol) {
                stats.mean_reversion_sign = +1;  // super-diffusive / trending
            } else {
                stats.mean_reversion_sign = 0;  // random-walk-like
            }
        }
    }

    stats.replay_hash =
        volatility_stats_hash(series, diurnal_profile, garch, stats);
    return stats;
}

// Read-only convenience wrapper. Holds a const view of an externally owned return
// series, diurnal profile, and FIXED GARCH parameters, and exposes ONLY the
// computation; it has no mutating member and no matching-engine coupling, making
// the offline / one-way-data-flow boundary explicit at the call site. The
// statistics it returns can be read but never written back into matching.
class VolatilityReturnStatisticsReplay {
public:
    VolatilityReturnStatisticsReplay(const std::vector<ReturnObservation>& series,
                                     const std::vector<double>& diurnal_profile,
                                     const GarchParams& garch) noexcept
        : series_(&series), diurnal_profile_(&diurnal_profile), garch_(garch) {}

    VolatilityReturnStats compute() const {
        return compute_volatility_return_statistics(*series_, *diurnal_profile_,
                                                    garch_);
    }
    ReplayHash hash() const { return compute().replay_hash; }
    std::size_t size() const noexcept { return series_->size(); }
    bool empty() const noexcept { return series_->empty(); }

private:
    const std::vector<ReturnObservation>* series_;
    const std::vector<double>* diurnal_profile_;
    GarchParams garch_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline statistics types do NOT,
// which is the type-level proof that an estimated value cannot be wired back into
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

// Static proof, evaluated at header-include time, that the offline types are pure
// data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => statistics are one-way OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<ReturnObservation>,
              "ReturnObservation must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<GarchParams>,
              "GarchParams must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<VolatilityReturnStats>,
              "VolatilityReturnStats must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<ReturnObservation>,
              "ReturnObservation must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<VolatilityReturnStats>,
              "VolatilityReturnStats must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<ReturnObservation>,
              "ReturnObservation must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<VolatilityReturnStats>,
              "VolatilityReturnStats must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<VolatilityReturnStatisticsReplay>,
              "VolatilityReturnStatisticsReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<VolatilityReturnStats,
                          decltype(&VolatilityReturnStatisticsReplay::compute),
                          const VolatilityReturnStatisticsReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
