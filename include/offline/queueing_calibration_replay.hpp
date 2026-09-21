#pragma once

// =============================================================================
// Offline PH (phase-type) / MAP (Markovian Arrival Process) queueing-calibration
// REPLAY fixture.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and event-driven*; every analytic
//   queueing object computed here (raw moments, PH family/phase fit, MAP
//   arrival-correlation signature) is a PURE OFFLINE artifact over an already-
//   recorded, replayable sample sequence. Data flows ONE WAY, OUT of replay and
//   into derived calibration; a fitted parameter may NEVER be fed back as a
//   matching input, a hot-path branch, or production latency truth. This header
//   is the deterministic proof of that one-way, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-analytic-queueing-theory-calibration` routes PH/MAP/QBD matrix-analytic
//   machinery (Breuer & Baum) to an OFFLINE calibration home and carries, as an
//   AMBER residual gate, "an implemented PH/MAP calibration replay-hash fixture".
//   integer-tick inter-arrival / service samples, it
//     * computes the exact first three raw-moment sums (S1,S2,S3) and the
//       mean / variance / squared coefficient of variation (SCV);
//     * fits a PH family by moment matching -- Erlang-k (hypoexponential,
//       SCV < 1, k = round(1/SCV)), Exponential (SCV == 1), or balanced-means
//       Hyperexponential H2 (SCV > 1) -- with a FIXED, data-independent amount
//       of work so the fit is deterministic;
//     * scores a MAP signature: the lag-1 arrival autocorrelation, which a
//       renewal PH process leaves at zero but a correlated MAP does not (this is
//       samples leaves all PH moments unchanged yet changes the MAP signature);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED input plus the fitted parameters, so the calibration is
//       reproducible and order-sensitive.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   calibrate() is a PURE function of the (immutable) sample sequence: an
//   identical ordered sequence yields an identical fit and an identical hash;
//   reordering the samples changes the MAP signature and the hash (order-
//   sensitive integrity), even though the PH moments are invariant. The
//   computation takes the samples by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// PH / MAP moment background (kept self-contained; no equation imported):
//   For Erlang-k the SCV is exactly 1/k; for Exponential it is 1; for a
//   two-phase Hyperexponential it exceeds 1. The balanced-means H2 fit matches
//   the mean and SCV with branch weights p1,p2 and rates lambda1,lambda2 such
//   that p1/lambda1 == p2/lambda2 (balanced means), which is the standard
//   deterministic closed form. None of this is reused from a rendered source
//   image -- it is the textbook moment relation re-stated here for the fixture.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick / event-time inter-event duration

// PH family identified by moment matching. Pure label; never a matching rule.
enum class PhFamily : std::uint8_t {
    Degenerate = 0,        // fewer than 2 samples, or zero variance (point mass)
    Erlang = 1,            // SCV < 1 (hypoexponential), k = round(1/SCV) phases
    Exponential = 2,       // SCV == 1 (memoryless single phase)
    Hyperexponential = 3,  // SCV > 1 (two-phase, balanced-means H2 fit)
};

// One replayable inter-arrival / service sample. Pure integer data: no pointer,
// no engine handle, no vtable. `seq` is the deterministic event ordinal/time;
// `interval` is the inter-event duration in ticks (>= 0).
struct CalibrationSample {
    std::uint64_t seq = 0;
    Tick interval = 0;
};

// Fitted PH/MAP parameters plus the order-sensitive replay hash. Pure value
// type: copyable, holds no reference to any external (matching) state.
struct PhMapFit {
    std::size_t n = 0;

    // Exact raw-moment sums over the integer samples (order-insensitive, exact).
    std::int64_t sum1 = 0;  // sum x
    std::int64_t sum2 = 0;  // sum x^2
    std::int64_t sum3 = 0;  // sum x^3

    // Central moments / shape (deterministic double arithmetic over the sums).
    double mean = 0.0;
    double variance = 0.0;
    double scv = 0.0;  // squared coefficient of variation = variance / mean^2

    // PH fit.
    PhFamily family = PhFamily::Degenerate;
    int erlang_k = 0;            // fitted phase count when Erlang-like (else 0)
    double rate1 = 0.0;          // primary phase/branch rate (lambda1)
    double rate2 = 0.0;          // H2 second branch rate (else 0)
    double weight1 = 1.0;        // H2 first branch weight p1 (else 1)
    double weight2 = 0.0;        // H2 second branch weight p2 (else 0)

    // MAP signature: lag-1 autocorrelation of the ORDERED arrival sequence. Zero
    // for an i.i.d. renewal (PH) process; non-zero reveals Markovian arrival
    // correlation that the marginal PH moments cannot see. Order-sensitive.
    double map_lag1_autocorr = 0.0;
    bool map_correlated = false;  // |autocorr| exceeds the renewal threshold

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

}  // namespace detail

// Analytic reference moments for known PH families, used to validate the fit on
// closed-form cases. These are textbook moment relations, computed here so the
// fixture is self-checking and self-contained.
namespace analytic {

// Erlang-k with per-phase rate lambda: mean = k/lambda, variance = k/lambda^2,
// SCV = 1/k.
inline constexpr double erlang_mean(int k, double lambda) noexcept {
    return static_cast<double>(k) / lambda;
}
inline constexpr double erlang_variance(int k, double lambda) noexcept {
    return static_cast<double>(k) / (lambda * lambda);
}
inline constexpr double erlang_scv(int k) noexcept {
    return 1.0 / static_cast<double>(k);
}

// Two-phase Hyperexponential H2 (weights p1,p2; rates l1,l2): mean and SCV.
inline constexpr double hyperexp_mean(double p1, double l1, double p2,
                                      double l2) noexcept {
    return p1 / l1 + p2 / l2;
}
inline double hyperexp_scv(double p1, double l1, double p2, double l2) noexcept {
    const double m1 = p1 / l1 + p2 / l2;
    const double m2 = 2.0 * (p1 / (l1 * l1) + p2 / (l2 * l2));  // raw 2nd moment
    return (m2 - m1 * m1) / (m1 * m1);
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered samples and the
// fitted parameters. Pure, read-only: folds each sample's ORDINAL with its bytes
// so reordering the input changes the hash; folds the integer moment sums, the
// PH family/phase fit, and the quantized shape / MAP-signature parameters; folds
// the count so truncation/extension cannot collide with a prefix.
inline ReplayHash queueing_fit_hash(const std::vector<CalibrationSample>& samples,
                                    const PhMapFit& fit) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const CalibrationSample& s : samples) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, s.seq);
        h = detail::fnv1a_i64(h, s.interval);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(fit.n));
    h = detail::fnv1a_i64(h, fit.sum1);
    h = detail::fnv1a_i64(h, fit.sum2);
    h = detail::fnv1a_i64(h, fit.sum3);
    h = detail::fnv1a_byte(h, static_cast<std::uint8_t>(fit.family));
    h = detail::fnv1a_i64(h, static_cast<std::int64_t>(fit.erlang_k));
    h = detail::fnv1a_i64(h, detail::quantize(fit.scv));
    h = detail::fnv1a_i64(h, detail::quantize(fit.rate1));
    h = detail::fnv1a_i64(h, detail::quantize(fit.rate2));
    h = detail::fnv1a_i64(h, detail::quantize(fit.weight1));
    h = detail::fnv1a_i64(h, detail::quantize(fit.weight2));
    h = detail::fnv1a_i64(h, detail::quantize(fit.map_lag1_autocorr));
    h = detail::fnv1a_byte(h, fit.map_correlated ? 1u : 0u);
    return h;
}

namespace detail {

// Integer rounding to nearest, ties away from zero (deterministic).
inline int round_to_int(double v) noexcept {
    return static_cast<int>(v >= 0.0 ? v + 0.5 : v - 0.5);
}

}  // namespace detail

// Calibrate a replayable sample sequence into a PH/MAP fit. PURE, READ-ONLY,
// OFFLINE: the samples are taken by const reference and never modified; the
// result is a plain value. The amount of work is FIXED per sample (two linear
// passes plus a constant-size moment-matching closed form), so the fit is
// deterministic and never iterates a data-dependent number of times. This
// cannot mutate, and is not consumed by, any matching state.
inline PhMapFit calibrate_queueing(const std::vector<CalibrationSample>& samples) {
    PhMapFit fit;
    fit.n = samples.size();

    // Pass 1: exact integer raw-moment sums.
    for (const CalibrationSample& s : samples) {
        const std::int64_t x = s.interval;
        fit.sum1 += x;
        fit.sum2 += x * x;
        fit.sum3 += x * x * x;
    }

    if (fit.n >= 1) {
        const double n = static_cast<double>(fit.n);
        fit.mean = static_cast<double>(fit.sum1) / n;
        const double raw_m2 = static_cast<double>(fit.sum2) / n;
        fit.variance = raw_m2 - fit.mean * fit.mean;
        if (fit.variance < 0.0) fit.variance = 0.0;  // guard FP noise
        if (fit.mean > 0.0) {
            fit.scv = fit.variance / (fit.mean * fit.mean);
        }
    }

    // PH family by moment matching. A small relative tolerance treats SCV close
    // to 1 as memoryless (Exponential).
    constexpr double kScvTol = 1e-9;
    if (fit.n < 2 || fit.variance == 0.0 || fit.mean <= 0.0) {
        fit.family = PhFamily::Degenerate;
        fit.erlang_k = (fit.mean > 0.0) ? 1 : 0;
        fit.weight1 = 1.0;
        fit.rate1 = (fit.mean > 0.0) ? 1.0 / fit.mean : 0.0;
    } else if (fit.scv < 1.0 - kScvTol) {
        // Hypoexponential -> Erlang-k with k = round(1/SCV) (>= 2 here).
        fit.family = PhFamily::Erlang;
        int k = detail::round_to_int(1.0 / fit.scv);
        if (k < 2) k = 2;
        fit.erlang_k = k;
        fit.weight1 = 1.0;
        // Erlang-k mean = k/lambda -> per-phase rate lambda = k/mean.
        fit.rate1 = static_cast<double>(k) / fit.mean;
    } else if (fit.scv <= 1.0 + kScvTol) {
        fit.family = PhFamily::Exponential;
        fit.erlang_k = 1;
        fit.weight1 = 1.0;
        fit.rate1 = 1.0 / fit.mean;  // memoryless rate
    } else {
        // Hyperexponential -> balanced-means H2 matching mean and SCV. Standard
        // deterministic closed form (Tijms): p1 = 0.5*(1 + sqrt((c-1)/(c+1))).
        fit.family = PhFamily::Hyperexponential;
        fit.erlang_k = 1;
        const double c = fit.scv;
        double disc = (c - 1.0) / (c + 1.0);
        if (disc < 0.0) disc = 0.0;
        // sqrt via a FIXED iteration count (Newton), keeping the fit free of any
        // <cmath> dependency and deterministic regardless of input magnitude.
        double root = disc;
        if (root > 0.0) {
            double g = (disc < 1.0) ? 1.0 : disc;  // bounded start
            for (int i = 0; i < 40; ++i) {         // fixed iterations
                g = 0.5 * (g + disc / g);
            }
            root = g;
        }
        const double p1 = 0.5 * (1.0 + root);
        const double p2 = 1.0 - p1;
        fit.weight1 = p1;
        fit.weight2 = p2;
        // Balanced means: p_i/lambda_i = mean/2 -> lambda_i = 2*p_i/mean.
        fit.rate1 = 2.0 * p1 / fit.mean;
        fit.rate2 = 2.0 * p2 / fit.mean;
    }

    // MAP signature: lag-1 autocorrelation of the ORDERED sequence. This is the
    // order-sensitive part of the fit -- a renewal (PH) process leaves it at
    // zero, a correlated MAP does not. Standard sample-autocorrelation form
    // (lag-1 cross-product sum over the total sum of squares), so |r1| <= 1.
    if (fit.n >= 2 && fit.variance > 0.0) {
        const double mean = fit.mean;
        double cross = 0.0;
        for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
            const double a = static_cast<double>(samples[i].interval) - mean;
            const double b = static_cast<double>(samples[i + 1].interval) - mean;
            cross += a * b;
        }
        const double sum_sq = fit.variance * static_cast<double>(fit.n);
        fit.map_lag1_autocorr = cross / sum_sq;
        constexpr double kCorrTol = 1e-6;
        fit.map_correlated =
            (fit.map_lag1_autocorr > kCorrTol) || (fit.map_lag1_autocorr < -kCorrTol);
    }

    fit.replay_hash = queueing_fit_hash(samples, fit);
    return fit;
}

// Read-only convenience wrapper. Holds a const view of an externally owned
// sample sequence and exposes ONLY the calibration; it has no mutating member
// and no matching-engine coupling, making the offline / one-way-data-flow
// boundary explicit at the call site. The fit it returns can be read but never
// written back into matching.
class QueueingCalibrationReplay {
public:
    explicit QueueingCalibrationReplay(
        const std::vector<CalibrationSample>& samples) noexcept
        : samples_(&samples) {}

    PhMapFit calibrate() const { return calibrate_queueing(*samples_); }
    ReplayHash hash() const { return calibrate_queueing(*samples_).replay_hash; }
    std::size_t size() const noexcept { return samples_->size(); }
    bool empty() const noexcept { return samples_->empty(); }

private:
    const std::vector<CalibrationSample>* samples_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline calibration types do NOT,
// which is the type-level proof that a fitted parameter cannot be wired back
// into the kernel through these structs. A test supplies a POSITIVE CONTROL (a
// type that DOES expose such a member) to prove the trait actually
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
//   * no `apply_to_matching` feedback member => the fit is one-way OUT;
//   * calibrate() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<CalibrationSample>,
              "CalibrationSample must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<PhMapFit>,
              "PhMapFit must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<CalibrationSample>,
              "CalibrationSample must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<PhMapFit>,
              "PhMapFit must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<CalibrationSample>,
              "CalibrationSample must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<PhMapFit>,
              "PhMapFit must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<QueueingCalibrationReplay>,
              "QueueingCalibrationReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<PhMapFit, decltype(&QueueingCalibrationReplay::calibrate),
                          const QueueingCalibrationReplay&>,
    "calibrate() must be a read-only (const) diagnostic");

}  // namespace offline
