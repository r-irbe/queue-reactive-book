#pragma once

// =============================================================================
// Offline MARKET-IMPACT-LAW diagnostic / estimation replay fixture.
//
// Deterministic, replay-neutral fixture: a (signed-volume,
// realized-price-move) sample is reduced to a FITTED square-root / concave
// impact-law statistic -- a post-hoc TCA diagnostic -- without ever touching
// matching. It is IMPLEMENTATION EVIDENCE ONLY (boundary-constructibility), NOT a
// proof that any impact law is the true market mechanism.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and price/size/time only*; every quantity
//   squares / R^2 goodness of fit, a per-share concavity/monotonicity signature,
//   a sequential cumulative-impact path peak) is a PURE OFFLINE diagnostic LABEL
//   over an already-recorded, replayable trade-and-impact sample. Data flows ONE
//   WAY, OUT of the sample and into derived statistics; a fitted/estimated value
//   may NEVER be fed back as a matching input, a fill/priority/tie-break rule, a
//   fee/inventory/accounting mutation, a replay-hash perturbation of the engine,
//   or a hot-path branch. This header is the deterministic proof of that one-way,
//   kernel-free contract.
//
// HONEST SCOPE (keep visible):
//   The fixture validates the DIAGNOSTIC / ESTIMATION DISCIPLINE: an impact law
//   is a FITTED STATISTIC on observed (signed-volume, price-move) data, evaluated
//   POST HOC. It proves the estimator is constructible, deterministic, and
//   replay-neutral, and that it recovers a KNOWN coefficient on a hand-built
//   synthetic dataset. It does NOT prove that the square-root law (or any
//   particular concave power law) is the TRUE price-formation mechanism: the
//   square-root regularity is an empirically-supported but contested STYLIZED
//   FACT, not a theorem. Nothing here is read back into matching/replay.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-market-impact-laws` keeps every market-impact law (square-root
//   functional form, propagator decay, no-arbitrage admissibility, duration/
//   volatility attribution) strictly in the offline TCA / calibration /
//   diagnostics layer, never inside `FlatLadderBook`. Its open
//   `needs-code-validation` channel asked for a worked, replay-neutral example
//   where a (signed-volume, price-move) sample is reduced to a fitted impact-law
//   statistic under a diagnostics-on/off replay hash. This fixture supplies
//   exactly that. Given an ORDERED sample of trades (signed volume q, average
//   daily volume V, volatility proxy sigma, realized price move y), it
//     * counts EXACT integer buy/sell splits and the total traded |q| -- the
//       order-INVARIANT sample aggregates;
//     * builds the square-root-law predictor x_i = sign(q_i)*sigma_i*sqrt(|q_i|/V_i)
//       and FITS the dimensionless coefficient Y by least-squares THROUGH THE
//       squares and an R^2 goodness of fit -- the order-INVARIANT estimation
//       statistic (on a noise-free synthetic set with |q|/V a perfect square the
//       arithmetic is exact and Y_hat recovers the generating coefficient);
//     * tracks the SEQUENTIAL cumulative realized-impact path and its peak
//       absolute excursion along the sample order -- the order-SENSITIVE path
//       metric (same multiset, different order => different peak, same fit);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED sample plus the derived statistics, so the diagnostic is
//       reproducible and order-sensitive.
//   Two closed-form OFFLINE diagnostic relations are provided as directly-testable
//   `analytic` helpers, faithful to the MOC's routed square-root law:
//     * the square-root impact predictor itself, impact(q) ~ sign(q)*Y*sigma*
//       sqrt(|q|/V) -- a concave, sub-linear function of size;
//     * its per-share impact Y*sigma/sqrt(|q|*V), which is STRICTLY DECREASING in
//       trade size (the concavity / diminishing-marginal-impact signature), and
//       the central second difference of impact(q) in size, which is NEGATIVE
//       (impact is CONCAVE in q) -- the hand-checkable monotonicity + concavity
//       diagnostics.
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered sample: an identical
//   ordered sample yields an identical result and an identical hash; reordering
//   the SAME multiset of trades leaves the buy/sell split, the traded volume, and
//   the least-squares fit (Y_hat, RSS, R^2) unchanged, yet changes the sequential
//   cumulative-impact peak and the hash (order-sensitive integrity). The
//   computation takes the sample by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Market-impact-law background (kept self-contained; no equation imported from
// any rendered source image -- these are stylized textbook regularities
// re-stated here for the fixture, and are empirically supported yet contested,
// NOT theorems):
//   * The square-root law (Bouchaud, Toth et al.) states that the AGGREGATE
//     impact of a metaorder of size Q scales as Y*sigma*sqrt(Q/V): concave,
//     sub-linear, with a dimensionless Y of order one. It is a stylized empirical
//     regularity, not a proven mechanism.
//   * Fitting Y is a least-squares regression of realized impact on the predictor
//     sign(q)*sigma*sqrt(|q|/V); through the origin, Y_hat = sum(x y)/sum(x^2).
//   * A concave impact law has per-share impact DECREASING in size and a NEGATIVE
//     second difference in size -- the diminishing-marginal-impact signature that
//     distinguishes square-root/concave laws from a linear (Kyle-lambda) law.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price move / volatility unit

// One replayable trade-and-impact observation. Pure integer data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal/time,
// `signed_volume` is the signed trade size q (buy +, sell -), `adv` is the
// average daily volume V (> 0), `sigma_ticks` is the volatility proxy sigma, and
// `observed_impact` is the realized signed price move y (in ticks) attributed to
// the trade. The fields are an already-recorded sample; the fixture only reads
// them.
struct ImpactObservation {
    std::uint64_t seq = 0;
    std::int64_t signed_volume = 0;    // q  (buy +, sell -)
    std::int64_t adv = 1;              // V  (average daily volume, > 0)
    std::int64_t sigma_ticks = 0;      // sigma (volatility proxy, >= 0)
    std::int64_t observed_impact = 0;  // y  (realized signed price move, ticks)
};

// Derived market-impact-law diagnostics plus the order-sensitive replay hash.
// Pure value type: copyable, holds no reference to any external (matching) state.
struct MarketImpactLawDiagnostics {
    std::size_t n = 0;

    // (1) EXACT integer sample aggregates (order-INVARIANT).
    std::int64_t num_buys = 0;        // count q > 0
    std::int64_t num_sells = 0;       // count q < 0
    std::int64_t sum_abs_volume = 0;  // sum |q|

    // (2) Square-root-law least-squares-through-origin fit (order-INVARIANT).
    double sxx = 0.0;             // sum x_i^2
    double sxy = 0.0;             // sum x_i y_i
    bool fit_valid = false;       // true iff sxx > 0
    double sqrt_law_coeff = 0.0;  // Y_hat = sxy / sxx
    double residual_ss = 0.0;     // sum (y_i - Y_hat * x_i)^2
    double total_ss = 0.0;        // sum y_i^2
    double r_squared = 0.0;       // 1 - residual_ss / total_ss (0 if total_ss==0)

    // (3) SEQUENTIAL cumulative-impact path peak (order-SENSITIVE): the largest
    // absolute running sum of realized impacts seen while replaying the sample in
    // order.
    std::int64_t peak_abs_cumulative_impact = 0;

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

// Quantize a real statistic to a fixed-point integer so it can be folded into
// the integer-only hash deterministically (no raw double bytes in the hash).
inline std::int64_t quantize(double v) noexcept {
    const double scaled = v * 1000000.0;  // 1e6 fixed-point
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

// Branch-free integer absolute value over int64 (no overflow on the inputs this
// fixture is exercised with).
inline std::int64_t iabs(std::int64_t v) noexcept { return v < 0 ? -v : v; }

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

// Analytic reference relations for the square-root impact law, used to validate
// the predictor, its concavity, and its monotonicity on closed-form cases.
// Stylized regularities, computed here so the fixture is self-checking and
// self-contained.
namespace analytic {

// Sign of a signed volume: +1 buy, -1 sell, 0 flat.
inline int trade_sign(std::int64_t q) noexcept {
    return (q > 0) ? 1 : (q < 0 ? -1 : 0);
}

// Square-root-law predictor x = sign(q) * sigma * sqrt(|q| / V). The DIRECTIONAL,
// concave, sub-linear shape of aggregate impact. A non-positive V returns 0
// (degenerate, no impact attributable). Deterministic double arithmetic via
// det_sqrt; no <cmath> dependency.
inline double sqrt_impact_predictor(std::int64_t q, std::int64_t sigma,
                                    std::int64_t adv) noexcept {
    if (adv <= 0) return 0.0;
    const double ratio = static_cast<double>(detail::iabs(q)) /
                         static_cast<double>(adv);
    return static_cast<double>(trade_sign(q)) * static_cast<double>(sigma) *
           detail::det_sqrt(ratio);
}

// Full fitted square-root impact for size q under a coefficient Y:
// impact(q) = Y * sign(q) * sigma * sqrt(|q| / V).
inline double sqrt_impact(double coeff, std::int64_t q, std::int64_t sigma,
                          std::int64_t adv) noexcept {
    return coeff * sqrt_impact_predictor(q, sigma, adv);
}

// Per-share (marginal-average) impact magnitude of the square-root law:
// |impact(q)| / |q| = Y * sigma / sqrt(|q| * V), which is STRICTLY DECREASING in
// trade size |q| -- the concavity / diminishing-marginal-impact signature. A
// non-positive size or V returns 0.
inline double per_share_sqrt_impact(double coeff, std::int64_t q,
                                    std::int64_t sigma, std::int64_t adv) noexcept {
    const std::int64_t a = detail::iabs(q);
    if (a <= 0 || adv <= 0) return 0.0;
    const double denom = detail::det_sqrt(static_cast<double>(a) *
                                          static_cast<double>(adv));
    if (denom <= 0.0) return 0.0;
    return coeff * static_cast<double>(sigma) / denom;
}

// Central second difference of the (unsigned, buy-side) square-root impact in
// size: impact(q+h) - 2*impact(q) + impact(q-h). For the concave square-root law
// this is NEGATIVE (impact is concave in size). Uses buy-side (positive q) impact
// magnitudes; requires q > h > 0 and V > 0.
inline double impact_second_difference(double coeff, std::int64_t q,
                                       std::int64_t h, std::int64_t sigma,
                                       std::int64_t adv) noexcept {
    const double up = sqrt_impact(coeff, q + h, sigma, adv);
    const double mid = sqrt_impact(coeff, q, sigma, adv);
    const double dn = sqrt_impact(coeff, q - h, sigma, adv);
    return up - 2.0 * mid + dn;
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered sample and the
// derived diagnostics. Pure, read-only: folds each observation's ORDINAL with its
// bytes so reordering the input changes the hash; folds the integer aggregates,
// the quantized fit statistics, the order-sensitive cumulative-impact peak, and
// the sample count so truncation/extension cannot collide with a prefix.
inline ReplayHash market_impact_law_hash(
    const std::vector<ImpactObservation>& sample,
    const MarketImpactLawDiagnostics& diag) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const ImpactObservation& o : sample) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, o.seq);
        h = detail::fnv1a_i64(h, o.signed_volume);
        h = detail::fnv1a_i64(h, o.adv);
        h = detail::fnv1a_i64(h, o.sigma_ticks);
        h = detail::fnv1a_i64(h, o.observed_impact);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the sample count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(diag.n));
    h = detail::fnv1a_i64(h, diag.num_buys);
    h = detail::fnv1a_i64(h, diag.num_sells);
    h = detail::fnv1a_i64(h, diag.sum_abs_volume);
    h = detail::fnv1a_i64(h, detail::quantize(diag.sxx));
    h = detail::fnv1a_i64(h, detail::quantize(diag.sxy));
    h = detail::fnv1a_byte(h, diag.fit_valid ? 1u : 0u);
    h = detail::fnv1a_i64(h, detail::quantize(diag.sqrt_law_coeff));
    h = detail::fnv1a_i64(h, detail::quantize(diag.residual_ss));
    h = detail::fnv1a_i64(h, detail::quantize(diag.total_ss));
    h = detail::fnv1a_i64(h, detail::quantize(diag.r_squared));
    h = detail::fnv1a_i64(h, diag.peak_abs_cumulative_impact);
    return h;
}

// Compute the offline market-impact-law diagnostics for a replayable ordered
// sample of trades. PURE, READ-ONLY, OFFLINE: the sample is taken by const
// reference and never modified; the result is a plain value. The work is FIXED
// per observation (two linear passes), so the computation is deterministic and
// never iterates a data-dependent number of times. This cannot mutate, and is not
// consumed by, any matching state.
inline MarketImpactLawDiagnostics compute_market_impact_law_diagnostics(
    const std::vector<ImpactObservation>& sample) {
    MarketImpactLawDiagnostics diag;
    diag.n = sample.size();

    // Pass 1: exact integer aggregates + least-squares-through-origin sums +
    // sequential cumulative-impact peak. The aggregates and fit sums are
    // order-invariant; the cumulative peak is order-sensitive.
    std::int64_t running_impact = 0;
    for (const ImpactObservation& o : sample) {
        if (o.signed_volume > 0) {
            ++diag.num_buys;
        } else if (o.signed_volume < 0) {
            ++diag.num_sells;
        }
        diag.sum_abs_volume += detail::iabs(o.signed_volume);

        const double x = analytic::sqrt_impact_predictor(o.signed_volume,
                                                         o.sigma_ticks, o.adv);
        const double y = static_cast<double>(o.observed_impact);
        diag.sxx += x * x;
        diag.sxy += x * y;
        diag.total_ss += y * y;

        running_impact += o.observed_impact;
        const std::int64_t mag = detail::iabs(running_impact);
        if (mag > diag.peak_abs_cumulative_impact) {
            diag.peak_abs_cumulative_impact = mag;
        }
    }

    // Least-squares-through-origin coefficient and goodness of fit.
    if (diag.sxx > 0.0) {
        diag.fit_valid = true;
        diag.sqrt_law_coeff = diag.sxy / diag.sxx;
    }

    // Pass 2: residual sum of squares under the fitted coefficient.
    if (diag.fit_valid) {
        for (const ImpactObservation& o : sample) {
            const double x = analytic::sqrt_impact_predictor(o.signed_volume,
                                                            o.sigma_ticks, o.adv);
            const double y = static_cast<double>(o.observed_impact);
            const double resid = y - diag.sqrt_law_coeff * x;
            diag.residual_ss += resid * resid;
        }
        if (diag.total_ss > 0.0) {
            diag.r_squared = 1.0 - diag.residual_ss / diag.total_ss;
        }
    }

    diag.replay_hash = market_impact_law_hash(sample, diag);
    return diag;
}

// Read-only convenience wrapper. Holds a const view of an externally owned trade
// sample and exposes ONLY the computation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The diagnostics it returns can be read but never
// written back into matching.
class MarketImpactLawReplay {
public:
    explicit MarketImpactLawReplay(
        const std::vector<ImpactObservation>& sample) noexcept
        : sample_(&sample) {}

    MarketImpactLawDiagnostics compute() const {
        return compute_market_impact_law_diagnostics(*sample_);
    }
    ReplayHash hash() const { return compute().replay_hash; }
    std::size_t size() const noexcept { return sample_->size(); }
    bool empty() const noexcept { return sample_->empty(); }

private:
    const std::vector<ImpactObservation>* sample_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline diagnostic types do NOT,
// which is the type-level proof that a fitted impact-law value cannot be wired
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

// Static proof, evaluated at header-include time, that the offline types are pure
// data with no matching-engine coupling:
//   * trivially copyable + standard layout => no engine handle, pointer-back, or
//     vtable that could reach matching state;
//   * no `apply_to_matching` feedback member => the fitted law is one-way OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<ImpactObservation>,
              "ImpactObservation must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<MarketImpactLawDiagnostics>,
              "MarketImpactLawDiagnostics must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<ImpactObservation>,
              "ImpactObservation must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<MarketImpactLawDiagnostics>,
              "MarketImpactLawDiagnostics must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<ImpactObservation>,
              "ImpactObservation must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<MarketImpactLawDiagnostics>,
              "MarketImpactLawDiagnostics must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<MarketImpactLawReplay>,
              "MarketImpactLawReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<MarketImpactLawDiagnostics,
                          decltype(&MarketImpactLawReplay::compute),
                          const MarketImpactLawReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
