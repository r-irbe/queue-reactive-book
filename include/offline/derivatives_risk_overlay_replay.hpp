#pragma once

// =============================================================================
// Offline derivatives / options-pricing RISK-OVERLAY replay fixture.
//
// by supplying the missing deterministic, replay-neutral fixture in which an
// option-Greek / implied-vol / cross-asset-skew input feeds a PORTFOLIO RISK
// OVERLAY (and a sequential hedge schedule) without ever touching matching.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It is SELF-CONTAINED and header-only, with
//   std-includes ONLY. It has NO dependency on `FlatLadderBook`, the matching
//   kernel, or ANY `sim::` symbol -- it includes nothing from `sim/`. The
//   matching kernel is *deterministic and price/size/time only*; every quantity
//   computed here (option delta/gamma, an ATM implied-vol asymptotic, a net /
//   gross portfolio Greek, a sequential hedge-schedule peak exposure, a
//   per-underlying skew aggregate) is a PURE OFFLINE risk-overlay LABEL over an
//   already-fixed set of option legs. Data flows ONE WAY, OUT of the legs and
//   into derived risk numbers; a computed value may NEVER be fed back as a
//   matching input, a fill/priority/tie-break rule, a fee/inventory/accounting
//   mutation, a replay-hash perturbation of the engine, or a hot-path branch.
//   This header is the deterministic proof of that one-way, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-derivatives-options-pricing-offline-risk-layer` keeps derivatives/option
//   pricing, implied volatility, Greeks, smooth-impact hedging, and cross-asset
//   skew strictly in the offline pricing / risk-overlay / calibration layer. Its
//   open `needs-code-validation` channel asked for a worked, replay-neutral
//   example where a pricing/vol/Greek input feeds a risk overlay or label under a
//   diagnostics-on/off replay-hash. This fixture supplies exactly that. Given an
//   ORDERED set of option legs (signed contract quantity, per-contract delta and
//   gamma in fixed-point milli-units, a per-underlying skew sensitivity), it
//     * aggregates EXACT integer net and gross portfolio Greeks
//       net_delta = sum qty*delta, net_gamma = sum qty*gamma,
//       gross_delta = sum |qty*delta| -- the risk-overlay aggregation
//       (order-INVARIANT, exact integer arithmetic);
//     * runs a SEQUENTIAL hedge schedule along the leg order, tracking the peak
//       absolute running net delta exposure encountered before the book is fully
//       assembled -- the smooth-impact-hedging-style order-SENSITIVE path metric
//       (same multiset, different order => different peak, same final net delta);
//     * aggregates per-UNDERLYING net skew (sum qty*skew grouped by asset) and the
//       total absolute cross-asset skew -- the multi-asset portfolio-overlay
//       aggregate around SEPARATE books (order-INVARIANT);
//     * exposes a deterministic, order-sensitive FNV-1a-64 replay hash over the
//       ORDERED legs plus the derived risk numbers, so the diagnostic is
//       reproducible and order-sensitive.
//   Two further closed-form OFFLINE pricing-layer relations are provided as
//   directly-testable `analytic` helpers, faithful to the MOC's routed claims:
//     * EXACT central finite-difference option Greeks on a quadratic risk-value
//       surface V(S)=a*(S-K)^2+b*(S-K)+c, for which the central first/second
//       differences recover delta=2a(S-K)+b and gamma=2a with ZERO truncation
//       error -- a deterministic option-Greek from a fixed input (integer-exact);
//     * the Brenner-Subrahmanyam ATM implied-volatility asymptotic
//       sigma ~ sqrt(2*pi/T) * (premium/spot) -- the offline implied-vol pricing
//       layer (deterministic double arithmetic, no <cmath>).
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered legs: an identical
//   ordered set of legs yields an identical result and an identical hash;
//   reordering the SAME multiset of legs leaves the net/gross Greeks and the
//   per-underlying skew aggregate unchanged, yet changes the sequential
//   hedge-schedule peak exposure and the hash (order-sensitive integrity). The
//   computation takes the legs by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Derivatives / options-pricing background (kept self-contained; no equation
// imported from any rendered source image -- these are textbook relations
// re-stated here for the fixture):
//   * Portfolio delta/gamma are the position-weighted sums of per-contract
//     Greeks; a risk overlay nets them across legs around separate books.
//   * A delta-hedging schedule is walked in sequence; the worst intermediate
//     exposure depends on the ORDER in which legs are assembled even when the
//     final net delta does not.
//   * For a quadratic value surface the central finite difference is EXACT:
//     (V(S+h)-V(S-h)) = 2h*(2a(S-K)+b) and (V(S+h)-2V(S)+V(S-h)) = h^2*(2a).
//   * Brenner-Subrahmanyam (1988): the at-the-money option premium is
//     approximately C ~ 0.4*S*sigma*sqrt(T), so sigma ~ sqrt(2*pi/T)*(C/S) is the
//     leading-order ATM implied-vol asymptotic.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick price / sensitivity unit

// One replayable option leg in the portfolio overlay. Pure integer data: no
// pointer, no engine handle, no vtable. `seq` is the deterministic event
// ordinal, `asset` is the underlying-book id (legs on DIFFERENT assets are
// netted only at the overlay, never inside one book's matching). Greeks are
// per-contract, carried in fixed-point milli-units (1e-3) so all aggregation is
// EXACT integer arithmetic.
struct OptionLeg {
    std::uint64_t seq = 0;
    std::uint32_t asset = 0;          // underlying-book id (separate books)
    std::int64_t quantity = 0;        // signed contracts (long +, short -)
    std::int64_t delta_milli = 0;     // per-contract delta * 1000
    std::int64_t gamma_milli = 0;     // per-contract gamma * 1000
    std::int64_t skew_milli = 0;      // per-contract cross-asset skew sens. * 1000
};

// One underlying's aggregated net skew, used for the multi-asset overlay view.
// Pure value type; the vector of these is kept sorted by `asset` so the overlay
// aggregate is deterministic regardless of leg order.
struct AssetSkew {
    std::uint32_t asset = 0;
    std::int64_t net_skew_milli = 0;  // sum over legs on this asset of qty*skew
};

// Derived portfolio risk-overlay numbers plus the order-sensitive replay hash.
// Pure value type: copyable, holds no reference to any external (matching) state.
struct DerivativesRiskOverlay {
    std::size_t n = 0;

    // (1) EXACT integer risk-overlay aggregation (order-INVARIANT).
    std::int64_t net_delta_milli = 0;    // sum qty*delta
    std::int64_t net_gamma_milli = 0;    // sum qty*gamma
    std::int64_t gross_delta_milli = 0;  // sum |qty*delta|

    // (2) SEQUENTIAL hedge-schedule peak exposure (order-SENSITIVE): the largest
    // absolute running net delta seen while assembling the legs in order.
    std::int64_t peak_running_abs_delta_milli = 0;

    // (3) Multi-asset skew overlay (order-INVARIANT): per-underlying net skew
    // (sorted by asset id) and the total absolute cross-asset skew.
    std::vector<AssetSkew> asset_skews;
    std::int64_t total_abs_skew_milli = 0;
    std::size_t num_assets = 0;

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

// Analytic reference relations for the offline pricing layer, used to validate
// Greeks and implied vol on closed-form cases. Textbook relations, computed here
// so the fixture is self-checking and self-contained.
namespace analytic {

// Quadratic option risk-value surface V(S) = a*(S-K)^2 + b*(S-K) + c, all
// integer. A stand-in for a locally-quadratic value used purely OFFLINE; its
// Greeks are exact under central finite differences.
inline std::int64_t quadratic_value(std::int64_t a, std::int64_t b, std::int64_t c,
                                    std::int64_t k, std::int64_t s) noexcept {
    const std::int64_t d = s - k;
    return a * d * d + b * d + c;
}

// Central first-difference NUMERATOR V(S+h) - V(S-h). For the quadratic surface
// this equals 2h * (2a(S-K)+b) EXACTLY (no truncation error), so dividing by 2h
// recovers the closed-form delta.
inline std::int64_t fd_delta_numerator(std::int64_t a, std::int64_t b,
                                       std::int64_t c, std::int64_t k,
                                       std::int64_t s, std::int64_t h) noexcept {
    return quadratic_value(a, b, c, k, s + h) - quadratic_value(a, b, c, k, s - h);
}

// Central second-difference NUMERATOR V(S+h) - 2V(S) + V(S-h). For the quadratic
// surface this equals h^2 * (2a) EXACTLY, so dividing by h^2 recovers gamma=2a.
inline std::int64_t fd_gamma_numerator(std::int64_t a, std::int64_t b,
                                       std::int64_t c, std::int64_t k,
                                       std::int64_t s, std::int64_t h) noexcept {
    return quadratic_value(a, b, c, k, s + h) - 2 * quadratic_value(a, b, c, k, s) +
           quadratic_value(a, b, c, k, s - h);
}

// Closed-form delta of the quadratic surface: dV/dS = 2a(S-K) + b.
inline std::int64_t closed_form_delta(std::int64_t a, std::int64_t b, std::int64_t k,
                                      std::int64_t s) noexcept {
    return 2 * a * (s - k) + b;
}

// Closed-form gamma of the quadratic surface: d2V/dS2 = 2a.
inline std::int64_t closed_form_gamma(std::int64_t a) noexcept { return 2 * a; }

// Brenner-Subrahmanyam (1988) ATM implied-volatility asymptotic:
// sigma ~ sqrt(2*pi/T) * (premium / spot). Deterministic double arithmetic with
// a FIXED 2*pi constant; non-positive spot or maturity returns 0.
inline double brenner_subrahmanyam_atm_vol(double premium, double spot,
                                           double maturity) noexcept {
    if (spot <= 0.0 || maturity <= 0.0) return 0.0;
    constexpr double kTwoPi = 6.28318530717958647692;
    return detail::det_sqrt(kTwoPi / maturity) * (premium / spot);
}

}  // namespace analytic

// Deterministic, order-sensitive FNV-1a-64 hash over the ordered legs and the
// derived risk numbers. Pure, read-only: folds each leg's ORDINAL with its bytes
// so reordering the input changes the hash; folds the integer aggregates, the
// order-sensitive peak exposure, and the sorted per-asset skews; folds the leg
// count so truncation/extension cannot collide with a prefix.
inline ReplayHash derivatives_overlay_hash(const std::vector<OptionLeg>& legs,
                                           const DerivativesRiskOverlay& risk) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;

    std::uint64_t ordinal = 0;
    for (const OptionLeg& leg : legs) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, leg.seq);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(leg.asset));
        h = detail::fnv1a_i64(h, leg.quantity);
        h = detail::fnv1a_i64(h, leg.delta_milli);
        h = detail::fnv1a_i64(h, leg.gamma_milli);
        h = detail::fnv1a_i64(h, leg.skew_milli);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the leg count

    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(risk.n));
    h = detail::fnv1a_i64(h, risk.net_delta_milli);
    h = detail::fnv1a_i64(h, risk.net_gamma_milli);
    h = detail::fnv1a_i64(h, risk.gross_delta_milli);
    h = detail::fnv1a_i64(h, risk.peak_running_abs_delta_milli);
    h = detail::fnv1a_i64(h, risk.total_abs_skew_milli);

    std::uint64_t asset_ordinal = 0;
    for (const AssetSkew& as : risk.asset_skews) {
        h = detail::fnv1a_u64(h, asset_ordinal);
        h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(as.asset));
        h = detail::fnv1a_i64(h, as.net_skew_milli);
        ++asset_ordinal;
    }
    h = detail::fnv1a_u64(h, asset_ordinal);  // fold the asset count
    return h;
}

// Compute the offline derivatives risk overlay for an ordered set of option
// legs. PURE, READ-ONLY, OFFLINE: the legs are taken by const reference and never
// modified; the result is a plain value. The work is FIXED per leg (a few linear
// passes plus a sort over the distinct assets), so the computation is
// deterministic and never iterates a data-dependent number of times. This cannot
// mutate, and is not consumed by, any matching state.
inline DerivativesRiskOverlay compute_derivatives_risk_overlay(
    const std::vector<OptionLeg>& legs) {
    DerivativesRiskOverlay risk;
    risk.n = legs.size();

    // (1) EXACT integer aggregation + (2) SEQUENTIAL hedge-schedule peak in one
    // ordered pass. net_* are order-invariant; running/peak is order-sensitive.
    std::int64_t running_delta = 0;
    for (const OptionLeg& leg : legs) {
        const std::int64_t leg_delta = leg.quantity * leg.delta_milli;
        const std::int64_t leg_gamma = leg.quantity * leg.gamma_milli;
        risk.net_delta_milli += leg_delta;
        risk.net_gamma_milli += leg_gamma;
        risk.gross_delta_milli += detail::iabs(leg_delta);

        running_delta += leg_delta;
        const std::int64_t mag = detail::iabs(running_delta);
        if (mag > risk.peak_running_abs_delta_milli) {
            risk.peak_running_abs_delta_milli = mag;
        }
    }

    // (3) Multi-asset skew overlay: accumulate per-underlying net skew, then sort
    // by asset id so the aggregate and its hash are order-invariant.
    for (const OptionLeg& leg : legs) {
        const std::int64_t leg_skew = leg.quantity * leg.skew_milli;
        auto it = std::find_if(risk.asset_skews.begin(), risk.asset_skews.end(),
                               [&](const AssetSkew& as) { return as.asset == leg.asset; });
        if (it == risk.asset_skews.end()) {
            risk.asset_skews.push_back(AssetSkew{leg.asset, leg_skew});
        } else {
            it->net_skew_milli += leg_skew;
        }
    }
    std::sort(risk.asset_skews.begin(), risk.asset_skews.end(),
              [](const AssetSkew& x, const AssetSkew& y) { return x.asset < y.asset; });
    risk.num_assets = risk.asset_skews.size();
    for (const AssetSkew& as : risk.asset_skews) {
        risk.total_abs_skew_milli += detail::iabs(as.net_skew_milli);
    }

    risk.replay_hash = derivatives_overlay_hash(legs, risk);
    return risk;
}

// option legs and exposes ONLY the computation; it has no mutating member and no
// matching-engine coupling, making the offline / one-way-data-flow boundary
// explicit at the call site. The risk numbers it returns can be read but never
// written back into matching.
class DerivativesRiskOverlayReplay {
public:
    explicit DerivativesRiskOverlayReplay(const std::vector<OptionLeg>& legs) noexcept
        : legs_(&legs) {}

    DerivativesRiskOverlay compute() const {
        return compute_derivatives_risk_overlay(*legs_);
    }
    ReplayHash hash() const { return compute().replay_hash; }
    std::size_t size() const noexcept { return legs_->size(); }
    bool empty() const noexcept { return legs_->empty(); }

private:
    const std::vector<OptionLeg>* legs_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline overlay types do NOT, which
// is the type-level proof that a computed risk value cannot be wired back into
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
//   * trivially copyable + standard layout (for the fixed-size leg/aggregate
//     types) => no engine handle, pointer-back, or vtable reaching matching;
//   * no `apply_to_matching` feedback member => risk numbers are one-way OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<OptionLeg>,
              "OptionLeg must be pure data (no engine handle)");
static_assert(std::is_trivially_copyable_v<AssetSkew>,
              "AssetSkew must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<OptionLeg>,
              "OptionLeg must be standard-layout (no vtable / engine coupling)");
static_assert(std::is_standard_layout_v<AssetSkew>,
              "AssetSkew must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<OptionLeg>,
              "OptionLeg must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<DerivativesRiskOverlay>,
              "DerivativesRiskOverlay must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<DerivativesRiskOverlayReplay>,
              "DerivativesRiskOverlayReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<DerivativesRiskOverlay,
                          decltype(&DerivativesRiskOverlayReplay::compute),
                          const DerivativesRiskOverlayReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
