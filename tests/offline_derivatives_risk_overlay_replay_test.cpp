#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/derivatives_risk_overlay_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the `needs-code-validation`
// (the channel asked for a worked, replay-neutral example where a
// pricing/vol/Greek input feeds a risk overlay or label without changing a
// replay hash -- supplied here as a portfolio risk-overlay + hedge schedule).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash is deterministic AND order-sensitive: the same ordered
//       legs -> the same hash; a reordering of the SAME multiset of legs -> a
//       different hash (even though the net/gross Greeks and the per-asset skew
//       aggregate are order-invariant);
//   (2) the risk numbers are correct on hand-built cases with KNOWN values:
//       exact integer net/gross portfolio Greeks, the order-sensitive sequential
//       hedge-schedule peak exposure, the multi-asset per-underlying skew
//       aggregate, the EXACT central finite-difference option Greeks on a
//       quadratic value surface, and the Brenner-Subrahmanyam ATM implied-vol
//       asymptotic;
//   (3) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// Depends only on the self-contained header
// `offline/derivatives_risk_overlay_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::AssetSkew;
using offline::DerivativesRiskOverlay;
using offline::DerivativesRiskOverlayReplay;
using offline::OptionLeg;

namespace analytic = offline::analytic;

// Build an ordered leg series from (asset, quantity, delta_milli, gamma_milli,
// skew_milli) tuples; seq = position so the order is explicit.
struct LegSpec {
    std::uint32_t asset;
    std::int64_t quantity;
    std::int64_t delta_milli;
    std::int64_t gamma_milli;
    std::int64_t skew_milli;
};

std::vector<OptionLeg> make_legs(const std::vector<LegSpec>& specs) {
    std::vector<OptionLeg> out;
    out.reserve(specs.size());
    std::uint64_t seq = 0;
    for (const LegSpec& s : specs) {
        out.push_back(OptionLeg{seq, s.asset, s.quantity, s.delta_milli,
                                s.gamma_milli, s.skew_milli});
        ++seq;
    }
    return out;
}

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (3) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<DerivativesRiskOverlay>,
              "DerivativesRiskOverlay must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<OptionLeg>,
              "OptionLeg must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<DerivativesRiskOverlay,
                          decltype(&DerivativesRiskOverlayReplay::compute),
                          const DerivativesRiskOverlayReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline derivatives: EXACT integer net / gross portfolio Greeks",
          "[offline][derivatives][greeks][overlay]") {
    // Two legs on one asset:
    //   +2 contracts, delta 0.500 (500 milli), gamma 0.100 (100 milli)
    //   -3 contracts, delta 0.250 (250 milli), gamma 0.050 (50 milli)
    // net_delta = 2*500 + (-3)*250 = 1000 - 750 = 250
    // net_gamma = 2*100 + (-3)*50  = 200 - 150 = 50
    // gross_delta = |1000| + |-750| = 1750
    const auto legs = make_legs({{0, 2, 500, 100, 0}, {0, -3, 250, 50, 0}});
    const DerivativesRiskOverlay r =
        DerivativesRiskOverlayReplay(legs).compute();

    REQUIRE(r.n == 2);
    REQUIRE(r.net_delta_milli == 250);
    REQUIRE(r.net_gamma_milli == 50);
    REQUIRE(r.gross_delta_milli == 1750);
}

TEST_CASE("offline derivatives: sequential hedge-schedule peak exposure is "
          "order-sensitive (same multiset, same net delta)",
          "[offline][derivatives][hedge-schedule][order]") {
    // Same multiset of leg deltas {+1000, -750, +400} contributed in two orders.
    // Net delta is order-invariant = 1000 - 750 + 400 = 650 in both.
    // Order A [+1000, -750, +400]: running 1000 -> 250 -> 650, peak = 1000.
    // Order B [-750, +1000, +400]: running -750 -> 250 -> 650, peak =  750.
    const auto a = make_legs({{0, 1, 1000, 0, 0}, {0, 1, -750, 0, 0}, {0, 1, 400, 0, 0}});
    const auto b = make_legs({{0, 1, -750, 0, 0}, {0, 1, 1000, 0, 0}, {0, 1, 400, 0, 0}});

    const DerivativesRiskOverlay ra = DerivativesRiskOverlayReplay(a).compute();
    const DerivativesRiskOverlay rb = DerivativesRiskOverlayReplay(b).compute();

    // Order-INVARIANT aggregate.
    REQUIRE(ra.net_delta_milli == 650);
    REQUIRE(rb.net_delta_milli == 650);
    REQUIRE(ra.gross_delta_milli == rb.gross_delta_milli);

    // Order-SENSITIVE hedge-schedule peak.
    REQUIRE(ra.peak_running_abs_delta_milli == 1000);
    REQUIRE(rb.peak_running_abs_delta_milli == 750);

    // Therefore the replay hash differs even though the net Greeks match.
    REQUIRE(ra.replay_hash != rb.replay_hash);
}

TEST_CASE("offline derivatives: multi-asset per-underlying skew aggregate "
          "(around SEPARATE books, order-invariant)",
          "[offline][derivatives][skew][multi-asset]") {
    // Assets {7, 3, 7}: asset 7 nets +2*300 + 4*(-100) = 600 - 400 = 200;
    // asset 3 nets -1*500 = -500. total_abs = |200| + |-500| = 700.
    const auto legs = make_legs(
        {{7, 2, 0, 0, 300}, {3, -1, 0, 0, 500}, {7, 4, 0, 0, -100}});
    const DerivativesRiskOverlay r = DerivativesRiskOverlayReplay(legs).compute();

    REQUIRE(r.num_assets == 2);
    // asset_skews are sorted by asset id => {3, 7}.
    REQUIRE(r.asset_skews.size() == 2);
    REQUIRE(r.asset_skews[0].asset == 3);
    REQUIRE(r.asset_skews[0].net_skew_milli == -500);
    REQUIRE(r.asset_skews[1].asset == 7);
    REQUIRE(r.asset_skews[1].net_skew_milli == 200);
    REQUIRE(r.total_abs_skew_milli == 700);

    // Reordering the legs leaves the per-asset aggregate and total unchanged.
    const auto reordered = make_legs(
        {{7, 4, 0, 0, -100}, {7, 2, 0, 0, 300}, {3, -1, 0, 0, 500}});
    const DerivativesRiskOverlay r2 =
        DerivativesRiskOverlayReplay(reordered).compute();
    REQUIRE(r2.total_abs_skew_milli == 700);
    REQUIRE(r2.asset_skews[0].net_skew_milli == -500);
    REQUIRE(r2.asset_skews[1].net_skew_milli == 200);
}

TEST_CASE("offline derivatives: EXACT central finite-difference Greeks on a "
          "quadratic value surface match the closed form",
          "[offline][derivatives][greeks][finite-difference]") {
    // V(S) = 3*(S-K)^2 + 5*(S-K) + 7, K = 100. Closed forms:
    //   delta(S) = 2*3*(S-K) + 5 = 6*(S-K) + 5
    //   gamma    = 2*3 = 6
    const std::int64_t a = 3, b = 5, c = 7, k = 100;
    for (std::int64_t s : {97, 100, 103, 120}) {
        for (std::int64_t h : {1, 2, 5}) {
            const std::int64_t dnum = analytic::fd_delta_numerator(a, b, c, k, s, h);
            const std::int64_t gnum = analytic::fd_gamma_numerator(a, b, c, k, s, h);
            const std::int64_t delta = analytic::closed_form_delta(a, b, k, s);
            const std::int64_t gamma = analytic::closed_form_gamma(a);

            // Central first difference is EXACT for a quadratic: numerator = 2h*delta.
            REQUIRE(dnum == 2 * h * delta);
            // Central second difference is EXACT: numerator = h^2 * gamma.
            REQUIRE(gnum == h * h * gamma);
        }
    }
    // Spot-check a concrete value: at S=103, delta = 6*3 + 5 = 23, gamma = 6.
    REQUIRE(analytic::closed_form_delta(a, b, k, 103) == 23);
    REQUIRE(analytic::closed_form_gamma(a) == 6);
}

TEST_CASE("offline derivatives: Brenner-Subrahmanyam ATM implied-vol asymptotic",
          "[offline][derivatives][implied-vol]") {
    // sigma ~ sqrt(2*pi/T) * (C/S). With T=1, C=4, S=100:
    //   sqrt(2*pi) ~ 2.5066282746, * (4/100) = 0.1002651...
    const double sigma = analytic::brenner_subrahmanyam_atm_vol(4.0, 100.0, 1.0);
    REQUIRE(sigma == Approx(0.10026513098).epsilon(1e-6));

    // Scaling premium linearly scales sigma (leading-order asymptotic).
    const double sigma2 = analytic::brenner_subrahmanyam_atm_vol(8.0, 100.0, 1.0);
    REQUIRE(sigma2 == Approx(2.0 * sigma).epsilon(1e-9));

    // Degenerate maturity / spot guarded to 0.
    REQUIRE(analytic::brenner_subrahmanyam_atm_vol(4.0, 100.0, 0.0) == Approx(0.0));
    REQUIRE(analytic::brenner_subrahmanyam_atm_vol(4.0, 0.0, 1.0) == Approx(0.0));
}

TEST_CASE("offline derivatives: replay-hash determinism",
          "[offline][derivatives][determinism]") {
    const auto legs = make_legs(
        {{1, 2, 500, 100, 300}, {2, -3, 250, 50, -200}, {1, 5, 120, 30, 40}});

    // Same ordered input -> identical hash, every time and across instances.
    const DerivativesRiskOverlayReplay x(legs);
    const DerivativesRiskOverlayReplay y(legs);
    const std::uint64_t h1 = x.compute().replay_hash;
    const std::uint64_t h2 = x.compute().replay_hash;  // idempotent
    const std::uint64_t h3 = y.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // An extra leg changes the hash, and the count fold prevents a prefix from
    // colliding.
    const auto extended = make_legs(
        {{1, 2, 500, 100, 300}, {2, -3, 250, 50, -200}, {1, 5, 120, 30, 40},
         {3, 1, 10, 1, 1}});
    REQUIRE(DerivativesRiskOverlayReplay(extended).hash() != h1);

    // Changing a single leg field (a delta) changes the hash.
    const auto perturbed = make_legs(
        {{1, 2, 501, 100, 300}, {2, -3, 250, 50, -200}, {1, 5, 120, 30, 40}});
    REQUIRE(DerivativesRiskOverlayReplay(perturbed).hash() != h1);
}

TEST_CASE("offline derivatives: hash order-sensitive while net Greeks and skew "
          "aggregate are order-invariant",
          "[offline][derivatives][determinism][order]") {
    // Same multiset of three legs on one asset in two orders. Net Greeks, gross
    // delta, and per-asset skew are order-invariant; the hedge-schedule peak and
    // the hash are order-sensitive.
    const auto blocks = make_legs(
        {{0, 1, 1000, 10, 5}, {0, 1, 1000, 10, 5}, {0, 1, -1500, 20, 7}});
    const auto zigzag = make_legs(
        {{0, 1, 1000, 10, 5}, {0, 1, -1500, 20, 7}, {0, 1, 1000, 10, 5}});

    const DerivativesRiskOverlay rb = DerivativesRiskOverlayReplay(blocks).compute();
    const DerivativesRiskOverlay rz = DerivativesRiskOverlayReplay(zigzag).compute();

    // Order-INVARIANT.
    REQUIRE(rb.net_delta_milli == rz.net_delta_milli);
    REQUIRE(rb.net_gamma_milli == rz.net_gamma_milli);
    REQUIRE(rb.gross_delta_milli == rz.gross_delta_milli);
    REQUIRE(rb.total_abs_skew_milli == rz.total_abs_skew_milli);

    // Order-SENSITIVE: blocks build to 2000 before the -1500 leg (peak 2000);
    // zigzag goes 1000 -> -500 -> 500 (peak 1000).
    REQUIRE(rb.peak_running_abs_delta_milli == 2000);
    REQUIRE(rz.peak_running_abs_delta_milli == 1000);

    // Therefore the replay hash MUST differ even though the marginal aggregates
    // are identical: reordering the legs is a detectable change.
    REQUIRE(rb.replay_hash != rz.replay_hash);
}

TEST_CASE("offline derivatives: degenerate and empty inputs are well-defined",
          "[offline][derivatives][determinism]") {
    // Single zero-quantity leg: all aggregates zero but a stable nonzero hash.
    const auto zero = make_legs({{4, 0, 999, 999, 999}});
    const DerivativesRiskOverlay rz = DerivativesRiskOverlayReplay(zero).compute();
    REQUIRE(rz.net_delta_milli == 0);
    REQUIRE(rz.gross_delta_milli == 0);
    REQUIRE(rz.peak_running_abs_delta_milli == 0);
    REQUIRE(rz.num_assets == 1);
    REQUIRE(rz.total_abs_skew_milli == 0);
    REQUIRE(rz.replay_hash != 0);

    // Empty -> well-defined zeros, no assets, but a stable deterministic hash.
    const std::vector<OptionLeg> none;
    const DerivativesRiskOverlayReplay empty(none);
    REQUIRE(empty.empty());
    const DerivativesRiskOverlay e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.net_delta_milli == 0);
    REQUIRE(e.num_assets == 0);
    REQUIRE(e.asset_skews.empty());
    REQUIRE(e.replay_hash == empty.hash());  // deterministic
}
