#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/market_impact_law_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture opening AND closing the `needs-code-validation`
// (the channel asks for a worked, replay-neutral example where a
// (signed-volume, price-move) sample is reduced to a fitted square-root /
// concave impact-law statistic without changing a replay hash -- supplied here
// as a least-squares square-root-law estimator + concavity/monotonicity
// diagnostics).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash is deterministic AND order-sensitive: the same ordered
//       sample -> the same hash; a reordering of the SAME multiset of trades ->
//       a different hash (even though the buy/sell split, traded volume, and the
//       least-squares fit Y_hat / RSS / R^2 are order-invariant);
//   (2) the diagnostics are correct on hand-built cases with KNOWN values: the
//       square-root-law coefficient is recovered EXACTLY (tight tolerance) on a
//       noise-free synthetic set with |q|/V a perfect square, RSS = 0 and
//       R^2 = 1; a noisy set gives RSS > 0 and R^2 < 1; the per-share impact is
//       strictly decreasing in size and the impact is concave (negative second
//       difference) -- the monotonicity + concavity signature of the law;
//   (3) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// HONEST SCOPE: this validates the diagnostic / ESTIMATION discipline (an impact
// law is a fitted statistic on observed data, evaluated post hoc) -- it is
// implementation evidence only. It does NOT prove the square-root law is the true
// market mechanism; that law is an empirically supported but contested stylized
// regularity, not a theorem. Nothing here feeds back into matching/replay.
//
// Depends only on the self-contained header
// `offline/market_impact_law_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::ImpactObservation;
using offline::MarketImpactLawDiagnostics;
using offline::MarketImpactLawReplay;

namespace analytic = offline::analytic;

// Build an ordered sample from (signed_volume, adv, sigma, observed_impact)
// tuples; seq = position so the order is explicit.
struct ObsSpec {
    std::int64_t signed_volume;
    std::int64_t adv;
    std::int64_t sigma;
    std::int64_t observed_impact;
};

std::vector<ImpactObservation> make_sample(const std::vector<ObsSpec>& specs) {
    std::vector<ImpactObservation> out;
    out.reserve(specs.size());
    std::uint64_t seq = 0;
    for (const ObsSpec& s : specs) {
        out.push_back(ImpactObservation{seq, s.signed_volume, s.adv, s.sigma,
                                        s.observed_impact});
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
static_assert(!offline::has_matching_feedback_v<MarketImpactLawDiagnostics>,
              "MarketImpactLawDiagnostics must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ImpactObservation>,
              "ImpactObservation must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<MarketImpactLawDiagnostics,
                          decltype(&MarketImpactLawReplay::compute),
                          const MarketImpactLawReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline impact-law: EXACT integer sample aggregates "
          "(buy/sell split, traded volume)",
          "[offline][impact-law][aggregates]") {
    // Three trades: +400, -900, +1600 -> 2 buys, 1 sell, sum|q| = 2900.
    const auto sample = make_sample(
        {{400, 100, 2, 12}, {-900, 100, 1, -9}, {1600, 100, 1, 12}});
    const MarketImpactLawDiagnostics d = MarketImpactLawReplay(sample).compute();

    REQUIRE(d.n == 3);
    REQUIRE(d.num_buys == 2);
    REQUIRE(d.num_sells == 1);
    REQUIRE(d.sum_abs_volume == 2900);
}

TEST_CASE("offline impact-law: least-squares square-root fit recovers the KNOWN "
          "coefficient on a noise-free synthetic set (RSS = 0, R^2 = 1)",
          "[offline][impact-law][fit][square-root]") {
    // Construct a noise-free synthetic set generated by impact = Y_true * x with
    // Y_true = 3 and |q|/V a PERFECT SQUARE so the predictor is exact:
    //   q=+400, V=100, sigma=2 -> sqrt(4)=2 -> x = +1*2*2 = +4 -> y = 3*4  =  12
    //   q=-900, V=100, sigma=1 -> sqrt(9)=3 -> x = -1*1*3 = -3 -> y = 3*-3 =  -9
    //   q=+1600,V=100, sigma=1 -> sqrt(16)=4-> x = +1*1*4 = +4 -> y = 3*4  =  12
    // Y_hat = sum(x*y)/sum(x^2) = (48 + 27 + 48)/(16 + 9 + 16) = 123/41 = 3.
    const auto sample = make_sample(
        {{400, 100, 2, 12}, {-900, 100, 1, -9}, {1600, 100, 1, 12}});
    const MarketImpactLawDiagnostics d = MarketImpactLawReplay(sample).compute();

    REQUIRE(d.fit_valid);
    REQUIRE(d.sxx == Approx(41.0).epsilon(1e-9));
    REQUIRE(d.sxy == Approx(123.0).epsilon(1e-9));
    REQUIRE(d.sqrt_law_coeff == Approx(3.0).epsilon(1e-9));
    REQUIRE(d.residual_ss == Approx(0.0).margin(1e-9));
    REQUIRE(d.total_ss == Approx(369.0).epsilon(1e-9));  // 144 + 81 + 144
    REQUIRE(d.r_squared == Approx(1.0).epsilon(1e-9));
}

TEST_CASE("offline impact-law: a noisy sample gives RSS > 0 and R^2 < 1",
          "[offline][impact-law][fit][residual]") {
    // Same predictors as the noise-free set (x = +4, -3, +4) but perturb the last
    // observed impact from 12 to 16 (a +4 shock).
    //   sxx = 41, sxy = 4*12 + (-3)*(-9) + 4*16 = 48 + 27 + 64 = 139
    //   Y_hat = 139/41 ~ 3.39024
    const auto sample = make_sample(
        {{400, 100, 2, 12}, {-900, 100, 1, -9}, {1600, 100, 1, 16}});
    const MarketImpactLawDiagnostics d = MarketImpactLawReplay(sample).compute();

    REQUIRE(d.fit_valid);
    REQUIRE(d.sxx == Approx(41.0).epsilon(1e-9));
    REQUIRE(d.sxy == Approx(139.0).epsilon(1e-9));
    REQUIRE(d.sqrt_law_coeff == Approx(139.0 / 41.0).epsilon(1e-9));
    REQUIRE(d.residual_ss > 0.0);
    REQUIRE(d.r_squared < 1.0);
    REQUIRE(d.r_squared > 0.0);  // the law still explains most of the variance
}

TEST_CASE("offline impact-law: per-share impact is STRICTLY DECREASING in size "
          "(concavity / diminishing-marginal-impact signature)",
          "[offline][impact-law][concavity][monotonicity]") {
    // per_share = Y * sigma / sqrt(|q| * V), strictly decreasing in |q|.
    const double Y = 2.0;
    const std::int64_t sigma = 5, V = 100;
    double prev = analytic::per_share_sqrt_impact(Y, 100, sigma, V);
    for (std::int64_t q : {400, 900, 1600, 2500, 3600}) {
        const double cur = analytic::per_share_sqrt_impact(Y, q, sigma, V);
        REQUIRE(cur < prev);  // strictly decreasing in trade size
        REQUIRE(cur > 0.0);
        prev = cur;
    }
    // Hand-checked value: |q|=400, V=100 -> sqrt(40000)=200 -> 2*5/200 = 0.05.
    REQUIRE(analytic::per_share_sqrt_impact(Y, 400, sigma, V) ==
            Approx(0.05).epsilon(1e-9));
}

TEST_CASE("offline impact-law: total impact is CONCAVE in size "
          "(negative central second difference)",
          "[offline][impact-law][concavity]") {
    // For the concave square-root law the second difference is negative.
    const double Y = 3.0;
    const std::int64_t sigma = 4, V = 1000;
    for (std::int64_t q : {200, 500, 900, 1600}) {
        const double d2 = analytic::impact_second_difference(Y, q, 100, sigma, V);
        REQUIRE(d2 < 0.0);  // impact(q+h) - 2 impact(q) + impact(q-h) < 0
    }
    // A linear law would give exactly zero second difference; the strict
    // negativity here is the concavity that distinguishes square-root impact.
    REQUIRE(analytic::impact_second_difference(Y, 900, 100, sigma, V) <
            -1e-9);
}

TEST_CASE("offline impact-law: fit is order-invariant while the cumulative-impact "
          "path peak and the hash are order-sensitive",
          "[offline][impact-law][order][determinism]") {
    // Same multiset of three trades in two orders. The least-squares fit and the
    // sample aggregates are order-invariant; the sequential cumulative-impact peak
    // and the hash are order-sensitive. The realized impacts {+10, +10, -16} sum
    // to a small final excursion (4) so the worst intermediate peak depends on
    // order.
    const auto a = make_sample(
        {{400, 100, 2, 10}, {1600, 100, 1, 10}, {-900, 100, 1, -16}});
    const auto b = make_sample(
        {{-900, 100, 1, -16}, {400, 100, 2, 10}, {1600, 100, 1, 10}});

    const MarketImpactLawDiagnostics ra = MarketImpactLawReplay(a).compute();
    const MarketImpactLawDiagnostics rb = MarketImpactLawReplay(b).compute();

    // Order-INVARIANT estimation + aggregates.
    REQUIRE(ra.sqrt_law_coeff == Approx(rb.sqrt_law_coeff).epsilon(1e-12));
    REQUIRE(ra.residual_ss == Approx(rb.residual_ss).margin(1e-9));
    REQUIRE(ra.r_squared == Approx(rb.r_squared).epsilon(1e-12));
    REQUIRE(ra.num_buys == rb.num_buys);
    REQUIRE(ra.sum_abs_volume == rb.sum_abs_volume);

    // Order-SENSITIVE cumulative-impact path peak:
    //   order A: running 10 -> 20 -> 4, peak = 20.
    //   order B: running -16 -> -6 -> 4, peak = 16.
    REQUIRE(ra.peak_abs_cumulative_impact == 20);
    REQUIRE(rb.peak_abs_cumulative_impact == 16);

    // Therefore the replay hash MUST differ even though the fit matches.
    REQUIRE(ra.replay_hash != rb.replay_hash);
}

TEST_CASE("offline impact-law: replay-hash determinism",
          "[offline][impact-law][determinism]") {
    const auto sample = make_sample(
        {{400, 100, 2, 12}, {-900, 100, 1, -9}, {1600, 100, 1, 12}});

    // Same ordered input -> identical hash, every time and across instances.
    const MarketImpactLawReplay x(sample);
    const MarketImpactLawReplay y(sample);
    const std::uint64_t h1 = x.compute().replay_hash;
    const std::uint64_t h2 = x.compute().replay_hash;  // idempotent
    const std::uint64_t h3 = y.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // An extra trade changes the hash, and the count fold prevents a prefix from
    // colliding.
    const auto extended = make_sample(
        {{400, 100, 2, 12}, {-900, 100, 1, -9}, {1600, 100, 1, 12},
         {100, 100, 1, 1}});
    REQUIRE(MarketImpactLawReplay(extended).hash() != h1);

    // Changing a single observed impact changes the hash.
    const auto perturbed = make_sample(
        {{400, 100, 2, 13}, {-900, 100, 1, -9}, {1600, 100, 1, 12}});
    REQUIRE(MarketImpactLawReplay(perturbed).hash() != h1);
}

TEST_CASE("offline impact-law: degenerate and empty inputs are well-defined",
          "[offline][impact-law][determinism]") {
    // A single zero-volume trade: predictor is 0 (no impact attributable), so the
    // fit is invalid but the diagnostic is still well-defined with a stable hash.
    const auto flat = make_sample({{0, 100, 5, 7}});
    const MarketImpactLawDiagnostics rf = MarketImpactLawReplay(flat).compute();
    REQUIRE(rf.num_buys == 0);
    REQUIRE(rf.num_sells == 0);
    REQUIRE(rf.sum_abs_volume == 0);
    REQUIRE_FALSE(rf.fit_valid);  // sxx == 0
    REQUIRE(rf.sqrt_law_coeff == Approx(0.0));
    REQUIRE(rf.peak_abs_cumulative_impact == 7);
    REQUIRE(rf.replay_hash != 0);

    // A non-positive ADV is guarded: predictor is 0, fit invalid, no crash.
    const auto bad_adv = make_sample({{400, 0, 2, 5}, {-100, -3, 1, -2}});
    const MarketImpactLawDiagnostics rb = MarketImpactLawReplay(bad_adv).compute();
    REQUIRE_FALSE(rb.fit_valid);
    REQUIRE(rb.replay_hash != 0);

    // Empty -> well-defined zeros, invalid fit, but a stable deterministic hash.
    const std::vector<ImpactObservation> none;
    const MarketImpactLawReplay empty(none);
    REQUIRE(empty.empty());
    const MarketImpactLawDiagnostics e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.num_buys == 0);
    REQUIRE_FALSE(e.fit_valid);
    REQUIRE(e.peak_abs_cumulative_impact == 0);
    REQUIRE(e.replay_hash == empty.hash());  // deterministic
}
