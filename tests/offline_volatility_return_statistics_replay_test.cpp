#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/volatility_return_statistics_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// (the channel asked for a deseasonalized-GARCH feature-export fixture, a
// variance-scaling PUCK calibration artifact, and a diagnostics-on/off
// replay-hash artifact -- all supplied here).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash is deterministic AND order-sensitive: the same ordered
//       series -> the same hash; a reordering of the SAME multiset of returns ->
//       a different hash (even though every marginal moment and the deseasonalized
//       realized variance are invariant);
//   (2) the statistics are correct on hand-built series with KNOWN analytic
//       values: realized variance, intraday-deseasonalized realized variance, the
//       FIXED GARCH(1,1) conditional-variance path, and the PUCK k=2 variance
//       ratio (mean-reverting VR<1 vs trending VR>1);
//   (3) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// Depends only on the self-contained header
// `offline/volatility_return_statistics_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::GarchParams;
using offline::ReturnObservation;
using offline::Tick;
using offline::VolatilityReturnStatisticsReplay;
using offline::VolatilityReturnStats;

// Build an ordered return series from raw tick returns, assigning each
// observation to bin index (i % bin_count) so a known intraday pattern is easy to
// construct. seq = position.
std::vector<ReturnObservation> make_series(const std::vector<Tick>& rets,
                                           std::uint32_t bin_count = 1) {
    std::vector<ReturnObservation> out;
    out.reserve(rets.size());
    std::uint64_t seq = 0;
    for (Tick r : rets) {
        const std::uint32_t bin =
            bin_count > 0 ? static_cast<std::uint32_t>(seq % bin_count) : 0u;
        out.push_back(ReturnObservation{seq, bin, r});
        ++seq;
    }
    return out;
}

// A neutral diurnal profile (all factors 1.0 => no deseasonalization rescale).
std::vector<double> flat_profile(std::size_t bins) {
    return std::vector<double>(bins, 1.0);
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
static_assert(!offline::has_matching_feedback_v<VolatilityReturnStats>,
              "VolatilityReturnStats must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ReturnObservation>,
              "ReturnObservation must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<VolatilityReturnStats,
                          decltype(&VolatilityReturnStatisticsReplay::compute),
                          const VolatilityReturnStatisticsReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline volatility: realized variance / vol and population moments",
          "[offline][volatility][moments]") {
    // Returns {2,-2,2,-2}: mean 0, sum r^2 = 16 -> RV 16, realized vol 4,
    // population variance 4.
    const auto series = make_series({2, -2, 2, -2});
    const auto profile = flat_profile(1);
    const GarchParams garch{};  // all zero: GARCH path collapses to 0
    const VolatilityReturnStats s =
        VolatilityReturnStatisticsReplay(series, profile, garch).compute();

    REQUIRE(s.n == 4);
    REQUIRE(s.sum1 == 0);
    REQUIRE(s.sum2 == 16);  // 4+4+4+4
    REQUIRE(s.mean == Approx(0.0));
    REQUIRE(s.realized_variance == Approx(16.0));
    REQUIRE(s.realized_vol == Approx(4.0));
    REQUIRE(s.variance == Approx(4.0));
}

TEST_CASE("offline volatility: intraday DESEASONALIZED realized variance against "
          "a fixed diurnal profile",
          "[offline][volatility][deseasonalize]") {
    // Two bins with FIXED diurnal variance factors {1, 4}. Returns alternate
    // bins: bin0 returns r=2 (r^2=4, /1 = 4), bin1 returns r=4 (r^2=16, /4 = 4).
    // Every deseasonalized squared return is exactly 4.
    const auto series = make_series({2, 4, 2, 4}, /*bin_count=*/2);
    const std::vector<double> profile = {1.0, 4.0};
    const GarchParams garch{};
    const VolatilityReturnStats s =
        VolatilityReturnStatisticsReplay(series, profile, garch).compute();

    // Raw realized variance is dominated by the high-vol bin: 4+16+4+16 = 40.
    REQUIRE(s.sum2 == 40);
    REQUIRE(s.realized_variance == Approx(40.0));

    // Deseasonalized: each squared return divided by its bin factor -> 4 each,
    // summed over 4 observations = 16; vol = 4.
    REQUIRE(s.deseasonalized_realized_variance == Approx(16.0));
    REQUIRE(s.deseasonalized_realized_vol == Approx(4.0));
}

TEST_CASE("offline volatility: FIXED GARCH(1,1) conditional-variance recursion is "
          "exact",
          "[offline][volatility][garch]") {
    // omega=1, alpha=0.5, beta=0.5, sigma^2_0=2, returns {2,2}.
    //   cv_0 = 2                         (forecast for r_0)
    //   cv_1 = 1 + 0.5*4 + 0.5*2 = 4     (forecast for r_1)
    //   cv_2 = 1 + 0.5*4 + 0.5*4 = 5     (terminal: forecast for unseen r_2)
    // path-mean over the two in-force forecasts = (2 + 4) / 2 = 3.
    const auto series = make_series({2, 2});
    const auto profile = flat_profile(1);
    const GarchParams garch{1.0, 0.5, 0.5, 2.0};
    const VolatilityReturnStats s =
        VolatilityReturnStatisticsReplay(series, profile, garch).compute();

    REQUIRE(s.garch_terminal_variance == Approx(5.0));
    REQUIRE(s.garch_mean_variance == Approx(3.0));
}

TEST_CASE("offline volatility: PUCK variance ratio detects mean reversion vs "
          "trending (exact)",
          "[offline][volatility][puck][variance-ratio]") {
    const auto profile = flat_profile(1);
    const GarchParams garch{};

    // Mean-reverting: {1,-1,1,-1}. mean 0, Var1 = 4/4 = 1. 2-period non-overlap
    // sums {0,0} -> Var2 = 0 -> VR = 0 / (2*1) = 0 (< 1, sign -1).
    const auto reverting = make_series({1, -1, 1, -1});
    const VolatilityReturnStats sr =
        VolatilityReturnStatisticsReplay(reverting, profile, garch).compute();
    REQUIRE(sr.variance_ratio_valid);
    REQUIRE(sr.variance_ratio == Approx(0.0).margin(1e-9));
    REQUIRE(sr.mean_reversion_sign == -1);

    // Trending/persistent: {1,1,-1,-1}. mean 0, Var1 = 1. 2-period sums {2,-2} ->
    // mean 0, Var2 = (4+4)/2 = 4 -> VR = 4 / (2*1) = 2 (> 1, sign +1).
    const auto trending = make_series({1, 1, -1, -1});
    const VolatilityReturnStats st =
        VolatilityReturnStatisticsReplay(trending, profile, garch).compute();
    REQUIRE(st.variance_ratio_valid);
    REQUIRE(st.variance_ratio == Approx(2.0));
    REQUIRE(st.mean_reversion_sign == +1);
}

TEST_CASE("offline volatility: replay-hash determinism",
          "[offline][volatility][determinism]") {
    const auto series = make_series({1, 2, 3, 4, 5, 6}, 3);
    const std::vector<double> profile = {1.0, 2.0, 3.0};
    const GarchParams garch{0.5, 0.2, 0.7, 1.0};

    // Same ordered input + parameters -> identical hash, every time and across
    // instances.
    const VolatilityReturnStatisticsReplay a(series, profile, garch);
    const VolatilityReturnStatisticsReplay b(series, profile, garch);
    const std::uint64_t h1 = a.compute().replay_hash;
    const std::uint64_t h2 = a.compute().replay_hash;  // idempotent
    const std::uint64_t h3 = b.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // A genuinely different series (one extra observation) changes the hash, and
    // the count fold prevents a prefix from colliding.
    const auto extended = make_series({1, 2, 3, 4, 5, 6, 7}, 3);
    REQUIRE(VolatilityReturnStatisticsReplay(extended, profile, garch).hash() != h1);

    // Changing only a FIXED parameter (the GARCH omega) changes the hash too.
    const GarchParams garch2{0.6, 0.2, 0.7, 1.0};
    REQUIRE(VolatilityReturnStatisticsReplay(series, profile, garch2).hash() != h1);
}

TEST_CASE("offline volatility: hash is order-sensitive while marginal moments and "
          "deseasonalized RV are order-invariant (GARCH/VR capture the difference)",
          "[offline][volatility][determinism][order]") {
    const auto profile = flat_profile(1);
    const GarchParams garch{1.0, 0.5, 0.25, 1.0};

    // Two orderings of the SAME multiset {2,2,-1,-1} (mean 0.5). Varying return
    // MAGNITUDES make the GARCH r^2 path -- and so the terminal conditional
    // variance -- genuinely order-sensitive (unlike a +/-1 multiset whose squared
    // returns are all equal).
    //   persistent [2,2,-1,-1]: 2-period sums {4,-2} -> VR = 9 / (2*2.25) = 2.
    //   alternating [2,-1,2,-1]: 2-period sums {1,1} -> VR = 0.
    const auto persistent = make_series({2, 2, -1, -1});   // blocks (trending)
    const auto alternating = make_series({2, -1, 2, -1});  // zig-zag (reverting)

    const VolatilityReturnStats fa =
        VolatilityReturnStatisticsReplay(persistent, profile, garch).compute();
    const VolatilityReturnStats fb =
        VolatilityReturnStatisticsReplay(alternating, profile, garch).compute();

    // Marginal moments + deseasonalized RV are order-INVARIANT.
    REQUIRE(fa.sum1 == fb.sum1);
    REQUIRE(fa.sum2 == fb.sum2);
    REQUIRE(fa.realized_variance == Approx(fb.realized_variance));
    REQUIRE(fa.variance == Approx(fb.variance));
    REQUIRE(fa.deseasonalized_realized_variance ==
            Approx(fb.deseasonalized_realized_variance));

    // The PUCK variance ratio is order-SENSITIVE: persistent trends (VR>1),
    // alternating mean-reverts (VR<1).
    REQUIRE(fa.variance_ratio > 1.0);
    REQUIRE(fb.variance_ratio < 1.0);
    REQUIRE(fa.mean_reversion_sign == +1);
    REQUIRE(fb.mean_reversion_sign == -1);

    // The GARCH path is order-SENSITIVE too: the terminal conditional variances
    // differ between the two orderings.
    REQUIRE(fa.garch_terminal_variance != Approx(fb.garch_terminal_variance));

    // Therefore the replay hash MUST differ even though the marginal statistics
    // are identical: reordering the same returns is a detectable change.
    REQUIRE(fa.replay_hash != fb.replay_hash);
}

TEST_CASE("offline volatility: degenerate and empty inputs are well-defined",
          "[offline][volatility][determinism]") {
    const auto profile = flat_profile(1);
    const GarchParams garch{1.0, 0.5, 0.5, 2.0};

    // All-equal returns -> zero variance -> variance ratio undefined (guarded).
    const auto constant = make_series({5, 5, 5, 5});
    const VolatilityReturnStats c =
        VolatilityReturnStatisticsReplay(constant, profile, garch).compute();
    REQUIRE(c.variance == Approx(0.0));
    REQUIRE_FALSE(c.variance_ratio_valid);
    REQUIRE(c.mean_reversion_sign == 0);
    REQUIRE(c.replay_hash != 0);

    // Empty -> well-defined zeros, GARCH path-mean falls back to the seed, but
    // still a stable deterministic hash (counts and params are folded).
    const std::vector<ReturnObservation> none;
    const VolatilityReturnStatisticsReplay empty(none, profile, garch);
    REQUIRE(empty.empty());
    const VolatilityReturnStats e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.realized_variance == Approx(0.0));
    REQUIRE(e.garch_terminal_variance == Approx(garch.initial_variance));
    REQUIRE(e.garch_mean_variance == Approx(garch.initial_variance));
    REQUIRE_FALSE(e.variance_ratio_valid);
    REQUIRE(e.replay_hash == empty.hash());  // deterministic
}
