#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/queueing_calibration_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the AMBER residual gate "Implemented
// (Wave-I canonical-readiness gate list, 2026-06-19).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the replay hash is deterministic AND order-sensitive: the same ordered
//       samples -> a different hash (even though every PH moment is invariant);
//   (2) the PH moment / family fit is correct on known analytic cases
//       (Erlang-k SCV = 1/k, Exponential SCV = 1, Hyperexponential SCV > 1) and
//       the MAP lag-1 signature detects arrival correlation the marginal PH
//       moments cannot see;
//   (3) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// Depends only on the self-contained header
// `offline/queueing_calibration_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::analytic::erlang_scv;
using offline::analytic::hyperexp_scv;
using offline::CalibrationSample;
using offline::PhFamily;
using offline::PhMapFit;
using offline::QueueingCalibrationReplay;
using offline::Tick;

// Build a sample sequence from raw interval ticks (seq = position) so a known
// multiset has known exact moments.
std::vector<CalibrationSample> make_samples(const std::vector<Tick>& intervals) {
    std::vector<CalibrationSample> out;
    out.reserve(intervals.size());
    std::uint64_t seq = 0;
    for (Tick v : intervals) {
        out.push_back(CalibrationSample{seq, v});
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

// (3) Compile-time read-only / non-feedback guarantees mirrored in the test TU
// so a regression that makes the diagnostic mutating or feedback-coupled fails
// THIS translation unit's compile. The positive control proves the trait
// genuinely discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<PhMapFit>,
              "PhMapFit must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<CalibrationSample>,
              "CalibrationSample must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<PhMapFit, decltype(&QueueingCalibrationReplay::calibrate),
                          const QueueingCalibrationReplay&>,
    "calibrate() must be callable on a const engine (read-only)");

TEST_CASE("offline queueing calibration: Erlang-2 analytic moments and fit",
          "[offline][queueing][ph][moments]") {
    // Multiset {0,4,4,8}: mean 4, variance 8, SCV 0.5 == 1/2 -> Erlang-2.
    const auto samples = make_samples({0, 4, 4, 8});
    const QueueingCalibrationReplay engine(samples);
    const PhMapFit fit = engine.calibrate();

    REQUIRE(fit.n == 4);
    REQUIRE(fit.sum1 == 16);
    REQUIRE(fit.sum2 == 96);   // 0 + 16 + 16 + 64
    REQUIRE(fit.sum3 == 640);  // 0 + 64 + 64 + 512
    REQUIRE(fit.mean == Approx(4.0));
    REQUIRE(fit.variance == Approx(8.0));
    REQUIRE(fit.scv == Approx(0.5));
    REQUIRE(fit.scv == Approx(erlang_scv(2)));  // analytic SCV of Erlang-2

    REQUIRE(fit.family == PhFamily::Erlang);
    REQUIRE(fit.erlang_k == 2);
    // Erlang-k per-phase rate lambda = k/mean.
    REQUIRE(fit.rate1 == Approx(2.0 / 4.0));
    REQUIRE_FALSE(fit.map_correlated);  // i.i.d.-shaped, but order-dependent below
}

TEST_CASE("offline queueing calibration: Exponential analytic case (SCV == 1)",
          "[offline][queueing][ph][moments]") {
    // Multiset {0,8}: mean 4, variance 16, SCV == 1 -> Exponential.
    const auto samples = make_samples({0, 8});
    const PhMapFit fit = QueueingCalibrationReplay(samples).calibrate();

    REQUIRE(fit.mean == Approx(4.0));
    REQUIRE(fit.variance == Approx(16.0));
    REQUIRE(fit.scv == Approx(1.0));
    REQUIRE(fit.family == PhFamily::Exponential);
    REQUIRE(fit.erlang_k == 1);
    REQUIRE(fit.rate1 == Approx(1.0 / 4.0));  // memoryless rate 1/mean
}

TEST_CASE("offline queueing calibration: Hyperexponential analytic case (SCV > 1)",
          "[offline][queueing][ph][moments]") {
    // Multiset {0,0,12}: mean 4, variance 32, SCV == 2 -> Hyperexponential.
    const auto samples = make_samples({0, 0, 12});
    const PhMapFit fit = QueueingCalibrationReplay(samples).calibrate();

    REQUIRE(fit.mean == Approx(4.0));
    REQUIRE(fit.variance == Approx(32.0));
    REQUIRE(fit.scv == Approx(2.0));
    REQUIRE(fit.family == PhFamily::Hyperexponential);

    // Balanced-means H2 weights/rates must reproduce the input mean and SCV when
    // fed back through the analytic moment relations (round-trip validation).
    REQUIRE(fit.weight1 + fit.weight2 == Approx(1.0));
    REQUIRE(fit.weight1 / fit.rate1 == Approx(fit.weight2 / fit.rate2));  // balanced
    const double round_trip_mean =
        fit.weight1 / fit.rate1 + fit.weight2 / fit.rate2;
    REQUIRE(round_trip_mean == Approx(4.0));
    const double round_trip_scv =
        hyperexp_scv(fit.weight1, fit.rate1, fit.weight2, fit.rate2);
    REQUIRE(round_trip_scv == Approx(2.0));
}

TEST_CASE("offline queueing calibration: replay-hash determinism",
          "[offline][queueing][determinism]") {
    const auto samples = make_samples({1, 2, 3, 4, 5, 6});

    // Same ordered input -> identical hash, every time and across instances.
    const QueueingCalibrationReplay a(samples);
    const QueueingCalibrationReplay b(samples);
    const std::uint64_t h1 = a.calibrate().replay_hash;
    const std::uint64_t h2 = a.calibrate().replay_hash;  // idempotent
    const std::uint64_t h3 = b.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // A genuinely different sequence (one extra sample) changes the hash, and
    // the count fold prevents a prefix from colliding.
    const auto extended = make_samples({1, 2, 3, 4, 5, 6, 7});
    REQUIRE(QueueingCalibrationReplay(extended).hash() != h1);
}

TEST_CASE("offline queueing calibration: hash is order-sensitive but PH moments "
          "are order-invariant (MAP captures the difference)",
          "[offline][queueing][determinism][map]") {
    // Two orderings of the SAME multiset {1,2,3,4,5,6}.
    const auto ascending = make_samples({1, 2, 3, 4, 5, 6});       // monotone
    const auto alternating = make_samples({1, 6, 2, 5, 3, 4});     // zig-zag

    const PhMapFit fa = QueueingCalibrationReplay(ascending).calibrate();
    const PhMapFit fb = QueueingCalibrationReplay(alternating).calibrate();

    // PH moments are order-INVARIANT: identical sums, mean, variance, SCV, fit.
    REQUIRE(fa.sum1 == fb.sum1);
    REQUIRE(fa.sum2 == fb.sum2);
    REQUIRE(fa.sum3 == fb.sum3);
    REQUIRE(fa.mean == Approx(fb.mean));
    REQUIRE(fa.variance == Approx(fb.variance));
    REQUIRE(fa.scv == Approx(fb.scv));
    REQUIRE(fa.family == fb.family);

    // The MAP lag-1 signature is order-SENSITIVE: a monotone sequence is
    // positively autocorrelated, the zig-zag is (strongly) negatively so.
    REQUIRE(fa.map_lag1_autocorr > 0.0);
    REQUIRE(fb.map_lag1_autocorr < 0.0);
    REQUIRE(fa.map_lag1_autocorr != Approx(fb.map_lag1_autocorr));

    // Therefore the replay hash MUST differ even though the marginal PH fit is
    // identical: reordering the same samples is a detectable change.
    REQUIRE(fa.replay_hash != fb.replay_hash);
}

TEST_CASE("offline queueing calibration: MAP signature is an exact, bounded "
          "function of arrival order",
          "[offline][queueing][map]") {
    // A block sequence {2,6,6,2,2,6,6,2} (mean 4, variance 4) with deviations
    // {-2,2,2,-2,-2,2,2,-2}. Lag-1 cross-product sum = -4 over total SS = 32, so
    // the MAP signature is exactly -1/8: a bounded value in [-1, 1] determined
    // entirely by arrival order.
    const auto samples = make_samples({2, 6, 6, 2, 2, 6, 6, 2});
    const PhMapFit fit = QueueingCalibrationReplay(samples).calibrate();
    REQUIRE(fit.mean == Approx(4.0));
    REQUIRE(fit.variance == Approx(4.0));
    REQUIRE(fit.map_lag1_autocorr == Approx(-1.0 / 8.0).margin(1e-9));
    REQUIRE(fit.map_lag1_autocorr >= -1.0);
    REQUIRE(fit.map_lag1_autocorr <= 1.0);
}

TEST_CASE("offline queueing calibration: degenerate and empty inputs are "
          "well-defined",
          "[offline][queueing][determinism]") {
    // All-equal -> zero variance -> Degenerate (point mass), no MAP correlation.
    const auto constant = make_samples({5, 5, 5, 5});
    const PhMapFit cfit = QueueingCalibrationReplay(constant).calibrate();
    REQUIRE(cfit.variance == Approx(0.0));
    REQUIRE(cfit.scv == Approx(0.0));
    REQUIRE(cfit.family == PhFamily::Degenerate);
    REQUIRE_FALSE(cfit.map_correlated);
    REQUIRE(cfit.replay_hash != 0);

    // Empty -> well-defined zeros, but still a stable non-trivial hash (the
    // count fold and fit fields are folded even with no samples).
    const std::vector<CalibrationSample> none;
    const QueueingCalibrationReplay empty(none);
    REQUIRE(empty.empty());
    const PhMapFit efit = empty.calibrate();
    REQUIRE(efit.n == 0);
    REQUIRE(efit.family == PhFamily::Degenerate);
    REQUIRE(efit.replay_hash == empty.hash());  // deterministic
}
