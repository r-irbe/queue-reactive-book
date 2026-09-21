#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/intraday_seasonality_deseasonalization_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the open code channel
// (the MOC records that "a MOC-level OFFLINE time-base fixture (scenario hash
// changes when the declared time scale / session policy changes ...) is NOT yet
// implemented"). The concrete, distinct worked vehicle is intraday-seasonality
// (time-of-day) deseasonalization under a DECLARED diurnal time-base.
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (a) the scenario hash is deterministic AND order-sensitive: the same ordered
//       series under the same declared policy -> the same hash; reordering the
//       SAME observations leaves the set-statistic seasonal PROFILE invariant but
//       changes the ordered deseasonalized output and therefore the hash;
//   (b) the scenario hash CHANGES when the DECLARED TIME SCALE (bins-per-session
//       / session length) or the DECLARED SESSION POLICY (session-open phase /
//       clock class) changes, even on byte-identical raw samples -- the MOC's
//       `scenario_hash_changes_when_time_scale_changes` falsifier, i.e. a
//       normalized timestamp is NOT self-sufficient without its declared
//       time-base;
//   (c) the DESEASONALIZATION is domain-correct on a hand-built series with known
//       per-bin means (a bin-constant series deseasonalizes exactly to the grand
//       mean; the multiplicative factors are bin_mean / grand_mean);
//   (d) the diagnostic is strictly read-only / non-feedback: deseasonalizing does
//       not mutate the input series, the offline types carry no matching-engine
//       coupling (static_assert discriminator + a POSITIVE control type), and the
//       output is a one-way OUT diagnostic, never a matching input or a live
//       trading signal.
//
// Depends only on the self-contained header
// `offline/intraday_seasonality_deseasonalization_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::DeseasonalizationDiagnostics;
using offline::DeseasonalizationReplayResult;
using offline::DeseasonalizedPoint;
using offline::deseasonalize_intraday;
using offline::IntradaySeasonalityDeseasonalizationReplay;
using offline::ObsValue;
using offline::SeasonalBin;
using offline::TimeBasePolicy;
using offline::TimeScaleClass;
using offline::TimeTick;
using offline::TodObservation;

struct RawObs {
    TimeTick tod_tick;
    ObsValue value;
};

// Build observations with seq = position so a hand-constructed series has known,
// exactly computable binning and deseasonalization behaviour.
std::vector<TodObservation> make_obs(const std::vector<RawObs>& raw) {
    std::vector<TodObservation> out;
    out.reserve(raw.size());
    std::uint64_t seq = 0;
    for (const RawObs& r : raw) {
        out.push_back(TodObservation{seq, r.tod_tick, r.value});
        ++seq;
    }
    return out;
}

// Base declared time-base: two diurnal bins over a session of span 2 ticks,
// opening at tick 0, on the session clock. tod 0 -> bin 0, tod 1 -> bin 1.
constexpr TimeBasePolicy kBase{/*clock=*/TimeScaleClass::SessionClock,
                               /*session_length_ticks=*/2,
                               /*session_open_ticks=*/0,
                               /*bins_per_session=*/2};

// A bin-constant series: bin 0 always 10, bin 1 always 20. Grand mean 15.
const std::vector<RawObs> kBinConstant{{0, 10}, {1, 20}, {0, 10}, {1, 20}};

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (d) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<DeseasonalizationDiagnostics>,
              "DeseasonalizationDiagnostics must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<TodObservation>,
              "TodObservation must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<DeseasonalizationReplayResult,
                          decltype(&IntradaySeasonalityDeseasonalizationReplay::run),
                          const IntradaySeasonalityDeseasonalizationReplay&>,
    "run() must be callable on a const engine (read-only)");

TEST_CASE("offline deseasonalization: bin-constant series flattens to grand mean",
          "[offline][timebase][deseasonalization]") {
    // bin 0 == 10, bin 1 == 20, grand mean 15. The multiplicative factors are
    // 10/15 = 2/3 and 20/15 = 4/3; dividing each value by its factor recovers the
    // grand mean exactly.
    const auto obs = make_obs(kBinConstant);
    const DeseasonalizationReplayResult r = deseasonalize_intraday(obs, kBase);
    const DeseasonalizationDiagnostics& d = r.diagnostics;

    REQUIRE(d.n == 4);
    REQUIRE(d.total_count == 4);
    REQUIRE(d.total_sum == 60);
    REQUIRE(d.bins_per_session == 2);
    REQUIRE(d.num_nonempty_bins == 2);
    REQUIRE(d.grand_mean == Approx(15.0));

    REQUIRE(r.profile.size() == 2);
    REQUIRE(r.profile[0].count == 2);
    REQUIRE(r.profile[0].sum == 20);
    REQUIRE(r.profile[0].bin_mean == Approx(10.0));
    REQUIRE(r.profile[0].seasonal_factor == Approx(2.0 / 3.0));
    REQUIRE(r.profile[1].count == 2);
    REQUIRE(r.profile[1].sum == 40);
    REQUIRE(r.profile[1].bin_mean == Approx(20.0));
    REQUIRE(r.profile[1].seasonal_factor == Approx(4.0 / 3.0));

    REQUIRE(d.min_factor == Approx(2.0 / 3.0));
    REQUIRE(d.max_factor == Approx(4.0 / 3.0));
    REQUIRE(d.seasonality_strength == Approx(2.0 / 3.0));
    REQUIRE(d.seasonality_strength > 0.0);  // seasonality IS resolvable at this scale

    // Every deseasonalized value collapses to the grand mean (15).
    REQUIRE(r.points.size() == 4);
    for (const DeseasonalizedPoint& p : r.points) {
        REQUIRE(p.deseasonalized == Approx(15.0));
    }
    REQUIRE(d.replay_hash != 0);
}

TEST_CASE("offline deseasonalization: scenario hash changes when the DECLARED "
          "time scale or session policy changes",
          "[offline][timebase][falsifier][scenario-hash]") {
    // THE MOC FALSIFIER. Identical raw samples; only the DECLARED time-base
    // changes. A normalized timestamp is NOT self-sufficient: re-declaring the
    // time scale or session policy yields a different scenario.
    const auto obs = make_obs(kBinConstant);
    const std::uint64_t base_hash = deseasonalize_intraday(obs, kBase).diagnostics.replay_hash;

    // (1) Declared time SCALE: collapse to ONE bin (bins_per_session = 1). At this
    // coarse scale the diurnal structure is unresolvable: no flattening, strength
    // 0, and a different scenario hash.
    TimeBasePolicy coarse = kBase;
    coarse.bins_per_session = 1;
    const DeseasonalizationReplayResult rc = deseasonalize_intraday(obs, coarse);
    REQUIRE(rc.diagnostics.bins_per_session == 1);
    REQUIRE(rc.diagnostics.num_nonempty_bins == 1);
    REQUIRE(rc.diagnostics.seasonality_strength == Approx(0.0));
    // Not flattened: the raw 10/20 alternation survives (factor == 1 everywhere).
    REQUIRE(rc.points[0].deseasonalized == Approx(10.0));
    REQUIRE(rc.points[1].deseasonalized == Approx(20.0));
    REQUIRE(rc.diagnostics.replay_hash != base_hash);

    // (2) Declared session LENGTH: widen the session span to 4 ticks. Now both
    // tod 0 and tod 1 fall in bin 0, changing the profile and the scenario.
    TimeBasePolicy wide = kBase;
    wide.session_length_ticks = 4;
    const DeseasonalizationReplayResult rw = deseasonalize_intraday(obs, wide);
    REQUIRE(rw.diagnostics.num_nonempty_bins == 1);  // both tods land in bin 0
    REQUIRE(rw.diagnostics.replay_hash != base_hash);

    // (3) Declared session-OPEN phase: shifting the open by one tick swaps which
    // tod maps to which bin. Same samples, different declared phase, different
    // scenario.
    TimeBasePolicy shifted = kBase;
    shifted.session_open_ticks = 1;
    const DeseasonalizationReplayResult rs = deseasonalize_intraday(obs, shifted);
    REQUIRE(rs.diagnostics.replay_hash != base_hash);

    // (4) Declared clock CLASS: the same binning under a different declared class
    // is a DISTINCT scenario -- the class must be declared, not assumed.
    TimeBasePolicy wall = kBase;
    wall.clock = TimeScaleClass::WallClock;
    const DeseasonalizationReplayResult rcl = deseasonalize_intraday(obs, wall);
    REQUIRE(rcl.diagnostics.clock == TimeScaleClass::WallClock);
    REQUIRE(rcl.diagnostics.replay_hash != base_hash);
}

TEST_CASE("offline deseasonalization: scenario hash is deterministic",
          "[offline][timebase][determinism]") {
    const auto obs = make_obs(kBinConstant);

    // Same ordered input + declared policy -> identical hash, every time and
    // across instances.
    const IntradaySeasonalityDeseasonalizationReplay a(obs, kBase);
    const IntradaySeasonalityDeseasonalizationReplay b(obs, kBase);
    const std::uint64_t h1 = a.run().diagnostics.replay_hash;
    const std::uint64_t h2 = a.hash();  // idempotent
    const std::uint64_t h3 = b.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // The free function agrees with the wrapper.
    REQUIRE(deseasonalize_intraday(obs, kBase).diagnostics.replay_hash == h1);

    // A genuinely different series (one extra sample) changes the hash; the count
    // fold prevents a prefix from colliding.
    auto extended = kBinConstant;
    extended.push_back({0, 10});
    REQUIRE(deseasonalize_intraday(make_obs(extended), kBase).diagnostics.replay_hash != h1);
}

TEST_CASE("offline deseasonalization: hash is order-sensitive but the seasonal "
          "profile is order-invariant",
          "[offline][timebase][determinism][order]") {
    // Same multiset of observations, two different arrival orders.
    const auto ordered = make_obs({{0, 10}, {1, 20}, {0, 10}, {1, 20}});
    const auto reordered = make_obs({{1, 20}, {0, 10}, {1, 20}, {0, 10}});

    const DeseasonalizationReplayResult ro = deseasonalize_intraday(ordered, kBase);
    const DeseasonalizationReplayResult rr = deseasonalize_intraday(reordered, kBase);

    // The seasonal PROFILE is a set statistic: invariant to arrival order.
    REQUIRE(ro.diagnostics.grand_mean == Approx(rr.diagnostics.grand_mean));
    REQUIRE(ro.diagnostics.num_nonempty_bins == rr.diagnostics.num_nonempty_bins);
    REQUIRE(ro.diagnostics.min_factor == Approx(rr.diagnostics.min_factor));
    REQUIRE(ro.diagnostics.max_factor == Approx(rr.diagnostics.max_factor));
    REQUIRE(ro.diagnostics.seasonality_strength ==
            Approx(rr.diagnostics.seasonality_strength));

    // But the ORDERED deseasonalized output differs (points are in input order)...
    REQUIRE(ro.points[0].value == 10);
    REQUIRE(rr.points[0].value == 20);
    // ...so the order-sensitive scenario hash differs.
    REQUIRE(ro.diagnostics.replay_hash != rr.diagnostics.replay_hash);
}

TEST_CASE("offline deseasonalization: a single-bin series shows no seasonality",
          "[offline][timebase][deseasonalization]") {
    // Four declared bins but every observation lands in bin 0: the lone nonempty
    // bin has factor 1, strength 0, and deseasonalization is the identity.
    TimeBasePolicy four = kBase;
    four.session_length_ticks = 4;
    four.bins_per_session = 4;
    const auto obs = make_obs({{0, 7}, {0, 7}, {0, 7}});
    const DeseasonalizationReplayResult r = deseasonalize_intraday(obs, four);

    REQUIRE(r.diagnostics.bins_per_session == 4);
    REQUIRE(r.diagnostics.num_nonempty_bins == 1);
    REQUIRE(r.diagnostics.grand_mean == Approx(7.0));
    REQUIRE(r.diagnostics.seasonality_strength == Approx(0.0));
    REQUIRE(r.profile[0].seasonal_factor == Approx(1.0));
    for (const DeseasonalizedPoint& p : r.points) {
        REQUIRE(p.bin == 0);
        REQUIRE(p.deseasonalized == Approx(7.0));
    }
}

TEST_CASE("offline deseasonalization: deseasonalizing does not mutate the input",
          "[offline][timebase][readonly]") {
    std::vector<TodObservation> obs = make_obs(kBinConstant);
    const std::vector<TodObservation> before = obs;

    const DeseasonalizationReplayResult r = deseasonalize_intraday(obs, kBase);
    (void)r;

    // (d) NO mutation of the input series: the diagnostic is offline/read-only and
    // cannot alter matching/fills because it never touches engine state.
    REQUIRE(obs.size() == before.size());
    for (std::size_t i = 0; i < obs.size(); ++i) {
        REQUIRE(obs[i].seq == before[i].seq);
        REQUIRE(obs[i].tod_tick == before[i].tod_tick);
        REQUIRE(obs[i].value == before[i].value);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<FeedbackControl>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::TodObservation>,
                  "TodObservation must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::TimeBasePolicy>,
                  "TimeBasePolicy must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::SeasonalBin>,
                  "SeasonalBin must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::DeseasonalizedPoint>,
                  "DeseasonalizedPoint must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::DeseasonalizationDiagnostics>,
                  "DeseasonalizationDiagnostics must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<offline::IntradaySeasonalityDeseasonalizationReplay>,
        "IntradaySeasonalityDeseasonalizationReplay must not be a matching input");

    // The diagnostic returns a plain value, not engine state (compile-time
    // evidence of the offline boundary).
    static_assert(
        std::is_same_v<decltype(deseasonalize_intraday(obs, kBase)),
                       offline::DeseasonalizationReplayResult>,
        "deseasonalization must return derived diagnostics, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::DeseasonalizationDiagnostics>,
                  "DeseasonalizationDiagnostics must be pure data");
}

TEST_CASE("offline deseasonalization: empty and defensive inputs are well-defined",
          "[offline][timebase][determinism]") {
    // Empty series -> well-defined zeros, but still a stable deterministic hash.
    const std::vector<TodObservation> none;
    const IntradaySeasonalityDeseasonalizationReplay empty(none, kBase);
    REQUIRE(empty.empty());
    const DeseasonalizationDiagnostics ed = empty.run().diagnostics;
    REQUIRE(ed.n == 0);
    REQUIRE(ed.total_count == 0);
    REQUIRE(ed.num_nonempty_bins == 0);
    REQUIRE(ed.grand_mean == Approx(0.0));
    REQUIRE(ed.seasonality_strength == Approx(0.0));
    REQUIRE(ed.replay_hash == empty.hash());  // deterministic

    // Defensive sanitation: a declared grid of 0 bins is treated as 1 so the
    // binning is always well-defined.
    TimeBasePolicy degenerate = kBase;
    degenerate.bins_per_session = 0;
    const auto obs = make_obs(kBinConstant);
    REQUIRE(deseasonalize_intraday(obs, degenerate).diagnostics.bins_per_session == 1);
}
