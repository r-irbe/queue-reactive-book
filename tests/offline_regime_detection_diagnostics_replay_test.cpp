#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/regime_detection_diagnostics_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the open code channel
// (the MOC records that "No offline regime/changepoint diagnostic fixture
// exists", so until a detection-latency / false-alarm artifact is produced the
// channel cannot close).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (a) the replay hash is deterministic AND order-sensitive: the same ordered
//       series -> the same hash; a reordering of the SAME samples -> a different
//       hash (and, in the constructed case, different alarm timing);
//   (b) the VALIDATION METRICS are correct on a hand-constructed series with
//       KNOWN injected changepoints: detection LATENCY (samples from the true
//       changepoint to the first in-segment alarm), FALSE-ALARM count/rate
//       (baseline-segment + redundant repeat alarms), and MISSED-DETECTION count
//       (a changepoint segment with no alarm);
//   (c) the diagnostic is strictly read-only / non-feedback: scoring does not
//       mutate the input series, the offline types carry no matching-engine
//       coupling (static_assert discriminator + a POSITIVE control type), and
//       the score is a one-way OUT validation metric, never a matching input or
//       a live trading signal.
//
// Depends only on the self-contained header
// `offline/regime_detection_diagnostics_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::ChangepointScore;
using offline::CusumArm;
using offline::CusumParams;
using offline::RegimeAlarm;
using offline::RegimeDetectionDiagnosticsReplay;
using offline::RegimeDiagnostics;
using offline::RegimeReplayResult;
using offline::RegimeSample;
using offline::SampleValue;

// Build a series from raw integer values (seq = position) so a hand-constructed
// series has known, exactly computable CUSUM behaviour.
std::vector<RegimeSample> make_series(const std::vector<SampleValue>& values) {
    std::vector<RegimeSample> out;
    out.reserve(values.size());
    std::uint64_t seq = 0;
    for (SampleValue v : values) {
        out.push_back(RegimeSample{seq, v});
        ++seq;
    }
    return out;
}

// Fixed detector parameters used throughout: mu0 = 0, slack k = 1, threshold
// h = 4. With these, a sustained +3 mean shift accumulates 2 per sample and
// crosses h two samples after it begins.
constexpr CusumParams kParams{/*target_mean=*/0, /*slack=*/1, /*threshold=*/4};

const ChangepointScore* find_score(const RegimeReplayResult& r, std::size_t cp) {
    for (const ChangepointScore& s : r.scores)
        if (s.cp_index == cp) return &s;
    return nullptr;
}

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (c) Compile-time read-only / non-feedback guarantees mirrored in the test TU
// so a regression that makes the diagnostic mutating or feedback-coupled fails
// THIS translation unit's compile. The positive control proves the trait
// genuinely discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<RegimeDiagnostics>,
              "RegimeDiagnostics must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<RegimeSample>,
              "RegimeSample must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<RegimeReplayResult,
                          decltype(&RegimeDetectionDiagnosticsReplay::score),
                          const RegimeDetectionDiagnosticsReplay&>,
    "score() must be callable on a const engine (read-only)");

TEST_CASE("offline regime diagnostics: clean detection latency, no false alarms",
          "[offline][regime][latency]") {
    // Baseline mean 0 (indices 0..4), then a +3 mean shift injected at index 5
    // (indices 5..9). With (mu0=0,k=1,h=4) the upper CUSUM accumulates 2/sample
    // from the shift and crosses h at index 7 -> latency 2, no false alarm.
    const auto series = make_series({0, 0, 0, 0, 0, 3, 3, 3, 3, 3});
    const std::vector<std::size_t> changepoints{5};

    const RegimeDetectionDiagnosticsReplay engine(series, changepoints, kParams);
    const RegimeReplayResult r = engine.score();
    const RegimeDiagnostics& d = r.diagnostics;

    REQUIRE(d.n == 10);
    REQUIRE(d.num_changepoints == 1);
    REQUIRE(d.num_alarms == 1);
    REQUIRE(r.alarms.size() == 1);
    REQUIRE(r.alarms[0].index == 7);
    REQUIRE(r.alarms[0].arm == CusumArm::Upper);

    const ChangepointScore* s = find_score(r, 5);
    REQUIRE(s != nullptr);
    REQUIRE(s->detected);
    REQUIRE(s->alarm_index == 7);
    REQUIRE(s->latency == 2);

    REQUIRE(d.detected_count == 1);
    REQUIRE(d.missed_count == 0);
    REQUIRE(d.false_alarm_count == 0);
    REQUIRE(d.total_latency == 2);
    REQUIRE(d.mean_detection_latency == Approx(2.0));
    REQUIRE(d.detection_rate == Approx(1.0));
    REQUIRE(d.false_alarm_rate == Approx(0.0));
    REQUIRE(d.replay_hash != 0);
}

TEST_CASE("offline regime diagnostics: false alarm AND missed detection scored",
          "[offline][regime][latency][falsealarm][missed]") {
    // Baseline with an in-control SPIKE at index 1 (value 6) that trips the
    // upper CUSUM (6-1=5 > 4) -> a FALSE ALARM (no changepoint there). A true +3
    // shift at index 5 is detected at index 7 (latency 2). A second changepoint
    // at index 8 (back to mean 0, only two samples) never accumulates past h ->
    // a MISSED detection.
    const auto series = make_series({0, 6, 0, 0, 0, 3, 3, 3, 0, 0});
    const std::vector<std::size_t> changepoints{5, 8};

    const RegimeReplayResult r =
        offline::score_regime_detection(series, changepoints, kParams);
    const RegimeDiagnostics& d = r.diagnostics;

    REQUIRE(d.num_changepoints == 2);
    REQUIRE(d.num_alarms == 2);
    REQUIRE(r.alarms[0].index == 1);  // false-alarm spike
    REQUIRE(r.alarms[1].index == 7);  // detects the index-5 changepoint

    const ChangepointScore* s5 = find_score(r, 5);
    REQUIRE(s5 != nullptr);
    REQUIRE(s5->detected);
    REQUIRE(s5->latency == 2);

    const ChangepointScore* s8 = find_score(r, 8);
    REQUIRE(s8 != nullptr);
    REQUIRE_FALSE(s8->detected);  // missed

    REQUIRE(d.detected_count == 1);
    REQUIRE(d.missed_count == 1);
    REQUIRE(d.false_alarm_count == 1);  // the baseline spike
    REQUIRE(d.total_latency == 2);
    REQUIRE(d.mean_detection_latency == Approx(2.0));
    REQUIRE(d.detection_rate == Approx(0.5));     // 1 of 2 detected
    REQUIRE(d.false_alarm_rate == Approx(0.1));   // 1 of 10 samples
}

TEST_CASE("offline regime diagnostics: redundant repeat alarms are false alarms",
          "[offline][regime][falsealarm]") {
    // A large sustained +5 shift at index 2 alarms repeatedly. Only the FIRST
    // alarm in the post-change segment detects the changepoint; later alarms in
    // the same segment are redundant repeats and count as false alarms.
    const auto series = make_series({0, 0, 5, 5, 5, 5, 5, 5});
    const std::vector<std::size_t> changepoints{2};

    const RegimeReplayResult r =
        offline::score_regime_detection(series, changepoints, kParams);
    const RegimeDiagnostics& d = r.diagnostics;

    REQUIRE(r.alarms.size() == 3);     // indices 3, 5, 7
    REQUIRE(r.alarms[0].index == 3);
    REQUIRE(r.alarms[1].index == 5);
    REQUIRE(r.alarms[2].index == 7);

    const ChangepointScore* s = find_score(r, 2);
    REQUIRE(s != nullptr);
    REQUIRE(s->detected);
    REQUIRE(s->alarm_index == 3);
    REQUIRE(s->latency == 1);

    REQUIRE(d.detected_count == 1);
    REQUIRE(d.missed_count == 0);
    REQUIRE(d.false_alarm_count == 2);  // the two later repeats
    REQUIRE(d.num_alarms == 3);
}

TEST_CASE("offline regime diagnostics: a downward shift fires the lower arm",
          "[offline][regime][cusum]") {
    // A -3 mean shift at index 3 drives the LOWER statistic across h.
    const auto series = make_series({0, 0, 0, -3, -3, -3, -3});
    const std::vector<std::size_t> changepoints{3};

    const RegimeReplayResult r =
        offline::score_regime_detection(series, changepoints, kParams);

    REQUIRE(r.alarms.size() == 1);
    REQUIRE(r.alarms[0].arm == CusumArm::Lower);
    const ChangepointScore* s = find_score(r, 3);
    REQUIRE(s != nullptr);
    REQUIRE(s->detected);
    REQUIRE(s->latency == 2);  // index 5 - index 3
}

TEST_CASE("offline regime diagnostics: replay-hash determinism",
          "[offline][regime][determinism]") {
    const auto series = make_series({0, 0, 0, 0, 0, 3, 3, 3, 3, 3});
    const std::vector<std::size_t> cps{5};

    // Same ordered input -> identical hash, every time and across instances.
    const RegimeDetectionDiagnosticsReplay a(series, cps, kParams);
    const RegimeDetectionDiagnosticsReplay b(series, cps, kParams);
    const std::uint64_t h1 = a.score().diagnostics.replay_hash;
    const std::uint64_t h2 = a.hash();  // idempotent
    const std::uint64_t h3 = b.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // The free function agrees with the wrapper.
    REQUIRE(offline::score_regime_detection(series, cps, kParams).diagnostics.replay_hash ==
            h1);

    // A genuinely different series (one extra sample) changes the hash, and the
    // count fold prevents a prefix from colliding.
    const auto extended = make_series({0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3});
    REQUIRE(RegimeDetectionDiagnosticsReplay(extended, cps, kParams).hash() != h1);

    // Different detector parameters change the hash even on the same series.
    const CusumParams looser{/*target_mean=*/0, /*slack=*/1, /*threshold=*/8};
    REQUIRE(RegimeDetectionDiagnosticsReplay(series, cps, looser).hash() != h1);
}

TEST_CASE("offline regime diagnostics: hash is order-sensitive and reordering "
          "changes the scored metrics",
          "[offline][regime][determinism][order]") {
    // The change is concentrated late in the ordered series, so the detector
    // alarms at index 7.
    const auto ordered = make_series({0, 0, 0, 0, 0, 3, 3, 3, 3, 3});
    // Front-load the same multiset of values: the run of 3s now starts at the
    // front, so the alarm timing (and therefore the score) is different, even
    // though the multiset of sample VALUES is identical.
    const auto reordered = make_series({3, 3, 3, 3, 3, 0, 0, 0, 0, 0});
    const std::vector<std::size_t> cps{5};

    const RegimeReplayResult ro =
        offline::score_regime_detection(ordered, cps, kParams);
    const RegimeReplayResult rr =
        offline::score_regime_detection(reordered, cps, kParams);

    // Order-sensitive integrity: the replay hash differs.
    REQUIRE(ro.diagnostics.replay_hash != rr.diagnostics.replay_hash);

    // And the change of arrival order genuinely changes the alarm timing: the
    // front-loaded series alarms before the (true) changepoint at index 5, so it
    // misses the changepoint and the early alarm is a baseline false alarm.
    REQUIRE(ro.alarms[0].index == 7);
    REQUIRE(rr.alarms[0].index == 2);
    REQUIRE(ro.diagnostics.detected_count == 1);
    REQUIRE(ro.diagnostics.false_alarm_count == 0);
    REQUIRE(rr.diagnostics.detected_count == 0);
    REQUIRE(rr.diagnostics.missed_count == 1);
    REQUIRE(rr.diagnostics.false_alarm_count >= 1);
}

TEST_CASE("offline regime diagnostics: scoring does not mutate the input series",
          "[offline][regime][readonly]") {
    std::vector<RegimeSample> series =
        make_series({0, 6, 0, 0, 0, 3, 3, 3, 0, 0});
    const std::vector<RegimeSample> before = series;
    const std::vector<std::size_t> cps{5, 8};

    const RegimeReplayResult r =
        offline::score_regime_detection(series, cps, kParams);
    (void)r;

    // (c) NO mutation of the input series: scoring is offline/read-only and
    // cannot alter matching/fills because it never touches engine state.
    REQUIRE(series.size() == before.size());
    for (std::size_t i = 0; i < series.size(); ++i) {
        REQUIRE(series[i].seq == before[i].seq);
        REQUIRE(series[i].value == before[i].value);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<FeedbackControl>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::RegimeDiagnostics>,
                  "RegimeDiagnostics must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::RegimeSample>,
                  "RegimeSample must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::RegimeAlarm>,
                  "RegimeAlarm must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::ChangepointScore>,
                  "ChangepointScore must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<offline::RegimeDetectionDiagnosticsReplay>,
        "RegimeDetectionDiagnosticsReplay must not be a matching input");

    // The scorer returns a plain value, not engine state (compile-time evidence
    // of the offline boundary).
    static_assert(
        std::is_same_v<decltype(offline::score_regime_detection(series, cps, kParams)),
                       offline::RegimeReplayResult>,
        "scoring must return derived diagnostics, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::RegimeDiagnostics>,
                  "RegimeDiagnostics must be pure data");
}

TEST_CASE("offline regime diagnostics: empty and changepoint-free inputs are "
          "well-defined",
          "[offline][regime][determinism]") {
    // Empty series -> well-defined zeros, but still a stable deterministic hash.
    const std::vector<RegimeSample> none;
    const std::vector<std::size_t> no_cps;
    const RegimeDetectionDiagnosticsReplay empty(none, no_cps, kParams);
    REQUIRE(empty.empty());
    const RegimeDiagnostics ed = empty.score().diagnostics;
    REQUIRE(ed.n == 0);
    REQUIRE(ed.num_alarms == 0);
    REQUIRE(ed.num_changepoints == 0);
    REQUIRE(ed.detected_count == 0);
    REQUIRE(ed.false_alarm_rate == Approx(0.0));
    REQUIRE(ed.replay_hash == empty.hash());  // deterministic

    // A flat in-control series with no injected changepoints raises no alarm and
    // scores no false alarm: a correct detector is silent in control.
    const auto flat = make_series({0, 0, 0, 0, 0, 0});
    const RegimeDiagnostics fd =
        offline::score_regime_detection(flat, no_cps, kParams).diagnostics;
    REQUIRE(fd.num_alarms == 0);
    REQUIRE(fd.false_alarm_count == 0);
    REQUIRE(fd.detected_count == 0);
    REQUIRE(fd.missed_count == 0);
}
