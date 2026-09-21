#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "offline/arrival_point_process_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// (the MOC routes Hawkes / Poisson intensities, conditional durations, and
// inter-event-time models, all under one disposition: OFFLINE feature /
// calibration / diagnostic over replayed traces, NEVER a matching-kernel arrival
// generator).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (a) the replay hash is deterministic AND order-sensitive: the same ordered
//       event stream -> the same hash; a reordering of the SAME events -> a
//       different hash (and a different inter-arrival / intensity path);
//   (b) the DOMAIN STATISTICS are correct on a hand-constructed stream with KNOWN
//       arrival times: integer INTER-ARRIVAL durations, the running COUNTING
//       PROCESS N(t), the EMPIRICAL INTENSITY N(T)/T, and the exact self-exciting
//       HAWKES intensity recursion lambda_i = mu + alpha * R_i with
//       R_1 = 0, R_i = 2^{-decay_shift*gap_i} * (R_{i-1} + 1), all hand-checked;
//   (c) the diagnostic is strictly read-only / non-feedback: analysis does not
//       mutate the input stream, the offline types carry no matching-engine
//       coupling (static_assert discriminator + a POSITIVE control type), and the
//       intensity/duration values are one-way OUT statistics, never a matching
//       input, an arrival generator, or a live trading signal.
//
// Depends only on the self-contained header
// `offline/arrival_point_process_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using Catch::Approx;
using offline::ArrivalDiagnostics;
using offline::ArrivalEvent;
using offline::ArrivalPointProcessReplay;
using offline::ArrivalReplayResult;
using offline::EventStat;
using offline::EventTime;
using offline::HawkesParams;
using offline::PointProcessKind;

// Build an ordered event stream from raw integer arrival times. seq = position;
// mark = 0 unless specified, so a hand-constructed stream has known, exactly
// computable inter-arrival and Hawkes behaviour.
std::vector<ArrivalEvent> make_stream(const std::vector<EventTime>& times) {
    std::vector<ArrivalEvent> out;
    out.reserve(times.size());
    std::uint64_t seq = 0;
    for (EventTime t : times) {
        out.push_back(ArrivalEvent{seq, t, 0});
        ++seq;
    }
    return out;
}

// Fixed Hawkes parameters used throughout: mu = 1, alpha = 2, decay_shift = 1, so
// the exponential kernel halves per tick (kernel = 2^{-gap}). With these, the
// excitation R and intensity lambda over a stream are exact dyadic rationals.
constexpr HawkesParams kParams{/*base_mu=*/1, /*excite_alpha=*/2, /*decay_shift=*/1};

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (c) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<ArrivalDiagnostics>,
              "ArrivalDiagnostics must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ArrivalEvent>,
              "ArrivalEvent must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<ArrivalReplayResult,
                          decltype(&ArrivalPointProcessReplay::analyze),
                          const ArrivalPointProcessReplay&>,
    "analyze() must be callable on a const engine (read-only)");

TEST_CASE("offline arrivals: exact self-exciting Hawkes intensity recursion",
          "[offline][arrival][hawkes][intensity]") {
    // Arrival times 0,1,2,4 -> gaps -,1,1,2. With (mu=1, alpha=2, decay_shift=1)
    // the exact recursion is R = 0, 0.5, 0.75, 0.4375 and
    // lambda = mu + alpha*R = 1, 2, 2.5, 1.875.
    const auto stream = make_stream({0, 1, 2, 4});
    const ArrivalPointProcessReplay engine(stream, kParams);
    const ArrivalReplayResult r = engine.analyze();

    REQUIRE(r.events.size() == 4);
    REQUIRE(r.diagnostics.n == 4);
    REQUIRE(r.diagnostics.process == PointProcessKind::SelfExcitingHawkes);

    REQUIRE(r.events[0].excitation == Approx(0.0));
    REQUIRE(r.events[1].excitation == Approx(0.5));
    REQUIRE(r.events[2].excitation == Approx(0.75));
    REQUIRE(r.events[3].excitation == Approx(0.4375));

    REQUIRE(r.events[0].intensity == Approx(1.0));
    REQUIRE(r.events[1].intensity == Approx(2.0));
    REQUIRE(r.events[2].intensity == Approx(2.5));
    REQUIRE(r.events[3].intensity == Approx(1.875));

    // Peak intensity is at the tightest cluster (event index 2); the last event
    // arrives after a longer gap so its intensity has decayed.
    REQUIRE(r.diagnostics.max_intensity == Approx(2.5));
    REQUIRE(r.diagnostics.final_intensity == Approx(1.875));
    REQUIRE(r.diagnostics.replay_hash != 0);
}

TEST_CASE("offline arrivals: integer inter-arrival durations and counting process",
          "[offline][arrival][interarrival][counting]") {
    // Arrival times 0,1,2,4 -> gaps 1,1,2; span 4; N at each event = 1,2,3,4.
    const auto stream = make_stream({0, 1, 2, 4});
    const ArrivalReplayResult r = offline::analyze_arrivals(stream, kParams);
    const ArrivalDiagnostics& d = r.diagnostics;

    // Exact integer / event-time arithmetic.
    REQUIRE(d.first_time == 0);
    REQUIRE(d.last_time == 4);
    REQUIRE(d.total_span == 4);
    REQUIRE(d.total_interarrival == 4);  // 1 + 1 + 2
    REQUIRE(d.min_interarrival == 1);
    REQUIRE(d.max_interarrival == 2);
    REQUIRE(d.mean_interarrival == Approx(4.0 / 3.0));

    // Per-event reconstructed gaps and running counting process N.
    REQUIRE(r.events[0].interarrival == 0);
    REQUIRE(r.events[1].interarrival == 1);
    REQUIRE(r.events[2].interarrival == 1);
    REQUIRE(r.events[3].interarrival == 2);
    REQUIRE(r.events[0].count_through == 1);
    REQUIRE(r.events[1].count_through == 2);
    REQUIRE(r.events[2].count_through == 3);
    REQUIRE(r.events[3].count_through == 4);

    // Empirical intensity = counting-process rate N(T)/T = 4/4 = 1.0.
    REQUIRE(d.empirical_intensity == Approx(1.0));
}

TEST_CASE("offline arrivals: counting process N(t) query at arbitrary times",
          "[offline][arrival][counting]") {
    // N(t) = number of arrivals with time <= t. Hand-checked for times 0,1,2,4.
    const auto stream = make_stream({0, 1, 2, 4});
    const ArrivalPointProcessReplay engine(stream, kParams);

    REQUIRE(engine.count_through(-1) == 0);  // before the first arrival
    REQUIRE(engine.count_through(0) == 1);
    REQUIRE(engine.count_through(1) == 2);
    REQUIRE(engine.count_through(2) == 3);
    REQUIRE(engine.count_through(3) == 3);   // nothing arrives at t=3
    REQUIRE(engine.count_through(4) == 4);
    REQUIRE(engine.count_through(100) == 4);  // all arrivals counted

    // The free function agrees with the wrapper.
    REQUIRE(offline::count_through_time(stream, 1) == 2);
}

TEST_CASE("offline arrivals: a tighter cluster yields a higher peak intensity",
          "[offline][arrival][hawkes][clustering]") {
    // Three events all one tick apart cluster harder than three events two ticks
    // apart, so the self-exciting peak intensity is strictly higher. This is the
    // qualitative self-excitation property, checked with exact values.
    //   tight (gaps 1,1): R = 0, 0.5, 0.75 -> lambda peak = 1 + 2*0.75 = 2.5
    //   loose (gaps 2,2): R = 0, 0.25, 0.3125 -> lambda peak = 1 + 2*0.3125 = 1.625
    const auto tight = make_stream({0, 1, 2});
    const auto loose = make_stream({0, 2, 4});

    const ArrivalDiagnostics dt =
        offline::analyze_arrivals(tight, kParams).diagnostics;
    const ArrivalDiagnostics dl =
        offline::analyze_arrivals(loose, kParams).diagnostics;

    REQUIRE(dt.max_intensity == Approx(2.5));
    REQUIRE(dl.max_intensity == Approx(1.625));
    REQUIRE(dt.max_intensity > dl.max_intensity);
}

TEST_CASE("offline arrivals: replay-hash determinism",
          "[offline][arrival][determinism]") {
    const auto stream = make_stream({0, 1, 2, 4});

    // Same ordered input -> identical hash, every time and across instances.
    const ArrivalPointProcessReplay a(stream, kParams);
    const ArrivalPointProcessReplay b(stream, kParams);
    const std::uint64_t h1 = a.analyze().diagnostics.replay_hash;
    const std::uint64_t h2 = a.hash();  // idempotent
    const std::uint64_t h3 = b.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // The free function agrees with the wrapper.
    REQUIRE(offline::analyze_arrivals(stream, kParams).diagnostics.replay_hash == h1);

    // A genuinely different stream (one extra event) changes the hash, and the
    // count fold prevents a prefix from colliding.
    const auto extended = make_stream({0, 1, 2, 4, 8});
    REQUIRE(ArrivalPointProcessReplay(extended, kParams).hash() != h1);

    // Different Hawkes parameters change the hash even on the same stream.
    const HawkesParams slower{/*base_mu=*/1, /*excite_alpha=*/2, /*decay_shift=*/2};
    REQUIRE(ArrivalPointProcessReplay(stream, slower).hash() != h1);
}

TEST_CASE("offline arrivals: hash is order-sensitive and reordering changes the "
          "reconstructed statistics",
          "[offline][arrival][determinism][order]") {
    const auto ordered = make_stream({0, 1, 2, 4});
    // The SAME multiset of arrival times, replayed in reverse: the inter-arrival
    // reconstruction and the intensity path change even though the set of times
    // is identical.
    const auto reordered = make_stream({4, 2, 1, 0});

    const ArrivalReplayResult ro = offline::analyze_arrivals(ordered, kParams);
    const ArrivalReplayResult rr = offline::analyze_arrivals(reordered, kParams);

    // Order-sensitive integrity: the replay hash differs.
    REQUIRE(ro.diagnostics.replay_hash != rr.diagnostics.replay_hash);

    // And the change of arrival order genuinely changes the reconstruction: the
    // forward stream has a positive first gap, the reversed stream a negative one,
    // and the spans are negated.
    REQUIRE(ro.events[1].interarrival == 1);
    REQUIRE(rr.events[1].interarrival == -2);
    REQUIRE(ro.diagnostics.total_span == 4);
    REQUIRE(rr.diagnostics.total_span == -4);
}

TEST_CASE("offline arrivals: analysis does not mutate the input stream",
          "[offline][arrival][readonly]") {
    std::vector<ArrivalEvent> stream = make_stream({0, 1, 2, 4});
    const std::vector<ArrivalEvent> before = stream;

    const ArrivalReplayResult r = offline::analyze_arrivals(stream, kParams);
    (void)r;

    // (c) NO mutation of the input stream: analysis is offline/read-only and
    // cannot alter matching/fills because it never touches engine state.
    REQUIRE(stream.size() == before.size());
    for (std::size_t i = 0; i < stream.size(); ++i) {
        REQUIRE(stream[i].seq == before[i].seq);
        REQUIRE(stream[i].time == before[i].time);
        REQUIRE(stream[i].mark == before[i].mark);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<FeedbackControl>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::ArrivalEvent>,
                  "ArrivalEvent must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::EventStat>,
                  "EventStat must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::ArrivalDiagnostics>,
                  "ArrivalDiagnostics must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::ArrivalReplayResult>,
                  "ArrivalReplayResult must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<offline::ArrivalPointProcessReplay>,
        "ArrivalPointProcessReplay must not be a matching input");

    // The analyzer returns a plain value, not engine state (compile-time evidence
    // of the offline boundary).
    static_assert(
        std::is_same_v<decltype(offline::analyze_arrivals(stream, kParams)),
                       offline::ArrivalReplayResult>,
        "analysis must return derived diagnostics, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::ArrivalDiagnostics>,
                  "ArrivalDiagnostics must be pure data");
}

TEST_CASE("offline arrivals: empty and single-event inputs are well-defined",
          "[offline][arrival][determinism]") {
    // Empty stream -> well-defined zeros, but still a stable deterministic hash.
    const std::vector<ArrivalEvent> none;
    const ArrivalPointProcessReplay empty(none, kParams);
    REQUIRE(empty.empty());
    const ArrivalDiagnostics ed = empty.analyze().diagnostics;
    REQUIRE(ed.n == 0);
    REQUIRE(ed.total_span == 0);
    REQUIRE(ed.empirical_intensity == Approx(0.0));
    REQUIRE(ed.max_intensity == Approx(0.0));
    REQUIRE(ed.replay_hash == empty.hash());  // deterministic
    REQUIRE(empty.count_through(0) == 0);

    // A single event: no inter-arrival gap, N=1, intensity = mu (no prior excite).
    const auto one = make_stream({7});
    const ArrivalReplayResult r = offline::analyze_arrivals(one, kParams);
    REQUIRE(r.diagnostics.n == 1);
    REQUIRE(r.events[0].interarrival == 0);
    REQUIRE(r.events[0].count_through == 1);
    REQUIRE(r.events[0].excitation == Approx(0.0));
    REQUIRE(r.events[0].intensity == Approx(1.0));  // mu only
    REQUIRE(r.diagnostics.total_span == 0);
    REQUIRE(r.diagnostics.mean_interarrival == Approx(0.0));  // < 2 events
    REQUIRE(r.diagnostics.empirical_intensity == Approx(0.0));  // span 0
}
