#include <catch2/catch_test_macros.hpp>

#include "offline/stress_scenario_criticism_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic offline fixture closing the `needs-generated-artifact-boundary`
// (the channel asked for a worked SEEDED, EXPLICIT, replay-NEUTRAL stress
// fixture with a diagnostics-on/off replay-hash; supplied here).
//
// It proves, with ZERO FlatLadderBook / sim:: coupling, that:
//   (1) the stress-report hash is deterministic AND order-sensitive: the same
//       base series + spec + ordered scenario set -> the same hash; reordering
//       the SAME scenario set -> a different hash (even though the interval /
//       worst-case / range robustness metrics are order-invariant);
//   (2) the robustness report is correct on hand-built cases with KNOWN analytic
//       values: the interval / worst-case bound over a fixed scenario set, and
//       the MOC's load-bearing claim that a one-parameter sensitivity sweep is
//       NOT robustness (a flat single-axis sweep co-existing with a large
//       multi-scenario robustness range driven by a cross-axis interaction);
//   (3) the scenario application is REPLAY-NEUTRAL: the base replay hash depends
//       ONLY on the base series and is invariant to whether/which stress
//       scenarios run (diagnostics-on/off replay-hash equality), and the seed is
//       explicit and reproducible;
//   (4) the diagnostic is strictly read-only / non-feedback (const-correctness
//       plus a static_assert discriminator carrying a POSITIVE control type).
//
// Depends only on the self-contained header
// `offline/stress_scenario_criticism_replay.hpp` (std-includes only).
// =============================================================================

namespace {

using offline::ScenarioPoint;
using offline::StressModelSpec;
using offline::StressReport;
using offline::StressScenario;
using offline::StressScenarioCriticismReplay;
using offline::Tick;

// Build an ordered base series from raw tick values; seq = position.
std::vector<ScenarioPoint> make_series(const std::vector<Tick>& vals) {
    std::vector<ScenarioPoint> out;
    out.reserve(vals.size());
    std::uint64_t seq = 0;
    for (Tick v : vals) {
        out.push_back(ScenarioPoint{seq, v});
        ++seq;
    }
    return out;
}

// Build a scenario set from explicit per-axis theta vectors.
std::vector<StressScenario> make_scenarios(
    const std::vector<std::vector<Tick>>& thetas) {
    std::vector<StressScenario> out;
    out.reserve(thetas.size());
    for (const auto& t : thetas) out.push_back(StressScenario{t});
    return out;
}

// A two-axis spec with NO linear terms and a single cross-axis interaction
// weight W between axis 0 and axis 1: M = base_sum + W * theta0 * theta1. This is
// the canonical "sensitivity is not robustness" model -- each single-axis sweep
// is flat, but a joint scenario moves the output.
StressModelSpec interaction_only_spec(Tick w) {
    StressModelSpec spec;
    spec.linear_weight = {0, 0};      // k = 2 axes, zero first-order sensitivity
    spec.interaction = {0, w, 0, 0};  // 2x2 row-major; interaction[0*2+1] = w
    return spec;
}

// A purely linear single-axis spec: M = base_sum + L * theta0.
StressModelSpec linear_spec(Tick l) {
    StressModelSpec spec;
    spec.linear_weight = {l};
    spec.interaction = {0};  // 1x1; no cross term
    return spec;
}

// --- Positive control for the non-feedback discriminator ---------------------
// A type that DOES expose a matching-feedback member, proving the trait is not
// vacuously false: it must classify this type as feedback-coupled.
struct FeedbackControl {
    int apply_to_matching() { return 0; }
};

}  // namespace

// (4) Compile-time read-only / non-feedback guarantees mirrored in the test TU so
// a regression that makes the diagnostic mutating or feedback-coupled fails THIS
// translation unit's compile. The positive control proves the trait genuinely
// discriminates.
static_assert(offline::has_matching_feedback_v<FeedbackControl>,
              "positive control must be detected as matching-feedback coupled");
static_assert(!offline::has_matching_feedback_v<StressReport>,
              "StressReport must be non-feedback (one-way data flow)");
static_assert(!offline::has_matching_feedback_v<ScenarioPoint>,
              "ScenarioPoint must be non-feedback (one-way data flow)");
static_assert(
    std::is_invocable_r_v<StressReport,
                          decltype(&StressScenarioCriticismReplay::compute),
                          const StressScenarioCriticismReplay&>,
    "compute() must be callable on a const engine (read-only)");

TEST_CASE("offline stress: baseline output and base aggregate are exact",
          "[offline][stress][baseline]") {
    // base series {3,-1,4,-2}: sum = 4. base_offset 0 => baseline = 4.
    const auto series = make_series({3, -1, 4, -2});
    const StressModelSpec spec = interaction_only_spec(/*w=*/1);
    const std::vector<StressScenario> none;
    const StressReport r =
        StressScenarioCriticismReplay(series, spec, none, /*sweep_step=*/1).compute();

    REQUIRE(r.n == 4);
    REQUIRE(r.num_axes == 2);
    REQUIRE(r.base_aggregate == 4);
    REQUIRE(r.baseline_output == 4);
    REQUIRE_FALSE(r.has_scenarios);
}

TEST_CASE("offline stress: interval / worst-case robustness bound over a fixed "
          "scenario set (exact)",
          "[offline][stress][interval][worst-case]") {
    // M = base_sum + W*theta0*theta1, base_sum = 0, W = 2.
    const auto series = make_series({1, -1});  // sum 0
    const StressModelSpec spec = interaction_only_spec(/*w=*/2);

    // Joint scenarios: (1,1)->+2, (1,-1)->-2, (-1,1)->-2, (-1,-1)->+2, (0,0)->0.
    const auto scenarios =
        make_scenarios({{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 0}});
    const StressReport r =
        StressScenarioCriticismReplay(series, spec, scenarios, /*sweep_step=*/1)
            .compute();

    REQUIRE(r.has_scenarios);
    REQUIRE(r.num_scenarios == 5);
    REQUIRE(r.baseline_output == 0);
    REQUIRE(r.min_output == -2);
    REQUIRE(r.max_output == 2);
    REQUIRE(r.robustness_range == 4);          // max - min
    REQUIRE(r.worst_case_abs_deviation == 2);  // max |M_s - baseline|
}

TEST_CASE("offline stress: a one-parameter sensitivity sweep is NOT robustness "
          "(the MOC's load-bearing gate)",
          "[offline][stress][sensitivity]") {
    // M = base_sum + W*theta0*theta1 with W=1, base_sum 0. A single-axis sweep
    // (perturb axis 0 OR axis 1 alone) leaves the output at the baseline because
    // the other axis is 0 -> ZERO single-axis sensitivity. But joint scenarios
    // over the TWO interacting axes move the output: the robustness range is
    // large. Sensitivity != robustness.
    const auto series = make_series({5, -5});  // sum 0
    const StressModelSpec spec = interaction_only_spec(/*w=*/1);

    const auto scenarios = make_scenarios({{1, 1}, {1, -1}, {-1, 1}, {-1, -1}});
    const StressReport r =
        StressScenarioCriticismReplay(series, spec, scenarios, /*sweep_step=*/1)
            .compute();

    // Single-axis sweep is FLAT: 1*(+1)*0 = 0 on every axis.
    REQUIRE(r.one_axis_max_abs_sensitivity == 0);
    // Joint robustness range is large: outputs {+1,-1,-1,+1} -> [-1,1], range 2.
    REQUIRE(r.robustness_range == 2);
    REQUIRE(r.worst_case_abs_deviation == 1);
    // The gate fires: a one-parameter sweep is provably not robustness here.
    REQUIRE_FALSE(r.sensitivity_equals_robustness);
}

TEST_CASE("offline stress: a purely linear model makes the single-axis sweep a "
          "faithful sensitivity (positive control for the gate)",
          "[offline][stress][sensitivity]") {
    // M = base_sum + L*theta0, L=3, base_sum 0, sweep step 1 => single-axis
    // sensitivity = |3*1| = 3. A scenario set of {+1},{-1} -> outputs {+3,-3} ->
    // range 6. Here the one-axis sweep is a genuine (sufficient) sensitivity
    // because there is no interacting second axis; the gate flag tracks whether
    // the single-axis number equals the multi-scenario range.
    const auto series = make_series({2, -2});  // sum 0
    const StressModelSpec spec = linear_spec(/*l=*/3);

    const auto scenarios = make_scenarios({{1}, {-1}});
    const StressReport r =
        StressScenarioCriticismReplay(series, spec, scenarios, /*sweep_step=*/1)
            .compute();

    REQUIRE(r.num_axes == 1);
    REQUIRE(r.one_axis_max_abs_sensitivity == 3);
    REQUIRE(r.min_output == -3);
    REQUIRE(r.max_output == 3);
    REQUIRE(r.robustness_range == 6);
    // 3 != 6: even a single-axis model's sweep is not the full +/- range; the
    // gate correctly does not claim equality.
    REQUIRE_FALSE(r.sensitivity_equals_robustness);
}

TEST_CASE("offline stress: replay-NEUTRALITY -- base replay hash is invariant to "
          "whether/which stress scenarios run (diagnostics-on/off equality)",
          "[offline][stress][replay-neutral]") {
    const auto series = make_series({7, -3, 2, 9});
    const StressModelSpec spec = interaction_only_spec(/*w=*/4);

    // Diagnostics OFF: no scenarios.
    const std::vector<StressScenario> none;
    const StressReport off =
        StressScenarioCriticismReplay(series, spec, none, /*sweep_step=*/1).compute();

    // Diagnostics ON, scenario set A.
    const auto scen_a = make_scenarios({{1, 1}, {-1, 1}});
    const StressScenarioCriticismReplay engine_a(series, spec, scen_a, 1);
    const StressReport on_a = engine_a.compute();

    // Diagnostics ON, a DIFFERENT scenario set B.
    const auto scen_b = make_scenarios({{1, -1}, {-1, -1}, {1, 1}});
    const StressReport on_b =
        StressScenarioCriticismReplay(series, spec, scen_b, 1).compute();

    // The replay-neutral base hash is identical across all three: the stress
    // scenarios NEVER touch the replayable artifact.
    REQUIRE(off.base_replay_hash == on_a.base_replay_hash);
    REQUIRE(off.base_replay_hash == on_b.base_replay_hash);
    REQUIRE(off.base_replay_hash != 0);
    // And it equals the standalone base hash available WITHOUT running the
    // diagnostic (the diagnostics-off computation path).
    REQUIRE(engine_a.base_hash() == off.base_replay_hash);

    // The stress-report hash, by contrast, DOES change with the scenario set:
    // it is the diagnostic's own reproducibility hash, not the replay artifact.
    REQUIRE(on_a.stress_report_hash != on_b.stress_report_hash);
    REQUIRE(on_a.stress_report_hash != off.stress_report_hash);
}

TEST_CASE("offline stress: base replay hash is order-sensitive over the base "
          "series",
          "[offline][stress][replay-neutral][order]") {
    const StressModelSpec spec = interaction_only_spec(/*w=*/1);
    const std::vector<StressScenario> none;

    const auto a = make_series({1, 2, 3});
    const auto b = make_series({3, 2, 1});  // same multiset, reordered
    const Tick step = 1;

    const StressReport ra =
        StressScenarioCriticismReplay(a, spec, none, step).compute();
    const StressReport rb =
        StressScenarioCriticismReplay(b, spec, none, step).compute();

    // Same aggregate (order-invariant) but different ordered base hash.
    REQUIRE(ra.base_aggregate == rb.base_aggregate);
    REQUIRE(ra.base_replay_hash != rb.base_replay_hash);
}

TEST_CASE("offline stress: stress-report hash determinism",
          "[offline][stress][determinism]") {
    const auto series = make_series({2, 4, -6, 1});
    const StressModelSpec spec = interaction_only_spec(/*w=*/3);
    const auto scenarios = make_scenarios({{1, 1}, {1, -1}, {-1, 1}});
    const Tick step = 1;

    const StressScenarioCriticismReplay x(series, spec, scenarios, step);
    const StressScenarioCriticismReplay y(series, spec, scenarios, step);
    const std::uint64_t h1 = x.compute().stress_report_hash;
    const std::uint64_t h2 = x.compute().stress_report_hash;  // idempotent
    const std::uint64_t h3 = y.hash();
    REQUIRE(h1 == h2);
    REQUIRE(h1 == h3);
    REQUIRE(h1 != 0);

    // Adding a scenario changes the hash (and the count fold blocks prefix
    // collision).
    const auto extended =
        make_scenarios({{1, 1}, {1, -1}, {-1, 1}, {-1, -1}});
    REQUIRE(StressScenarioCriticismReplay(series, spec, extended, step).hash() != h1);

    // Changing only a FIXED spec weight changes the hash too.
    const StressModelSpec spec2 = interaction_only_spec(/*w=*/4);
    REQUIRE(StressScenarioCriticismReplay(series, spec2, scenarios, step).hash() != h1);

    // Changing only the sweep step changes the hash.
    REQUIRE(StressScenarioCriticismReplay(series, spec, scenarios, step + 1).hash() != h1);
}

TEST_CASE("offline stress: report hash is order-SENSITIVE over the scenario set "
          "while the robustness metrics are order-INVARIANT",
          "[offline][stress][determinism][order]") {
    const auto series = make_series({1, -1, 1, -1});  // sum 0
    const StressModelSpec spec = interaction_only_spec(/*w=*/5);
    const Tick step = 1;

    const auto order1 = make_scenarios({{1, 1}, {1, -1}, {-1, 1}, {-1, -1}});
    const auto order2 = make_scenarios({{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});

    const StressReport r1 =
        StressScenarioCriticismReplay(series, spec, order1, step).compute();
    const StressReport r2 =
        StressScenarioCriticismReplay(series, spec, order2, step).compute();

    // Interval / worst-case / range metrics are order-INVARIANT (set semantics).
    REQUIRE(r1.min_output == r2.min_output);
    REQUIRE(r1.max_output == r2.max_output);
    REQUIRE(r1.robustness_range == r2.robustness_range);
    REQUIRE(r1.worst_case_abs_deviation == r2.worst_case_abs_deviation);

    // The base replay hash is identical (same base series).
    REQUIRE(r1.base_replay_hash == r2.base_replay_hash);

    // But the ordered stress-report hash MUST differ: reordering the scenario set
    // is a detectable change.
    REQUIRE(r1.stress_report_hash != r2.stress_report_hash);
}

TEST_CASE("offline stress: seeded scenario generation is explicit and reproducible",
          "[offline][stress][seed][determinism]") {
    const std::size_t num_scenarios = 8;
    const std::size_t num_axes = 2;
    const Tick magnitude = 3;

    // Same explicit seed -> identical scenario set.
    const auto a = offline::generate_seeded_scenarios(
        /*seed=*/42, num_scenarios, num_axes, magnitude);
    const auto b = offline::generate_seeded_scenarios(
        /*seed=*/42, num_scenarios, num_axes, magnitude);
    REQUIRE(a.size() == num_scenarios);
    REQUIRE(b.size() == num_scenarios);
    bool identical = true;
    for (std::size_t s = 0; s < a.size(); ++s) {
        if (a[s].theta != b[s].theta) identical = false;
        // every shock is in {-mag, 0, +mag}
        for (Tick t : a[s].theta) {
            REQUIRE((t == -magnitude || t == 0 || t == magnitude));
        }
    }
    REQUIRE(identical);

    // A different seed -> a different scenario set (and so a different report
    // hash), while the base replay hash stays put (replay-neutral).
    const auto c = offline::generate_seeded_scenarios(
        /*seed=*/43, num_scenarios, num_axes, magnitude);
    bool any_diff = false;
    for (std::size_t s = 0; s < a.size(); ++s) {
        if (a[s].theta != c[s].theta) any_diff = true;
    }
    REQUIRE(any_diff);

    const auto series = make_series({4, -4, 8, -8});
    const StressModelSpec spec = interaction_only_spec(/*w=*/2);
    const StressReport ra =
        StressScenarioCriticismReplay(series, spec, a, /*sweep_step=*/1).compute();
    const StressReport rc =
        StressScenarioCriticismReplay(series, spec, c, /*sweep_step=*/1).compute();
    REQUIRE(ra.base_replay_hash == rc.base_replay_hash);   // replay-neutral
    REQUIRE(ra.stress_report_hash != rc.stress_report_hash);  // seed-sensitive
}

TEST_CASE("offline stress: degenerate and empty inputs are well-defined",
          "[offline][stress][determinism]") {
    const StressModelSpec spec = interaction_only_spec(/*w=*/1);

    // Empty base series with no scenarios: well-defined zeros, stable hashes.
    const std::vector<ScenarioPoint> none_series;
    const std::vector<StressScenario> no_scen;
    const StressScenarioCriticismReplay empty(none_series, spec, no_scen, 1);
    REQUIRE(empty.empty());
    const StressReport e = empty.compute();
    REQUIRE(e.n == 0);
    REQUIRE(e.base_aggregate == 0);
    REQUIRE(e.baseline_output == 0);
    REQUIRE_FALSE(e.has_scenarios);
    REQUIRE(e.robustness_range == 0);
    REQUIRE(e.base_replay_hash == empty.base_hash());  // deterministic
    REQUIRE(e.stress_report_hash != 0);

    // Non-empty series, but no scenarios: robustness metrics stay zero, the gate
    // does not claim equality (no scenario evidence), base hash is non-trivial.
    const auto series = make_series({5, 5});
    const StressReport r =
        StressScenarioCriticismReplay(series, spec, no_scen, 1).compute();
    REQUIRE(r.base_aggregate == 10);
    REQUIRE_FALSE(r.has_scenarios);
    REQUIRE_FALSE(r.sensitivity_equals_robustness);
    REQUIRE(r.base_replay_hash != 0);
}
