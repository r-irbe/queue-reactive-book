#pragma once

// =============================================================================
// Offline stress-testing / robustness / scenario-criticism REPLAY fixture.
//
// by providing the missing worked, SEEDED, EXPLICIT, replay-NEUTRAL stress
// fixture the MOC names but did not yet implement: a deterministic seeded
// scenario-perturbation of a replayable input series, an interval / worst-case
// robustness bound over a fixed scenario set, and a one-parameter sensitivity
// sweep that is provably NOT robustness -- all carried by a diagnostics-on/off
// stable base replay hash.
//
// SCOPE / BOUNDARY (hard invariant the MOC defends):
//   This is OFFLINE model-risk tooling ONLY. It is SELF-CONTAINED and
//   header-only, with std-includes ONLY. It has NO dependency on
//   `FlatLadderBook`, the matching kernel, or ANY `sim::` symbol -- it includes
//   nothing from `sim/`. The matching kernel is *deterministic and
//   event-driven*; every stress object computed here (a seeded scenario
//   perturbation, an interval/worst-case bound, a sensitivity number, a
//   robustness range) is a PURE OFFLINE model-risk input over an
//   already-recorded, replayable base series. Data flows ONE WAY, OUT of replay
//   and into a model-risk report; a stress/robustness value may NEVER be fed
//   back as a matching input, a hot-path branch, a price-process mechanic, an
//   adaptive-strategy state, or production state. The scenario application is
//   REPLAY-NEUTRAL: it never mutates the base series, the replay events, or any
//   matching state, and the replayable artifact's hash is invariant to whether
//   or which stress scenarios were run. This header is the deterministic proof
//   of that one-way, replay-neutral, kernel-free contract.
//
// WHAT IT PROVES (the boundary claim the MOC defends):
//   `moc-stress-testing-robustness-scenario-criticism` keeps robustness/
//   resilience analogies, sensitivity sweeps, interval/robust-control bounds,
//   and macro/coupled-system stress assumptions strictly in the OFFLINE
//   model-risk / scenario-criticism layer. Its open
//   `needs-generated-artifact-boundary` channel asked for a worked seeded
//   replay-neutral stress fixture with a diagnostics-on/off replay-hash. This
//   fixture supplies exactly that: given a replayable, deterministic ORDERED
//   base series (integer-tick, event-time) and a FIXED, declared stress-model
//   spec, it
//     * evaluates a FIXED, declared stress model M(base, theta) = a linear
//       functional of the base series plus declared per-axis linear weights and
//       declared cross-axis INTERACTION weights over a stress-parameter vector
//       theta -- so regimes/interactions can dominate, exactly as the MOC's
//       "sensitivity is not robustness" claim requires;
//     * scores an INTERVAL / WORST-CASE robustness bound over a FIXED scenario
//       set (min, max, range, and worst-case absolute deviation from the
//       baseline) -- the offline interval/robust-control stress tool;
//     * runs a one-parameter (single-axis) SENSITIVITY sweep at +/- a fixed
//       step and exposes that a flat one-axis sweep can co-exist with a large
//       multi-scenario robustness range -- the faithful encoding of "a
//       one-parameter sensitivity sweep is not robustness";
//     * generates the scenario set from an EXPLICIT seed via a deterministic
//       LCG, so the seed is explicit and the scenarios are reproducible;
//     * exposes TWO deterministic, order-sensitive FNV-1a-64 hashes: a
//       `base_replay_hash` over the ORDERED base series ONLY (replay-neutral --
//       invariant to the stress scenarios) and a `stress_report_hash` over the
//       base series plus the FIXED spec, the scenario set, and the derived
//       robustness report (the diagnostic's own reproducibility hash).
//
// SEMANTIC CONTRACT (the replay-equivalence / determinism guarantee):
//   compute() is a PURE function of the (immutable) ordered base series, the
//   FIXED spec, the scenario set, and the FIXED sweep step: identical inputs
//   yield an identical report and identical hashes. The `base_replay_hash`
//   depends ONLY on the base series, so adding, removing, reordering, or
//   re-weighting stress scenarios CANNOT change it -- this is the
//   diagnostics-on/off replay-hash equality. Reordering the scenario set leaves
//   the interval/worst-case/range robustness metrics unchanged (set semantics)
//   yet changes the `stress_report_hash` (order-sensitive integrity). The
//   computation takes every input by const reference, returns a plain value, and
//   exposes no API that could mutate or be consumed by matching state.
//
// Stress / robustness background (kept self-contained; no equation imported from
// any rendered source image -- these are textbook model-risk relations re-stated
// here for the fixture):
//   * An interval / worst-case stress bound over a scenario set is the [min,max]
//     of the model output across the declared scenarios; the robustness range is
//     its width and the worst-case deviation is the largest absolute departure
//     from the baseline output.
//   * A one-parameter sensitivity sweep perturbs ONE axis at a time; it is NOT
//     robustness when cross-axis interactions, regimes, or sign changes dominate
//     -- a sweep can read flat (zero single-axis sensitivity) while a joint
//     scenario over two interacting axes moves the output materially.
//   * The stress parameters theta are FIXED, declared OFFLINE inputs and the
//     scenario seed is explicit; nothing here is fitted in a feedback loop from
//     the very series being stressed, and nothing is fed back into matching.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using ReplayHash = std::uint64_t;
using Tick = std::int64_t;  // integer-tick base value / stress shock / model output

// One replayable base-series observation. Pure integer data: no pointer, no
// engine handle, no vtable. `seq` is the deterministic event ordinal/time and
// `value` is the recorded integer-tick quantity (e.g. a price change / P&L
// increment) over this event interval (may be negative).
struct ScenarioPoint {
    std::uint64_t seq = 0;
    Tick value = 0;
};

// FIXED, declared OFFLINE stress-model spec. `linear_weight[a]` is the declared
// first-order sensitivity of the model output to stress axis a; `interaction`
// is a row-major k*k matrix whose upper triangle (a<b) holds the declared
// cross-axis interaction weight. Chosen OFFLINE; never derived from the series
// being stressed in a feedback loop. Pure value type.
struct StressModelSpec {
    Tick base_offset = 0;              // declared constant added to the base aggregate
    std::vector<Tick> linear_weight;   // length k = number of stress axes
    std::vector<Tick> interaction;     // k*k row-major; interaction[a*k+b], a<b
};

// One declared stress scenario: a stress-parameter vector theta (one shock per
// axis). EXPLICIT -- either hand-declared or produced by the seeded generator.
// Pure value type; applying it NEVER mutates the base series.
struct StressScenario {
    std::vector<Tick> theta;
};

// The offline model-risk / robustness report plus the two order-sensitive replay
// hashes. Pure value type: copyable, holds no reference to any external
// (matching) state.
struct StressReport {
    std::size_t n = 0;            // base-series length
    std::size_t num_axes = 0;     // number of stress axes (= linear_weight.size())
    std::size_t num_scenarios = 0;

    Tick base_aggregate = 0;      // sum of base-series values (order-invariant)
    Tick baseline_output = 0;     // M(base, theta = 0) = base_aggregate + base_offset

    // Interval / worst-case robustness bound over the FIXED scenario set.
    // Order-INVARIANT (set semantics over the scenario outputs).
    bool has_scenarios = false;
    Tick min_output = 0;
    Tick max_output = 0;
    Tick robustness_range = 0;          // max_output - min_output
    Tick worst_case_abs_deviation = 0;  // max_s |M_s - baseline_output|

    // One-parameter (single-axis) sensitivity sweep at +/- sweep_step.
    Tick one_axis_max_abs_sensitivity = 0;

    // The MOC's gate: a one-parameter sweep is NOT robustness. False whenever the
    // single-axis sensitivity disagrees with the multi-scenario robustness range.
    bool sensitivity_equals_robustness = false;

    ReplayHash base_replay_hash = 0;    // over the base series ONLY (replay-neutral)
    ReplayHash stress_report_hash = 0;  // over base + spec + scenarios + report
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

inline constexpr Tick det_abs(Tick v) noexcept { return v < 0 ? -v : v; }

inline constexpr Tick det_min(Tick a, Tick b) noexcept { return a < b ? a : b; }
inline constexpr Tick det_max(Tick a, Tick b) noexcept { return a > b ? a : b; }

}  // namespace detail

// Analytic reference relations, used to validate the report on closed-form
// cases. Textbook model-risk relations, computed here so the fixture is
// self-checking and self-contained.
namespace analytic {

// Evaluate the FIXED, declared stress model for a stress-parameter vector theta:
//   M(base, theta) = base_aggregate + base_offset
//                    + sum_a linear_weight[a] * theta[a]
//                    + sum_{a<b} interaction[a*k+b] * theta[a] * theta[b].
// PURE: reads the (immutable) base aggregate and spec; the base series itself is
// never mutated by applying a scenario. Out-of-range axes are treated as 0.
inline Tick evaluate_model(Tick base_aggregate, const StressModelSpec& spec,
                           const std::vector<Tick>& theta) noexcept {
    Tick out = base_aggregate + spec.base_offset;
    const std::size_t k = spec.linear_weight.size();
    for (std::size_t a = 0; a < k; ++a) {
        const Tick ta = (a < theta.size()) ? theta[a] : 0;
        out += spec.linear_weight[a] * ta;
    }
    for (std::size_t a = 0; a < k; ++a) {
        const Tick ta = (a < theta.size()) ? theta[a] : 0;
        if (ta == 0) continue;
        for (std::size_t b = a + 1; b < k; ++b) {
            const Tick tb = (b < theta.size()) ? theta[b] : 0;
            const std::size_t idx = a * k + b;
            const Tick w = (idx < spec.interaction.size()) ? spec.interaction[idx] : 0;
            out += w * ta * tb;
        }
    }
    return out;
}

}  // namespace analytic

// Order-invariant aggregate of the replayable base series (sum of values). The
// model baseline is built from this; it is a pure read of the series.
inline Tick base_series_aggregate(const std::vector<ScenarioPoint>& series) noexcept {
    Tick s = 0;
    for (const ScenarioPoint& p : series) s += p.value;
    return s;
}

// Deterministic, order-sensitive FNV-1a-64 hash over the ORDERED base series
// ONLY. This is the REPLAY-NEUTRAL artifact: it depends on nothing about the
// stress scenarios, the spec, or the sweep, so running (or not running) any
// stress diagnostic leaves it unchanged -- the diagnostics-on/off replay-hash
// equality. Reordering the base series changes it (order-sensitive integrity).
inline ReplayHash base_series_hash(const std::vector<ScenarioPoint>& series) noexcept {
    ReplayHash h = detail::kFnvOffsetBasis;
    std::uint64_t ordinal = 0;
    for (const ScenarioPoint& p : series) {
        h = detail::fnv1a_u64(h, ordinal);
        h = detail::fnv1a_u64(h, p.seq);
        h = detail::fnv1a_i64(h, p.value);
        ++ordinal;
    }
    h = detail::fnv1a_u64(h, ordinal);  // fold the count
    return h;
}

// Deterministic, order-sensitive FNV-1a-64 hash over the base series, the FIXED
// spec, the ORDERED scenario set, and the derived report. Folds each scenario's
// ORDINAL so reordering the scenarios changes the hash; folds the spec and sweep
// step so a parameter change is detectable; folds the report fields and all
// counts so truncation/extension cannot collide with a prefix.
inline ReplayHash stress_report_hash(const std::vector<ScenarioPoint>& series,
                                     const StressModelSpec& spec,
                                     const std::vector<StressScenario>& scenarios,
                                     Tick sweep_step,
                                     const StressReport& report) noexcept {
    ReplayHash h = base_series_hash(series);

    h = detail::fnv1a_i64(h, spec.base_offset);
    std::uint64_t axis_ordinal = 0;
    for (Tick w : spec.linear_weight) {
        h = detail::fnv1a_u64(h, axis_ordinal);
        h = detail::fnv1a_i64(h, w);
        ++axis_ordinal;
    }
    h = detail::fnv1a_u64(h, axis_ordinal);  // fold the axis count
    std::uint64_t inter_ordinal = 0;
    for (Tick w : spec.interaction) {
        h = detail::fnv1a_u64(h, inter_ordinal);
        h = detail::fnv1a_i64(h, w);
        ++inter_ordinal;
    }
    h = detail::fnv1a_u64(h, inter_ordinal);  // fold the interaction-matrix size

    std::uint64_t scen_ordinal = 0;
    for (const StressScenario& sc : scenarios) {
        h = detail::fnv1a_u64(h, scen_ordinal);
        std::uint64_t t_ordinal = 0;
        for (Tick t : sc.theta) {
            h = detail::fnv1a_u64(h, t_ordinal);
            h = detail::fnv1a_i64(h, t);
            ++t_ordinal;
        }
        h = detail::fnv1a_u64(h, t_ordinal);  // fold the per-scenario axis count
        ++scen_ordinal;
    }
    h = detail::fnv1a_u64(h, scen_ordinal);  // fold the scenario count

    h = detail::fnv1a_i64(h, sweep_step);
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(report.n));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(report.num_axes));
    h = detail::fnv1a_u64(h, static_cast<std::uint64_t>(report.num_scenarios));
    h = detail::fnv1a_i64(h, report.base_aggregate);
    h = detail::fnv1a_i64(h, report.baseline_output);
    h = detail::fnv1a_byte(h, report.has_scenarios ? 1u : 0u);
    h = detail::fnv1a_i64(h, report.min_output);
    h = detail::fnv1a_i64(h, report.max_output);
    h = detail::fnv1a_i64(h, report.robustness_range);
    h = detail::fnv1a_i64(h, report.worst_case_abs_deviation);
    h = detail::fnv1a_i64(h, report.one_axis_max_abs_sensitivity);
    h = detail::fnv1a_byte(h, report.sensitivity_equals_robustness ? 1u : 0u);
    return h;
}

// Generate a FIXED scenario set from an EXPLICIT seed via a deterministic 64-bit
// LCG. Each axis shock is mapped to {-magnitude, 0, +magnitude}. Reproducible:
// the same seed/count/axes/magnitude yields the identical scenario set; a
// different seed yields a different set. The seed is an explicit parameter and
// the produced scenarios are explicit, declared model-risk inputs -- never a
// feedback loop on the series being stressed.
inline std::vector<StressScenario> generate_seeded_scenarios(std::uint64_t seed,
                                                             std::size_t num_scenarios,
                                                             std::size_t num_axes,
                                                             Tick magnitude) {
    std::vector<StressScenario> out;
    out.reserve(num_scenarios);
    std::uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    for (std::size_t s = 0; s < num_scenarios; ++s) {
        StressScenario sc;
        sc.theta.reserve(num_axes);
        for (std::size_t a = 0; a < num_axes; ++a) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            const std::uint64_t r = (state >> 33) % 3ULL;  // 0,1,2
            const Tick shock = (r == 0) ? -magnitude : (r == 1) ? 0 : magnitude;
            sc.theta.push_back(shock);
        }
        out.push_back(std::move(sc));
    }
    return out;
}

// Compute the offline stress / robustness report for a replayable base series
// under a FIXED spec and a FIXED scenario set, with a FIXED single-axis sweep
// step. PURE, READ-ONLY, OFFLINE: every input is taken by const reference and
// never modified; the result is a plain value. Applying a scenario evaluates the
// model -- it NEVER mutates the base series and is NEVER fed back into matching.
inline StressReport compute_stress_report(const std::vector<ScenarioPoint>& series,
                                          const StressModelSpec& spec,
                                          const std::vector<StressScenario>& scenarios,
                                          Tick sweep_step) {
    StressReport report;
    report.n = series.size();
    report.num_axes = spec.linear_weight.size();
    report.num_scenarios = scenarios.size();
    report.base_aggregate = base_series_aggregate(series);

    const std::vector<Tick> zero_theta(report.num_axes, 0);
    report.baseline_output =
        analytic::evaluate_model(report.base_aggregate, spec, zero_theta);

    // Interval / worst-case robustness bound over the FIXED scenario set.
    if (!scenarios.empty()) {
        report.has_scenarios = true;
        bool first = true;
        for (const StressScenario& sc : scenarios) {
            const Tick out =
                analytic::evaluate_model(report.base_aggregate, spec, sc.theta);
            if (first) {
                report.min_output = out;
                report.max_output = out;
                first = false;
            } else {
                report.min_output = detail::det_min(report.min_output, out);
                report.max_output = detail::det_max(report.max_output, out);
            }
            const Tick dev = detail::det_abs(out - report.baseline_output);
            report.worst_case_abs_deviation =
                detail::det_max(report.worst_case_abs_deviation, dev);
        }
        report.robustness_range = report.max_output - report.min_output;
    }

    // One-parameter (single-axis) sensitivity sweep at +/- sweep_step. Each axis
    // is perturbed ALONE; the metric is the largest absolute single-axis move.
    Tick sens = 0;
    for (std::size_t a = 0; a < report.num_axes; ++a) {
        std::vector<Tick> theta_plus(report.num_axes, 0);
        std::vector<Tick> theta_minus(report.num_axes, 0);
        theta_plus[a] = sweep_step;
        theta_minus[a] = -sweep_step;
        const Tick dp = detail::det_abs(
            analytic::evaluate_model(report.base_aggregate, spec, theta_plus) -
            report.baseline_output);
        const Tick dm = detail::det_abs(
            analytic::evaluate_model(report.base_aggregate, spec, theta_minus) -
            report.baseline_output);
        sens = detail::det_max(sens, detail::det_max(dp, dm));
    }
    report.one_axis_max_abs_sensitivity = sens;
    report.sensitivity_equals_robustness =
        report.has_scenarios && (sens == report.robustness_range);

    report.base_replay_hash = base_series_hash(series);
    report.stress_report_hash =
        stress_report_hash(series, spec, scenarios, sweep_step, report);
    return report;
}

// Read-only convenience wrapper. Holds a const view of an externally owned base
// series, FIXED spec, and FIXED scenario set, and exposes ONLY the computation;
// it has no mutating member and no matching-engine coupling, making the offline /
// one-way-data-flow / replay-neutral boundary explicit at the call site. The
// report it returns can be read but never written back into matching.
class StressScenarioCriticismReplay {
public:
    StressScenarioCriticismReplay(const std::vector<ScenarioPoint>& series,
                                  const StressModelSpec& spec,
                                  const std::vector<StressScenario>& scenarios,
                                  Tick sweep_step) noexcept
        : series_(&series),
          spec_(&spec),
          scenarios_(&scenarios),
          sweep_step_(sweep_step) {}

    StressReport compute() const {
        return compute_stress_report(*series_, *spec_, *scenarios_, sweep_step_);
    }
    ReplayHash hash() const { return compute().stress_report_hash; }
    // The replay-neutral base hash, available WITHOUT running the stress
    // diagnostic -- the diagnostics-off side of the equality.
    ReplayHash base_hash() const { return base_series_hash(*series_); }
    std::size_t size() const noexcept { return series_->size(); }
    bool empty() const noexcept { return series_->empty(); }

private:
    const std::vector<ScenarioPoint>* series_;
    const StressModelSpec* spec_;
    const std::vector<StressScenario>* scenarios_;
    Tick sweep_step_;
};

// --- Compile-time non-feedback / no-matching-coupling detector ----------------
//
// Detects whether a type T exposes a conventional matching-feedback member
// (a method named `apply_to_matching`). The offline stress types do NOT, which
// is the type-level proof that a stress/robustness value cannot be wired back
// into the kernel through these structs. A test supplies a POSITIVE CONTROL (a
// type that DOES expose such a member) to prove the trait actually discriminates
// and is not vacuously false.
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
//   * no `apply_to_matching` feedback member => the report is one-way OUT;
//   * compute() is invocable on a CONST wrapper => read-only diagnostic.
static_assert(std::is_trivially_copyable_v<ScenarioPoint>,
              "ScenarioPoint must be pure data (no engine handle)");
static_assert(std::is_standard_layout_v<ScenarioPoint>,
              "ScenarioPoint must be standard-layout (no vtable / engine coupling)");
static_assert(!has_matching_feedback_v<ScenarioPoint>,
              "ScenarioPoint must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<StressReport>,
              "StressReport must not expose a matching-feedback member (one-way data flow)");
static_assert(!has_matching_feedback_v<StressScenarioCriticismReplay>,
              "StressScenarioCriticismReplay must not expose a matching-feedback member (one-way data flow)");
static_assert(
    std::is_invocable_r_v<StressReport,
                          decltype(&StressScenarioCriticismReplay::compute),
                          const StressScenarioCriticismReplay&>,
    "compute() must be a read-only (const) diagnostic");

}  // namespace offline
