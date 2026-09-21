#pragma once

#include "event_trace.hpp"   // EventSeq (kernel-free: pulls only types.hpp)
#include "export_schema.hpp" // ExportSchemaMeta, LabelAvailability, units
#include "types.hpp"         // Side, TimeNs

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sim {

// --- Offline market-impact DIAGNOSTIC scorer (Tier-3 item 12, diagnostics-only)
//
// Roadmap item 12 asks for "calibrated impact models". This scaffold deliberately
// does NOT fit/calibrate anything and makes NO model claim: it is an offline
// SCORER that, given a caller-supplied coefficient, measures how well a
// square-root impact magnitude law tracks realized post-trade mid moves. Any
// calibration/claim of correctness stays DEFERRED until real data exists. It is
// header-only, cold-path, and never feeds back into the frozen matching kernel.
//
// Sign convention (FIXED, direction-normalized -- never mixed with raw signed):
//   side_sign = +1 for Buy, -1 for Sell.
//   realized_impact_ticks = side_sign * (post_mid_x2 - pre_mid_x2) / 2.0
//     -> POSITIVE when the mid moves WITH the aggressing trade (buy pushes up,
//        sell pushes down); NEGATIVE on adverse/mean-reverting moves.
//   predicted_impact_ticks = coef * sqrt(|participation|)  -- magnitude only,
//     always >= 0 (a concave square-root law; Almgren/Kyle style). The model
//     predicts the MAGNITUDE of the with-trade move, hence direction-normalized
//     realized values are the correct target.
//   residual = realized - predicted.
//
// Mids are doubled integer ticks (mid_x2) so half-ticks are exact; differencing
// is done in int64 then divided by 2.0 (no overflow for Tick-derived values).
// Observations whose forward mid is not yet realized (Pending) or that lack a
// pre/post mid are EXCLUDED from scoring -- never imputed (no look-ahead, no
// fabricated data).

struct ImpactObservation {
    EventSeq seq = 0;
    TimeNs ts = 0;
    Side side = Side::Buy;
    // Dimensionless participation (caller-supplied, e.g. trade qty / reference
    // depth or ADV fraction). |participation| is used; sign is ignored.
    double participation = 0.0;
    std::int64_t pre_mid_x2 = 0;   // doubled-tick mid at trade time
    std::int64_t post_mid_x2 = 0;  // doubled-tick mid at the forward horizon
    bool has_pre_mid = false;
    bool has_post_mid = false;
    // Availability of the forward (post-horizon) mid; only `Available`
    // observations with both mids are scored.
    LabelAvailability availability = LabelAvailability::NotApplicable;
};

inline double impact_side_sign(Side s) noexcept {
    return s == Side::Buy ? 1.0 : -1.0;
}

// Direction-normalized realized impact in ticks (may be negative on adverse moves).
inline double realized_impact_ticks(const ImpactObservation& o) noexcept {
    const std::int64_t diff = o.post_mid_x2 - o.pre_mid_x2;  // exact int64
    return impact_side_sign(o.side) * (static_cast<double>(diff) / 2.0);
}

// Square-root magnitude law. NO fitting: `coef` is supplied by the caller.
inline double predicted_impact_ticks(double coef, double participation) noexcept {
    return coef * std::sqrt(std::fabs(participation));
}

inline bool impact_observation_scorable(const ImpactObservation& o) noexcept {
    return o.has_pre_mid && o.has_post_mid &&
           o.availability == LabelAvailability::Available;
}

// Online accumulator: RMSE/bias of (realized - predicted) over scorable
// observations. All accumulators are double; counts are uint64. Any non-finite
// residual flips `finite=false` and the observation is counted as excluded
// rather than polluting the accumulators.
struct ImpactScore {
    std::uint64_t included = 0;  // observations actually scored
    std::uint64_t excluded = 0;  // pending / missing-mid / non-finite skipped
    double sum_sq_residual = 0.0;
    double sum_residual = 0.0;
    bool finite = true;

    void add(const ImpactObservation& o, double coef) noexcept {
        if (!impact_observation_scorable(o)) {
            ++excluded;
            return;
        }
        const double residual =
            realized_impact_ticks(o) - predicted_impact_ticks(coef, o.participation);
        if (!std::isfinite(residual)) {
            finite = false;
            ++excluded;
            return;
        }
        ++included;
        sum_sq_residual += residual * residual;
        sum_residual += residual;
    }

    double rmse() const noexcept {
        return included == 0 ? 0.0
                             : std::sqrt(sum_sq_residual / static_cast<double>(included));
    }
    double bias() const noexcept {
        return included == 0 ? 0.0 : sum_residual / static_cast<double>(included);
    }
};

inline ImpactScore score_impacts(const ImpactObservation* obs, std::size_t n,
                                 double coef) noexcept {
    ImpactScore s{};
    for (std::size_t i = 0; i < n; ++i) s.add(obs[i], coef);
    return s;
}

// Schema descriptor for an exported impact-diagnostic frame. Residuals are
// fractional ticks (a derived double diagnostic), event-time indexed, carrying a
// realized forward label horizon. Diagnostics-only: no calibration claim.
inline ExportSchemaMeta impact_diagnostics_schema_meta() noexcept {
    ExportSchemaMeta m{};
    m.schema_name = "impact_diagnostics";
    m.timestamp_basis = TimestampBasis::EventTime;
    m.price_unit = QuantityUnit::Ticks;
    m.quantity_unit = QuantityUnit::Ratio;  // dimensionless residual/diagnostic
    m.label_availability = LabelAvailability::Available;
    m.missing_policy = MissingPolicy::Forbidden;
    return m;
}

} // namespace sim
