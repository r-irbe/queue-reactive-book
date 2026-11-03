// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "export_schema.hpp"
#include "replay_driver.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace sim {

struct ReplayDiagnostics {
    std::uint64_t input_events = 0;
    std::uint64_t accepted_events = 0;
    std::uint64_t rejected_events = 0;
    std::uint64_t limit_submits = 0;
    std::uint64_t market_submits = 0;
    std::uint64_t cancel_events = 0;
    std::uint64_t cancel_rejects = 0;
    std::uint64_t snapshot_events = 0;
    std::uint64_t trades = 0;
    Qty filled_qty = 0;

    std::uint64_t mid_samples = 0;
    std::int64_t spread_sum = 0;
    Tick min_spread = std::numeric_limits<Tick>::max();
    Tick max_spread = std::numeric_limits<Tick>::min();
    std::uint64_t empty_book_samples = 0;
    std::uint64_t one_sided_samples = 0;

    std::uint64_t inter_arrival_samples = 0;
    std::int64_t inter_arrival_sum = 0;
    TimeNs min_inter_arrival = std::numeric_limits<TimeNs>::max();
    TimeNs max_inter_arrival = std::numeric_limits<TimeNs>::min();
    bool has_last_ts = false;
    TimeNs last_ts = 0;

    std::uint64_t replay_hash = 1469598103934665603ULL;

    // Causal-ordering self-checks: a deterministic replay must feed events in
    // non-decreasing event-time and strictly increasing seq. These count steps
    // that violate that contract (e.g. a reordered/duplicated event), which
    // would otherwise silently corrupt inter-arrival deltas and any downstream
    // event-time feature/label. Aggregation semantics are left unchanged; the
    // counters only surface the violation.
    std::uint64_t seq_order_violations = 0;
    std::uint64_t time_order_violations = 0;
    bool has_last_seq = false;
    EventSeq last_seq = 0;
};

inline constexpr std::size_t kReplayDiagnosticsExportCount = 25;
inline constexpr std::array<ExportColumn, kReplayDiagnosticsExportCount>
    kReplayDiagnosticsExportSchema = {{{"input_events"},
                                       {"accepted_events"},
                                       {"rejected_events"},
                                       {"limit_submits"},
                                       {"market_submits"},
                                       {"cancel_events"},
                                       {"cancel_rejects"},
                                       {"snapshot_events"},
                                       {"trades"},
                                       {"filled_qty"},
                                       {"mid_samples"},
                                       {"spread_sum"},
                                       {"min_spread"},
                                       {"max_spread"},
                                       {"average_spread"},
                                       {"empty_book_samples"},
                                       {"one_sided_samples"},
                                       {"inter_arrival_samples"},
                                       {"inter_arrival_sum"},
                                       {"min_inter_arrival"},
                                       {"max_inter_arrival"},
                                       {"average_inter_arrival_ns"},
                                       {"replay_hash"},
                                       {"seq_order_violations"},
                                       {"time_order_violations"}}};

inline void hash_mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash ^= value;
    hash *= 1099511628211ULL;
}

inline std::uint64_t hash_i64(std::int64_t value) noexcept {
    return static_cast<std::uint64_t>(value);
}

inline double average_spread(const ReplayDiagnostics& diagnostics) noexcept {
    if (diagnostics.mid_samples == 0) return 0.0;
    return static_cast<double>(diagnostics.spread_sum) /
           static_cast<double>(diagnostics.mid_samples);
}

inline double average_inter_arrival_ns(const ReplayDiagnostics& diagnostics) noexcept {
    if (diagnostics.inter_arrival_samples == 0) return 0.0;
    return static_cast<double>(diagnostics.inter_arrival_sum) /
           static_cast<double>(diagnostics.inter_arrival_samples);
}

inline std::array<double, kReplayDiagnosticsExportCount>
export_replay_diagnostics(const ReplayDiagnostics& diagnostics) noexcept {
    return {static_cast<double>(diagnostics.input_events),
            static_cast<double>(diagnostics.accepted_events),
            static_cast<double>(diagnostics.rejected_events),
            static_cast<double>(diagnostics.limit_submits),
            static_cast<double>(diagnostics.market_submits),
            static_cast<double>(diagnostics.cancel_events),
            static_cast<double>(diagnostics.cancel_rejects),
            static_cast<double>(diagnostics.snapshot_events),
            static_cast<double>(diagnostics.trades),
            static_cast<double>(diagnostics.filled_qty),
            static_cast<double>(diagnostics.mid_samples),
            static_cast<double>(diagnostics.spread_sum),
            diagnostics.mid_samples > 0 ? static_cast<double>(diagnostics.min_spread)
                                        : 0.0,
            diagnostics.mid_samples > 0 ? static_cast<double>(diagnostics.max_spread)
                                        : 0.0,
            average_spread(diagnostics),
            static_cast<double>(diagnostics.empty_book_samples),
            static_cast<double>(diagnostics.one_sided_samples),
            static_cast<double>(diagnostics.inter_arrival_samples),
            static_cast<double>(diagnostics.inter_arrival_sum),
            diagnostics.inter_arrival_samples > 0
                ? static_cast<double>(diagnostics.min_inter_arrival)
                : 0.0,
            diagnostics.inter_arrival_samples > 0
                ? static_cast<double>(diagnostics.max_inter_arrival)
                : 0.0,
            average_inter_arrival_ns(diagnostics),
            static_cast<double>(diagnostics.replay_hash),
            static_cast<double>(diagnostics.seq_order_violations),
            static_cast<double>(diagnostics.time_order_violations)};
}

inline void add_observable_sample(ReplayDiagnostics& diagnostics,
                                  const BookObservables& obs) noexcept {
    if (obs.has_mid) {
        ++diagnostics.mid_samples;
        diagnostics.spread_sum += obs.spread;
        if (obs.spread < diagnostics.min_spread) diagnostics.min_spread = obs.spread;
        if (obs.spread > diagnostics.max_spread) diagnostics.max_spread = obs.spread;
    }
    if (!obs.has_bid && !obs.has_ask) {
        ++diagnostics.empty_book_samples;
    } else if (!obs.has_bid || !obs.has_ask) {
        ++diagnostics.one_sided_samples;
    }
}

inline void add_order_violation_checks(ReplayDiagnostics& diagnostics,
                                       EventSeq seq, TimeNs ts) noexcept {
    if (diagnostics.has_last_seq && seq <= diagnostics.last_seq) {
        ++diagnostics.seq_order_violations;
    }
    diagnostics.has_last_seq = true;
    diagnostics.last_seq = seq;
    // Read last_ts/has_last_ts BEFORE add_inter_arrival_sample updates them.
    if (diagnostics.has_last_ts && ts < diagnostics.last_ts) {
        ++diagnostics.time_order_violations;
    }
}

inline void add_inter_arrival_sample(ReplayDiagnostics& diagnostics, TimeNs ts) noexcept {
    if (diagnostics.has_last_ts) {
        const TimeNs delta = static_cast<TimeNs>(ts - diagnostics.last_ts);
        ++diagnostics.inter_arrival_samples;
        diagnostics.inter_arrival_sum += delta;
        if (delta < diagnostics.min_inter_arrival) diagnostics.min_inter_arrival = delta;
        if (delta > diagnostics.max_inter_arrival) diagnostics.max_inter_arrival = delta;
    }
    diagnostics.has_last_ts = true;
    diagnostics.last_ts = ts;
}

inline void add_replay_hash_sample(ReplayDiagnostics& diagnostics,
                                   const ReplayStep& step) noexcept {
    hash_mix(diagnostics.replay_hash, static_cast<std::uint64_t>(step.input.kind));
    hash_mix(diagnostics.replay_hash, step.input.seq);
    hash_mix(diagnostics.replay_hash, hash_i64(step.input.ts));
    hash_mix(diagnostics.replay_hash, step.input.order_id);
    hash_mix(diagnostics.replay_hash, static_cast<std::uint64_t>(step.input.side));
    hash_mix(diagnostics.replay_hash, hash_i64(step.input.price));
    hash_mix(diagnostics.replay_hash, hash_i64(step.input.qty));
    hash_mix(diagnostics.replay_hash, step.accepted ? 1ULL : 0ULL);
    hash_mix(diagnostics.replay_hash, hash_i64(step.execution.filled));
    hash_mix(diagnostics.replay_hash, hash_i64(step.execution.remaining));
    hash_mix(diagnostics.replay_hash, static_cast<std::uint64_t>(step.fills.size()));
    hash_mix(diagnostics.replay_hash, step.after.has_bid ? 1ULL : 0ULL);
    hash_mix(diagnostics.replay_hash, step.after.has_ask ? 1ULL : 0ULL);
    hash_mix(diagnostics.replay_hash, hash_i64(step.after.best_bid));
    hash_mix(diagnostics.replay_hash, hash_i64(step.after.best_ask));
    hash_mix(diagnostics.replay_hash, hash_i64(step.after.total_bid_qty));
    hash_mix(diagnostics.replay_hash, hash_i64(step.after.total_ask_qty));
}

inline void add_replay_step(ReplayDiagnostics& diagnostics,
                            const ReplayStep& step) noexcept {
    ++diagnostics.input_events;
    if (step.accepted) {
        ++diagnostics.accepted_events;
    } else {
        ++diagnostics.rejected_events;
    }

    switch (step.input.kind) {
    case TraceEventKind::SubmitLimit:
        ++diagnostics.limit_submits;
        break;
    case TraceEventKind::SubmitMarket:
        ++diagnostics.market_submits;
        break;
    case TraceEventKind::Cancel:
        ++diagnostics.cancel_events;
        if (!step.cancel_accepted) ++diagnostics.cancel_rejects;
        break;
    case TraceEventKind::Snapshot:
        ++diagnostics.snapshot_events;
        break;
    case TraceEventKind::Trade:
    case TraceEventKind::Reject:
        break;
    }

    diagnostics.trades += step.trade_events.size();
    diagnostics.filled_qty =
        static_cast<Qty>(diagnostics.filled_qty + step.execution.filled);

    add_observable_sample(diagnostics, step.after);
    add_order_violation_checks(diagnostics, step.input.seq, step.input.ts);
    add_inter_arrival_sample(diagnostics, step.input.ts);
    add_replay_hash_sample(diagnostics, step);
}

class ReplayDiagnosticsCollector {
public:
    void add(const ReplayStep& step) noexcept { add_replay_step(diagnostics_, step); }
    const ReplayDiagnostics& diagnostics() const noexcept { return diagnostics_; }

private:
    ReplayDiagnostics diagnostics_{};
};

} // namespace sim
