#pragma once

#include "export_schema.hpp"
#include "inventory_tca.hpp"
#include "markout_observer.hpp"
#include "parent_order_tracker.hpp"
#include "replay_diagnostics.hpp"
#include "risk_guard.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace sim {

struct ReplayPipelineConfig {
    RiskLimits risk_limits{};
    Tick min_tick = 0;
    Tick max_tick = 0;
    std::size_t expected_orders = 4096;
    bool enable_compaction = true;
    CompactionConfig compaction_cfg{};
    std::size_t arena_bytes = 0;
    FeeSchedule fees{};
    std::vector<MarkoutHorizon> markout_horizons{};
};

struct ReplayPipelineStep {
    RiskDecision risk{};
    bool replayed = false;
    ReplayStep replay{};
    std::vector<MarkoutRecord> markouts{};
};

inline std::size_t risk_reject_reason_index(RiskRejectReason reason) noexcept {
    return static_cast<std::size_t>(reason);
}

struct ReplayPipelineDiagnostics {
    std::uint64_t intents = 0;
    std::uint64_t accepted_intents = 0;
    std::uint64_t rejected_intents = 0;
    std::uint64_t replayed_events = 0;
    std::array<std::uint64_t, kRiskRejectReasonCount> reject_reason_counts{};
};

inline constexpr std::size_t kReplayPipelineDiagnosticsCount =
    4U + kRiskRejectReasonCount;

inline std::array<ExportColumn, kReplayPipelineDiagnosticsCount>
replay_pipeline_diagnostics_schema() noexcept {
    std::array<ExportColumn, kReplayPipelineDiagnosticsCount> out{};
    std::size_t pos = 0;
    out[pos++] = ExportColumn{.name = "intents"};
    out[pos++] = ExportColumn{.name = "accepted_intents"};
    out[pos++] = ExportColumn{.name = "rejected_intents"};
    out[pos++] = ExportColumn{.name = "replayed_events"};
    for (std::size_t reason = 0; reason < kRiskRejectReasonCount; ++reason) {
        out[pos++] = ExportColumn{.name = "reject_reason_count", .index = reason};
    }
    return out;
}

inline std::array<double, kReplayPipelineDiagnosticsCount>
export_replay_pipeline_diagnostics(
    const ReplayPipelineDiagnostics& diagnostics) noexcept {
    std::array<double, kReplayPipelineDiagnosticsCount> out{};
    std::size_t pos = 0;
    out[pos++] = static_cast<double>(diagnostics.intents);
    out[pos++] = static_cast<double>(diagnostics.accepted_intents);
    out[pos++] = static_cast<double>(diagnostics.rejected_intents);
    out[pos++] = static_cast<double>(diagnostics.replayed_events);
    for (std::uint64_t count : diagnostics.reject_reason_counts) {
        out[pos++] = static_cast<double>(count);
    }
    return out;
}

class ReplayPipeline {
public:
    explicit ReplayPipeline(ReplayPipelineConfig config)
        : fees_(config.fees),
          risk_(config.risk_limits),
          replay_(config.min_tick, config.max_tick, config.expected_orders,
                  config.enable_compaction, config.compaction_cfg,
                  config.arena_bytes),
          markouts_(std::move(config.markout_horizons)) {}

    ReplayPipelineStep apply(const StrategyIntent& intent,
                             const RiskState& state = {}) {
        ReplayPipelineStep out{};
        ++pipeline_diagnostics_.intents;
        out.risk = risk_.evaluate(intent, state);
        if (!out.risk.accepted) {
            ++pipeline_diagnostics_.rejected_intents;
            ++pipeline_diagnostics_
                  .reject_reason_counts[risk_reject_reason_index(
                      out.risk.reject_reason)];
            return out;
        }
        ++pipeline_diagnostics_.accepted_intents;

        parent_orders_.record_submission(intent);
        out.replay = replay_.apply(out.risk.event);
        out.replayed = true;
        ++pipeline_diagnostics_.replayed_events;
        diagnostics_.add(out.replay);

        for (const FillRecord& fill : out.replay.fills) {
            parent_orders_.record_maker_fill(fill);
            parent_orders_.record_taker_fill(fill);
            tca_.add_maker_fill_with_fees(fill, fees_);
            tca_.add_taker_fill_with_fees(fill, fees_);
            inventory_.add_maker_fill(fill, fees_);
            inventory_.add_taker_fill(fill, fees_);
            markouts_.record_fill(fill);
        }

        out.markouts = markouts_.evaluate(out.replay.after);
        for (const MarkoutRecord& markout : out.markouts) {
            tca_.add_maker_markout(markout);
            tca_.add_taker_markout(markout);
        }
        return out;
    }

    const FeeSchedule& fees() const noexcept { return fees_; }
    const RiskGuard& risk() const noexcept { return risk_; }
    const ReplayDriver& replay() const noexcept { return replay_; }
    ReplayDriver& replay() noexcept { return replay_; }
    const ReplayDiagnostics& diagnostics() const noexcept {
        return diagnostics_.diagnostics();
    }
    const ReplayPipelineDiagnostics& pipeline_diagnostics() const noexcept {
        return pipeline_diagnostics_;
    }
    const ParentOrderTracker& parent_orders() const noexcept {
        return parent_orders_;
    }
    const TcaAccumulator& tca() const noexcept { return tca_; }
    const InventoryTracker& inventory() const noexcept { return inventory_; }
    const MarkoutObserver& markout_observer() const noexcept { return markouts_; }

private:
    FeeSchedule fees_{};
    RiskGuard risk_;
    ReplayDriver replay_;
    ReplayPipelineDiagnostics pipeline_diagnostics_{};
    ReplayDiagnosticsCollector diagnostics_{};
    ParentOrderTracker parent_orders_{};
    TcaAccumulator tca_{};
    InventoryTracker inventory_{};
    MarkoutObserver markouts_;
};

} // namespace sim
