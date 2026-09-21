#pragma once

#include "fill_attribution.hpp"
#include "parent_order_tracker.hpp"
#include "export_schema.hpp"

#include <array>
#include <cstddef>

namespace sim {

inline constexpr std::size_t kFillLabelCount = 12;
inline constexpr std::size_t kMarkoutLabelCount = 20;
inline constexpr std::size_t kParentCompletionLabelCount = 24;
inline constexpr std::array<ExportColumn, kFillLabelCount> kFillLabelSchema = {
    {{"seq"},
     {"ts"},
     {"source"},
     {"maker_parent_id"},
     {"taker_parent_id"},
     {"maker"},
     {"taker"},
     {"maker_side"},
     {"taker_side"},
     {"price"},
     {"qty"},
     {"mid_x2_at_fill"}}};
inline constexpr std::array<ExportColumn, kMarkoutLabelCount> kMarkoutLabelSchema = {
    {{"fill_seq"},
     {"fill_ts"},
     {"eval_seq"},
     {"eval_ts"},
     {"horizon_events"},
     {"horizon_ns"},
     {"source"},
     {"maker_parent_id"},
     {"taker_parent_id"},
     {"maker_side"},
     {"taker_side"},
     {"price"},
     {"qty"},
     {"has_mid_at_fill"},
     {"mid_x2_at_fill"},
     {"has_mid_after"},
     {"mid_x2_after"},
     {"maker_markout_x2"},
     {"taker_markout_x2"},
     {"taker_favorable"}}};
inline constexpr std::array<ExportColumn, kParentCompletionLabelCount>
    kParentCompletionLabelSchema = {{{"parent_id"},
                                     {"agent"},
                                     {"source"},
                                     {"side"},
                                     {"submitted_qty"},
                                     {"passive_submitted_qty"},
                                     {"aggressive_submitted_qty"},
                                     {"passive_filled_qty"},
                                     {"aggressive_filled_qty"},
                                     {"filled_qty"},
                                     {"remaining_qty"},
                                     {"completion_rate"},
                                     {"passive_fill_rate"},
                                     {"aggressive_fill_rate"},
                                     {"passive_queue_miss_qty"},
                                     {"passive_queue_miss_rate"},
                                     {"fill_notional"},
                                     {"average_fill_price"},
                                     {"has_benchmark_mid"},
                                     {"has_terminal_mid"},
                                     {"unfilled_qty"},
                                     {"opportunity_cost_x2_per_qty"},
                                     {"opportunity_cost_x2_qty"},
                                     {"average_opportunity_cost_ticks"}}};

inline double side_label(Side side) noexcept {
    return is_buy(side) ? 1.0 : -1.0;
}

inline double favorable_label(Tick signed_markout_x2) noexcept {
    if (signed_markout_x2 > 0) return 1.0;
    if (signed_markout_x2 < 0) return -1.0;
    return 0.0;
}

inline std::array<double, kFillLabelCount>
export_fill_labels(const FillRecord& fill) noexcept {
    return {static_cast<double>(fill.seq),
            static_cast<double>(fill.ts),
            static_cast<double>(fill.source),
            static_cast<double>(fill.maker_parent_id),
            static_cast<double>(fill.taker_parent_id),
            static_cast<double>(fill.maker),
            static_cast<double>(fill.taker),
            side_label(fill.maker_side),
            side_label(fill.taker_side),
            static_cast<double>(fill.price),
            static_cast<double>(fill.qty),
            fill.has_mid_at_fill ? static_cast<double>(fill.mid_x2_at_fill) : 0.0};
}

inline std::array<double, kMarkoutLabelCount>
export_markout_labels(const MarkoutRecord& markout) noexcept {
    const FillRecord& fill = markout.fill;
    const Tick maker_markout_x2 = markout.has_mid_after ? markout.maker_markout_x2 : 0;
    const Tick taker_markout_x2 = markout.has_mid_after ? markout.taker_markout_x2 : 0;
    return {static_cast<double>(fill.seq),
            static_cast<double>(fill.ts),
            static_cast<double>(markout.eval_seq),
            static_cast<double>(markout.eval_ts),
            static_cast<double>(markout.horizon_events),
            static_cast<double>(markout.horizon_ns),
            static_cast<double>(fill.source),
            static_cast<double>(fill.maker_parent_id),
            static_cast<double>(fill.taker_parent_id),
            side_label(fill.maker_side),
            side_label(fill.taker_side),
            static_cast<double>(fill.price),
            static_cast<double>(fill.qty),
            fill.has_mid_at_fill ? 1.0 : 0.0,
            fill.has_mid_at_fill ? static_cast<double>(fill.mid_x2_at_fill) : 0.0,
            markout.has_mid_after ? 1.0 : 0.0,
            markout.has_mid_after ? static_cast<double>(markout.mid_x2_after) : 0.0,
            static_cast<double>(maker_markout_x2),
            static_cast<double>(taker_markout_x2),
            favorable_label(taker_markout_x2)};
}

inline std::array<double, kParentCompletionLabelCount>
export_parent_completion_labels(const ParentOrderProgress& progress,
                                const OpportunityCostMetrics& opportunity) noexcept {
    return {static_cast<double>(progress.key.parent_id),
            static_cast<double>(progress.key.agent),
            static_cast<double>(progress.key.source),
            side_label(progress.key.side),
            static_cast<double>(progress.submitted_qty),
            static_cast<double>(progress.passive_submitted_qty),
            static_cast<double>(progress.aggressive_submitted_qty),
            static_cast<double>(progress.passive_filled_qty),
            static_cast<double>(progress.aggressive_filled_qty),
            static_cast<double>(filled_qty(progress)),
            static_cast<double>(remaining_qty(progress)),
            completion_rate(progress),
            passive_fill_rate(progress),
            aggressive_fill_rate(progress),
            static_cast<double>(passive_queue_miss_qty(progress)),
            passive_queue_miss_rate(progress),
            static_cast<double>(progress.fill_notional),
            average_parent_fill_price(progress),
            opportunity.has_benchmark_mid ? 1.0 : 0.0,
            opportunity.has_terminal_mid ? 1.0 : 0.0,
            static_cast<double>(opportunity.unfilled_qty),
            static_cast<double>(opportunity.opportunity_cost_x2_per_qty),
            static_cast<double>(opportunity.opportunity_cost_x2_qty),
            average_opportunity_cost_ticks(opportunity)};
}

} // namespace sim
