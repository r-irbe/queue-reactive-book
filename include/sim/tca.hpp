// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "export_schema.hpp"
#include "fill_attribution.hpp"

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <unordered_map>

namespace sim {

struct TcaSummary {
    Qty passive_qty = 0;
    Qty aggressive_qty = 0;
    std::int64_t passive_notional = 0;
    std::int64_t aggressive_notional = 0;
    std::int64_t passive_fee_cost = 0;
    std::int64_t aggressive_fee_cost = 0;
    std::int64_t passive_markout_x2_qty = 0;
    std::int64_t aggressive_markout_x2_qty = 0;
    Qty passive_markout_qty = 0;
    Qty aggressive_markout_qty = 0;
};

struct FeeSchedule {
    std::int64_t maker_fee_per_qty = 0;
    std::int64_t taker_fee_per_qty = 0;
};

// TCA convention descriptor (fee sign, markout horizon, arrival/terminal
// benchmarks, residual handling, and assumed latency) for comparing reports.
enum class FeeSignConvention : std::uint8_t {
    Unspecified = 0,
    PositiveIsCost,  // fee_per_qty/fee_cost > 0 is a cost paid; < 0 is a rebate
};

enum class BenchmarkReference : std::uint8_t {
    Unspecified = 0,
    ArrivalMid,    // shortfall vs mid at parent arrival
    FillMid,       // effective spread vs mid at fill
    IntervalVwap,
    PrevClose,
};

enum class TerminalMarkConvention : std::uint8_t {
    Unspecified = 0,
    MarkToMid,            // residual inventory marked at terminal mid
    BookLiquidation,      // residual liquidated against the book (impact-aware)
    ForcedFlatNoResidual, // assume flat; residual ignored (optimistic)
};

enum class ResidualInventoryPolicy : std::uint8_t {
    Unspecified = 0,
    MarkedAtTerminal,
    Ignored,
    Penalized,
};

struct TcaConventions {
    FeeSignConvention fee_sign = FeeSignConvention::Unspecified;
    BenchmarkReference benchmark = BenchmarkReference::Unspecified;
    TerminalMarkConvention terminal_mark = TerminalMarkConvention::Unspecified;
    ResidualInventoryPolicy residual_policy = ResidualInventoryPolicy::Unspecified;
    // Markout horizon the report's markout columns were evaluated at (scalars to
    // avoid coupling tca.hpp to the markout-observer header).
    EventSeq markout_horizon_events = 0;
    TimeNs markout_horizon_ns = 0;
    // Latency baked into the replay that produced these fills (0 = zero-latency
    // idealization). Documents the assumption; does not apply it.
    TimeNs assumed_latency_ns = 0;
    // Fee schedule actually applied, so the report carries its own fee basis.
    FeeSchedule fees{};
};

// Two TCA reports may be compared head-to-head only if every convention axis
// matches; otherwise their metrics are not on the same footing.
inline constexpr bool conventions_comparable(const TcaConventions& a,
                                             const TcaConventions& b) noexcept {
    return a.fee_sign == b.fee_sign && a.benchmark == b.benchmark &&
           a.terminal_mark == b.terminal_mark &&
           a.residual_policy == b.residual_policy &&
           a.markout_horizon_events == b.markout_horizon_events &&
           a.markout_horizon_ns == b.markout_horizon_ns &&
           a.assumed_latency_ns == b.assumed_latency_ns &&
           a.fees.maker_fee_per_qty == b.fees.maker_fee_per_qty &&
           a.fees.taker_fee_per_qty == b.fees.taker_fee_per_qty;
}

struct FillTcaMetrics {
    bool has_arrival_mid = false;
    bool has_fill_mid = false;
    bool has_future_mid = false;
    Tick implementation_shortfall_x2 = 0;
    Tick taker_effective_spread_x2 = 0;
    Tick maker_effective_spread_x2 = 0;
    Tick maker_realized_spread_x2 = 0;
    Tick taker_adverse_selection_x2 = 0;
};

inline constexpr std::size_t kTcaSummaryExportCount = 21;
inline constexpr std::size_t kFillTcaMetricsExportCount = 8;
inline constexpr std::array<ExportColumn, kTcaSummaryExportCount>
    kTcaSummaryExportSchema = {{{"passive_qty"},
                                {"aggressive_qty"},
                                {"total_qty"},
                                {"passive_notional"},
                                {"aggressive_notional"},
                                {"total_notional"},
                                {"passive_fee_cost"},
                                {"aggressive_fee_cost"},
                                {"total_fee_cost"},
                                {"total_net_notional"},
                                {"average_fill_price"},
                                {"average_passive_fee_per_qty"},
                                {"average_aggressive_fee_per_qty"},
                                {"passive_markout_x2_qty"},
                                {"aggressive_markout_x2_qty"},
                                {"passive_markout_qty"},
                                {"aggressive_markout_qty"},
                                {"average_passive_markout_ticks"},
                                {"average_aggressive_markout_ticks"},
                                {"average_net_passive_markout_ticks"},
                                {"average_net_aggressive_markout_ticks"}}};
inline constexpr std::array<ExportColumn, kFillTcaMetricsExportCount>
    kFillTcaMetricsExportSchema = {{{"has_arrival_mid"},
                                    {"has_fill_mid"},
                                    {"has_future_mid"},
                                    {"implementation_shortfall_x2"},
                                    {"taker_effective_spread_x2"},
                                    {"maker_effective_spread_x2"},
                                    {"maker_realized_spread_x2"},
                                    {"taker_adverse_selection_x2"}}};

struct TcaKey {
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;

    friend bool operator==(const TcaKey& lhs, const TcaKey& rhs) noexcept {
        return lhs.parent_id == rhs.parent_id && lhs.agent == rhs.agent &&
               lhs.source == rhs.source;
    }
};

struct TcaKeyHash {
    std::size_t operator()(const TcaKey& key) const noexcept {
        std::size_t seed = std::hash<ParentOrderId>{}(key.parent_id);
        seed ^= std::hash<AgentId>{}(key.agent) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<ScenarioId>{}(key.source) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

inline TcaKey maker_tca_key(const FillRecord& fill) noexcept {
    return TcaKey{.parent_id = fill.maker_parent_id,
                  .agent = fill.maker,
                  .source = fill.source};
}

inline TcaKey taker_tca_key(const FillRecord& fill) noexcept {
    return TcaKey{.parent_id = fill.taker_parent_id,
                  .agent = fill.taker,
                  .source = fill.source};
}

inline Qty total_qty(const TcaSummary& summary) noexcept {
    return static_cast<Qty>(summary.passive_qty + summary.aggressive_qty);
}

inline std::int64_t total_notional(const TcaSummary& summary) noexcept {
    return summary.passive_notional + summary.aggressive_notional;
}

inline std::int64_t total_fee_cost(const TcaSummary& summary) noexcept {
    return summary.passive_fee_cost + summary.aggressive_fee_cost;
}

inline std::int64_t total_net_notional(const TcaSummary& summary) noexcept {
    return total_notional(summary) + total_fee_cost(summary);
}

inline double average_fill_price(const TcaSummary& summary) noexcept {
    const Qty qty = total_qty(summary);
    if (qty == 0) return 0.0;
    return static_cast<double>(total_notional(summary)) / static_cast<double>(qty);
}

inline double average_fee_per_qty(std::int64_t fee_cost, Qty qty) noexcept {
    if (qty == 0) return 0.0;
    return static_cast<double>(fee_cost) / static_cast<double>(qty);
}

inline double average_passive_fee_per_qty(const TcaSummary& summary) noexcept {
    return average_fee_per_qty(summary.passive_fee_cost, summary.passive_qty);
}

inline double average_aggressive_fee_per_qty(const TcaSummary& summary) noexcept {
    return average_fee_per_qty(summary.aggressive_fee_cost, summary.aggressive_qty);
}

inline double average_passive_markout_ticks(const TcaSummary& summary) noexcept {
    if (summary.passive_markout_qty == 0) return 0.0;
    return static_cast<double>(summary.passive_markout_x2_qty) /
           (2.0 * static_cast<double>(summary.passive_markout_qty));
}

inline double average_aggressive_markout_ticks(const TcaSummary& summary) noexcept {
    if (summary.aggressive_markout_qty == 0) return 0.0;
    return static_cast<double>(summary.aggressive_markout_x2_qty) /
           (2.0 * static_cast<double>(summary.aggressive_markout_qty));
}

inline double average_net_passive_markout_ticks(const TcaSummary& summary) noexcept {
    return average_passive_markout_ticks(summary) -
           average_passive_fee_per_qty(summary);
}

inline double average_net_aggressive_markout_ticks(const TcaSummary& summary) noexcept {
    return average_aggressive_markout_ticks(summary) -
           average_aggressive_fee_per_qty(summary);
}

inline std::array<double, kTcaSummaryExportCount>
export_tca_summary(const TcaSummary& summary) noexcept {
    return {static_cast<double>(summary.passive_qty),
            static_cast<double>(summary.aggressive_qty),
            static_cast<double>(total_qty(summary)),
            static_cast<double>(summary.passive_notional),
            static_cast<double>(summary.aggressive_notional),
            static_cast<double>(total_notional(summary)),
            static_cast<double>(summary.passive_fee_cost),
            static_cast<double>(summary.aggressive_fee_cost),
            static_cast<double>(total_fee_cost(summary)),
            static_cast<double>(total_net_notional(summary)),
            average_fill_price(summary),
            average_passive_fee_per_qty(summary),
            average_aggressive_fee_per_qty(summary),
            static_cast<double>(summary.passive_markout_x2_qty),
            static_cast<double>(summary.aggressive_markout_x2_qty),
            static_cast<double>(summary.passive_markout_qty),
            static_cast<double>(summary.aggressive_markout_qty),
            average_passive_markout_ticks(summary),
            average_aggressive_markout_ticks(summary),
            average_net_passive_markout_ticks(summary),
            average_net_aggressive_markout_ticks(summary)};
}

inline Tick signed_shortfall_x2(Side side,
                                Tick fill_price,
                                Tick benchmark_mid_x2) noexcept {
    const Tick fill_x2 = static_cast<Tick>(fill_price * 2);
    const Tick sign = 1 - 2 * static_cast<Tick>(side_idx(side));
    return static_cast<Tick>((fill_x2 - benchmark_mid_x2) * sign);
}

inline Tick taker_effective_spread_x2(const FillRecord& fill) noexcept {
    if (!fill.has_mid_at_fill) return 0;
    return signed_shortfall_x2(fill.taker_side, fill.price, fill.mid_x2_at_fill);
}

inline Tick maker_effective_spread_x2(const FillRecord& fill) noexcept {
    if (!fill.has_mid_at_fill) return 0;
    return signed_markout_x2(fill.maker_side, fill.price, fill.mid_x2_at_fill);
}

inline Tick maker_realized_spread_x2(const MarkoutRecord& markout) noexcept {
    if (!markout.has_mid_after) return 0;
    return markout.maker_markout_x2;
}

inline Tick taker_adverse_selection_x2(const MarkoutRecord& markout) noexcept {
    if (!markout.has_mid_after) return 0;
    return static_cast<Tick>(-markout.taker_markout_x2);
}

inline FillTcaMetrics make_fill_tca_metrics(const FillRecord& fill,
                                            const MarkoutRecord& markout,
                                            bool has_arrival_mid = false,
                                            Tick arrival_mid_x2 = 0) noexcept {
    FillTcaMetrics out{.has_arrival_mid = has_arrival_mid,
                       .has_fill_mid = fill.has_mid_at_fill,
                       .has_future_mid = markout.has_mid_after,
                       .implementation_shortfall_x2 = 0,
                       .taker_effective_spread_x2 = taker_effective_spread_x2(fill),
                       .maker_effective_spread_x2 = maker_effective_spread_x2(fill),
                       .maker_realized_spread_x2 = maker_realized_spread_x2(markout),
                       .taker_adverse_selection_x2 = taker_adverse_selection_x2(markout)};
    if (has_arrival_mid) {
        out.implementation_shortfall_x2 =
            signed_shortfall_x2(fill.taker_side, fill.price, arrival_mid_x2);
    }
    return out;
}

inline std::array<double, kFillTcaMetricsExportCount>
export_fill_tca_metrics(const FillTcaMetrics& metrics) noexcept {
    return {metrics.has_arrival_mid ? 1.0 : 0.0,
            metrics.has_fill_mid ? 1.0 : 0.0,
            metrics.has_future_mid ? 1.0 : 0.0,
            static_cast<double>(metrics.implementation_shortfall_x2),
            static_cast<double>(metrics.taker_effective_spread_x2),
            static_cast<double>(metrics.maker_effective_spread_x2),
            static_cast<double>(metrics.maker_realized_spread_x2),
            static_cast<double>(metrics.taker_adverse_selection_x2)};
}

inline void add_passive_fill(TcaSummary& summary, const FillRecord& fill) noexcept {
    summary.passive_qty = static_cast<Qty>(summary.passive_qty + fill.qty);
    summary.passive_notional += static_cast<std::int64_t>(fill.price) *
                                static_cast<std::int64_t>(fill.qty);
}

inline void add_aggressive_fill(TcaSummary& summary, const FillRecord& fill) noexcept {
    summary.aggressive_qty = static_cast<Qty>(summary.aggressive_qty + fill.qty);
    summary.aggressive_notional += static_cast<std::int64_t>(fill.price) *
                                   static_cast<std::int64_t>(fill.qty);
}

inline void add_maker_fill(TcaSummary& summary, const FillRecord& fill) noexcept {
    add_passive_fill(summary, fill);
}

inline void add_taker_fill(TcaSummary& summary, const FillRecord& fill) noexcept {
    add_aggressive_fill(summary, fill);
}

inline void add_maker_fee(TcaSummary& summary,
                          const FillRecord& fill,
                          const FeeSchedule& fees) noexcept {
    summary.passive_fee_cost += fees.maker_fee_per_qty *
                                static_cast<std::int64_t>(fill.qty);
}

inline void add_taker_fee(TcaSummary& summary,
                          const FillRecord& fill,
                          const FeeSchedule& fees) noexcept {
    summary.aggressive_fee_cost += fees.taker_fee_per_qty *
                                   static_cast<std::int64_t>(fill.qty);
}

inline void add_maker_fill_with_fees(TcaSummary& summary,
                                     const FillRecord& fill,
                                     const FeeSchedule& fees) noexcept {
    add_maker_fill(summary, fill);
    add_maker_fee(summary, fill, fees);
}

inline void add_taker_fill_with_fees(TcaSummary& summary,
                                     const FillRecord& fill,
                                     const FeeSchedule& fees) noexcept {
    add_taker_fill(summary, fill);
    add_taker_fee(summary, fill, fees);
}

inline void add_maker_markout(TcaSummary& summary, const MarkoutRecord& markout) noexcept {
    if (!markout.has_mid_after) return;
    summary.passive_markout_x2_qty +=
        static_cast<std::int64_t>(markout.maker_markout_x2) *
        static_cast<std::int64_t>(markout.fill.qty);
    summary.passive_markout_qty =
        static_cast<Qty>(summary.passive_markout_qty + markout.fill.qty);
}

inline void add_taker_markout(TcaSummary& summary, const MarkoutRecord& markout) noexcept {
    if (!markout.has_mid_after) return;
    summary.aggressive_markout_x2_qty +=
        static_cast<std::int64_t>(markout.taker_markout_x2) *
        static_cast<std::int64_t>(markout.fill.qty);
    summary.aggressive_markout_qty =
        static_cast<Qty>(summary.aggressive_markout_qty + markout.fill.qty);
}

class TcaAccumulator {
public:
    using Map = std::unordered_map<TcaKey, TcaSummary, TcaKeyHash>;

    void add_maker_fill(const FillRecord& fill) {
        sim::add_maker_fill(summaries_[maker_tca_key(fill)], fill);
    }

    void add_taker_fill(const FillRecord& fill) {
        sim::add_taker_fill(summaries_[taker_tca_key(fill)], fill);
    }

    void add_maker_fill_with_fees(const FillRecord& fill, const FeeSchedule& fees) {
        sim::add_maker_fill_with_fees(summaries_[maker_tca_key(fill)], fill, fees);
    }

    void add_taker_fill_with_fees(const FillRecord& fill, const FeeSchedule& fees) {
        sim::add_taker_fill_with_fees(summaries_[taker_tca_key(fill)], fill, fees);
    }

    void add_maker_markout(const MarkoutRecord& markout) {
        sim::add_maker_markout(summaries_[maker_tca_key(markout.fill)], markout);
    }

    void add_taker_markout(const MarkoutRecord& markout) {
        sim::add_taker_markout(summaries_[taker_tca_key(markout.fill)], markout);
    }

    const TcaSummary* find(const TcaKey& key) const {
        const auto it = summaries_.find(key);
        if (it == summaries_.end()) return nullptr;
        return &it->second;
    }

    const Map& summaries() const noexcept { return summaries_; }
    std::size_t size() const noexcept { return summaries_.size(); }

private:
    Map summaries_{};
};

} // namespace sim
