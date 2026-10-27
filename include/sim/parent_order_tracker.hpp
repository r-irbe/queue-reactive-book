// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "fill_attribution.hpp"
#include "risk_guard.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace sim {

struct ParentOrderKey {
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;
    Side side = Side::Buy;

    friend bool operator==(const ParentOrderKey& lhs,
                           const ParentOrderKey& rhs) noexcept {
        return lhs.parent_id == rhs.parent_id && lhs.agent == rhs.agent &&
               lhs.source == rhs.source && lhs.side == rhs.side;
    }
};

struct ParentOrderKeyHash {
    std::size_t operator()(const ParentOrderKey& key) const noexcept {
        std::size_t seed = std::hash<ParentOrderId>{}(key.parent_id);
        seed ^= std::hash<AgentId>{}(key.agent) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<ScenarioId>{}(key.source) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.side)) +
                0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct ParentOrderProgress {
    ParentOrderKey key{};
    Qty submitted_qty = 0;
    Qty passive_submitted_qty = 0;
    Qty aggressive_submitted_qty = 0;
    Qty passive_filled_qty = 0;
    Qty aggressive_filled_qty = 0;
    std::int64_t fill_notional = 0;
};

struct OpportunityCostMetrics {
    bool has_benchmark_mid = false;
    bool has_terminal_mid = false;
    Qty unfilled_qty = 0;
    Tick opportunity_cost_x2_per_qty = 0;
    std::int64_t opportunity_cost_x2_qty = 0;
};

inline Qty filled_qty(const ParentOrderProgress& progress) noexcept {
    return static_cast<Qty>(progress.passive_filled_qty +
                            progress.aggressive_filled_qty);
}

inline Qty remaining_qty(const ParentOrderProgress& progress) noexcept {
    const Qty filled = filled_qty(progress);
    if (filled >= progress.submitted_qty) return 0;
    return static_cast<Qty>(progress.submitted_qty - filled);
}

inline double completion_rate(const ParentOrderProgress& progress) noexcept {
    if (progress.submitted_qty <= 0) return 0.0;
    return static_cast<double>(filled_qty(progress)) /
           static_cast<double>(progress.submitted_qty);
}

inline Qty passive_queue_miss_qty(const ParentOrderProgress& progress) noexcept {
    if (progress.passive_filled_qty >= progress.passive_submitted_qty) return 0;
    return static_cast<Qty>(progress.passive_submitted_qty -
                            progress.passive_filled_qty);
}

inline double passive_fill_rate(const ParentOrderProgress& progress) noexcept {
    if (progress.passive_submitted_qty <= 0) return 0.0;
    return static_cast<double>(progress.passive_filled_qty) /
           static_cast<double>(progress.passive_submitted_qty);
}

inline double aggressive_fill_rate(const ParentOrderProgress& progress) noexcept {
    if (progress.aggressive_submitted_qty <= 0) return 0.0;
    return static_cast<double>(progress.aggressive_filled_qty) /
           static_cast<double>(progress.aggressive_submitted_qty);
}

inline double passive_queue_miss_rate(const ParentOrderProgress& progress) noexcept {
    if (progress.passive_submitted_qty <= 0) return 0.0;
    return static_cast<double>(passive_queue_miss_qty(progress)) /
           static_cast<double>(progress.passive_submitted_qty);
}

inline double average_parent_fill_price(const ParentOrderProgress& progress) noexcept {
    const Qty filled = filled_qty(progress);
    if (filled <= 0) return 0.0;
    return static_cast<double>(progress.fill_notional) / static_cast<double>(filled);
}

inline Tick opportunity_cost_x2_per_qty(Side side,
                                        Tick benchmark_mid_x2,
                                        Tick terminal_mid_x2) noexcept {
    return is_buy(side) ? static_cast<Tick>(terminal_mid_x2 - benchmark_mid_x2)
                        : static_cast<Tick>(benchmark_mid_x2 - terminal_mid_x2);
}

inline OpportunityCostMetrics make_opportunity_cost_metrics(
    const ParentOrderProgress& progress,
    bool has_benchmark_mid,
    Tick benchmark_mid_x2,
    bool has_terminal_mid,
    Tick terminal_mid_x2) noexcept {
    OpportunityCostMetrics out{.has_benchmark_mid = has_benchmark_mid,
                               .has_terminal_mid = has_terminal_mid,
                               .unfilled_qty = remaining_qty(progress),
                               .opportunity_cost_x2_per_qty = 0,
                               .opportunity_cost_x2_qty = 0};
    if (!has_benchmark_mid || !has_terminal_mid) return out;

    out.opportunity_cost_x2_per_qty =
        opportunity_cost_x2_per_qty(progress.key.side, benchmark_mid_x2, terminal_mid_x2);
    out.opportunity_cost_x2_qty =
        static_cast<std::int64_t>(out.opportunity_cost_x2_per_qty) *
        static_cast<std::int64_t>(out.unfilled_qty);
    return out;
}

inline double average_opportunity_cost_ticks(
    const OpportunityCostMetrics& metrics) noexcept {
    if (!metrics.has_benchmark_mid || !metrics.has_terminal_mid ||
        metrics.unfilled_qty == 0) {
        return 0.0;
    }
    return static_cast<double>(metrics.opportunity_cost_x2_qty) /
           (2.0 * static_cast<double>(metrics.unfilled_qty));
}

inline ParentOrderKey parent_key_from_intent(const StrategyIntent& intent) noexcept {
    return ParentOrderKey{.parent_id = intent.parent_id,
                          .agent = intent.agent,
                          .source = intent.source,
                          .side = intent.side};
}

inline ParentOrderKey maker_parent_key(const FillRecord& fill) noexcept {
    return ParentOrderKey{.parent_id = fill.maker_parent_id,
                          .agent = fill.maker,
                          .source = fill.source,
                          .side = fill.maker_side};
}

inline ParentOrderKey taker_parent_key(const FillRecord& fill) noexcept {
    return ParentOrderKey{.parent_id = fill.taker_parent_id,
                          .agent = fill.taker,
                          .source = fill.source,
                          .side = fill.taker_side};
}

class ParentOrderTracker {
public:
    using Map = std::unordered_map<ParentOrderKey, ParentOrderProgress, ParentOrderKeyHash>;

    void record_submission(const StrategyIntent& intent) {
        if (intent.kind == StrategyIntentKind::Cancel || intent.qty <= 0) return;
        const ParentOrderKey key = parent_key_from_intent(intent);
        ParentOrderProgress& progress = progress_[key];
        progress.key = key;
        progress.submitted_qty = static_cast<Qty>(progress.submitted_qty + intent.qty);
        if (intent.kind == StrategyIntentKind::SubmitLimit) {
            progress.passive_submitted_qty =
                static_cast<Qty>(progress.passive_submitted_qty + intent.qty);
        } else if (intent.kind == StrategyIntentKind::SubmitMarket) {
            progress.aggressive_submitted_qty =
                static_cast<Qty>(progress.aggressive_submitted_qty + intent.qty);
        }
    }

    void record_maker_fill(const FillRecord& fill) {
        ParentOrderProgress& progress = progress_[maker_parent_key(fill)];
        progress.key = maker_parent_key(fill);
        progress.passive_filled_qty =
            static_cast<Qty>(progress.passive_filled_qty + fill.qty);
        progress.fill_notional += static_cast<std::int64_t>(fill.price) *
                                  static_cast<std::int64_t>(fill.qty);
    }

    void record_taker_fill(const FillRecord& fill) {
        ParentOrderProgress& progress = progress_[taker_parent_key(fill)];
        progress.key = taker_parent_key(fill);
        progress.aggressive_filled_qty =
            static_cast<Qty>(progress.aggressive_filled_qty + fill.qty);
        progress.fill_notional += static_cast<std::int64_t>(fill.price) *
                                  static_cast<std::int64_t>(fill.qty);
    }

    const ParentOrderProgress* find(const ParentOrderKey& key) const {
        const auto it = progress_.find(key);
        if (it == progress_.end()) return nullptr;
        return &it->second;
    }

    const Map& progress() const noexcept { return progress_; }
    std::size_t size() const noexcept { return progress_.size(); }

private:
    Map progress_{};
};

} // namespace sim
