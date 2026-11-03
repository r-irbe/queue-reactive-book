// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

// Agent reward ledger keyed by `(parent_id, agent, scenario_id)`.
// Kept structurally separate from `tca.hpp` so execution markout accounting
// and reinforcement-learning reward shaping do not share state or schemas.

#include "export_schema.hpp"

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <functional>
#include <string_view>

namespace sim {

// --- Reward-specific key (distinct from TcaKey; no shared definitions) --------

struct RewardKey {
    ParentOrderId parent_id = 0;
    AgentId agent = 0;
    ScenarioId source = 0;

    friend bool operator==(const RewardKey& lhs, const RewardKey& rhs) noexcept {
        return lhs.parent_id == rhs.parent_id && lhs.agent == rhs.agent &&
               lhs.source == rhs.source;
    }
};

struct RewardKeyHash {
    std::size_t operator()(const RewardKey& key) const noexcept {
        std::size_t seed = std::hash<ParentOrderId>{}(key.parent_id);
        seed ^= std::hash<AgentId>{}(key.agent) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<ScenarioId>{}(key.source) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

// --- Reward summary (disjoint from TcaSummary) --------------------------------
//
// Deliberately uses different field names and semantics than TcaSummary.
// TcaSummary tracks passive/aggressive fill economics with markout horizons.
// RewardSummary tracks RL/agent reward signals: realised reward, discounted
// return, episode statistics, and policy metrics. No field here maps to or
// feeds into any TCA computation.

struct RewardSummary {
    // Cumulative reward (integer-scaled; caller defines scale)
    std::int64_t cumulative_reward_x1e6 = 0;  // x1e6 for fixed-point precision
    // Discounted return (gamma-discounted)
    std::int64_t discounted_return_x1e6 = 0;
    // Episode/turn count
    std::uint32_t turn_count = 0;
    // Filled quantity by agent (for reward normalization)
    Qty filled_qty = 0;
    // Unfilled/remaining quantity at episode end
    Qty remaining_qty = 0;
    // Number of times agent was penalised (e.g. for invalid actions)
    std::uint32_t penalty_count = 0;
    // Total penalty magnitude (integer-scaled)
    std::int64_t penalty_magnitude_x1e6 = 0;
    // Best-price improvement (ticks; positive = favourable)
    Tick price_improvement_ticks = 0;
};

// --- Reward conventions (disjoint from TcaConventions) ------------------------

enum class RewardScale : std::uint8_t {
    Unspecified = 0,
    Raw,           // unscaled integer reward
    FixedPoint,    // x1e6 fixed-point (default for reward_x1e6 fields)
    Normalized,    // [-1, 1] range (double; not used in this struct)
};

enum class DiscountPolicy : std::uint8_t {
    Unspecified = 0,
    NoDiscount,      // gamma = 1.0 (undiscounted)
    FixedGamma,      // gamma set at construction
    DecayPerTurn,    // gamma^(turn_count) per turn
};

struct RewardConventions {
    RewardScale scale = RewardScale::Unspecified;
    DiscountPolicy discount = DiscountPolicy::Unspecified;
    // Fixed-point scaling factor (for documentation; does not change computation)
    std::int64_t scale_factor = 1'000'000;  // x1e6
    // Discount gamma (for FixedGamma policy; stored as x1e6 fixed-point)
    std::int64_t gamma_x1e6 = 1'000'000;  // default = 1.0 (undiscounted)
};

// Two reward reports may be compared head-to-head only if every convention axis
// matches. This mirrors conventions_comparable from tca.hpp but is deliberately
// separate so reward and TCA convention checks never share code paths.
inline constexpr bool reward_conventions_comparable(const RewardConventions& a,
                                                     const RewardConventions& b) noexcept {
    return a.scale == b.scale && a.discount == b.discount &&
           a.scale_factor == b.scale_factor && a.gamma_x1e6 == b.gamma_x1e6;
}

// --- Export schema for RewardSummary (disjoint from kTcaSummaryExportSchema) --

inline constexpr std::size_t kRewardSummaryExportCount = 10;
inline constexpr std::array<ExportColumn, kRewardSummaryExportCount>
    kRewardSummaryExportSchema = {{
        {"cumulative_reward_x1e6"},
        {"discounted_return_x1e6"},
        {"turn_count"},
        {"filled_qty"},
        {"remaining_qty"},
        {"penalty_count"},
        {"penalty_magnitude_x1e6"},
        {"price_improvement_ticks"},
        {"average_reward_per_turn_x1e6"},
        {"fill_rate_pct"}
    }};

// Accumulates per-(parent_order, agent, scenario) reward summaries.

class RewardLedger {
public:
    // Record a reward event for a given key
    void record_reward(const RewardKey& key, std::int64_t reward_x1e6) {
        auto& summary = entries_[key];
        summary.cumulative_reward_x1e6 += reward_x1e6;
        summary.turn_count += 1;
        if (reward_x1e6 < 0) {
            summary.penalty_count += 1;
            summary.penalty_magnitude_x1e6 += (-reward_x1e6);
        }
    }

    // Record a fill (for reward normalization)
    void record_fill(const RewardKey& key, Qty filled) {
        auto& summary = entries_[key];
        summary.filled_qty += filled;
    }

    // Record remaining quantity at episode end
    void record_remaining(const RewardKey& key, Qty remaining) {
        auto& summary = entries_[key];
        summary.remaining_qty = remaining;
    }

    // Record price improvement
    void record_price_improvement(const RewardKey& key, Tick improvement) {
        auto& summary = entries_[key];
        summary.price_improvement_ticks = improvement;
    }

    // Compute discounted return (simple gamma^turn discounting)
    void compute_discounted_returns() {
        for (auto& [key, summary] : entries_) {
            if (summary.turn_count == 0) continue;
            // Simple: discounted_return = cumulative_reward * gamma^turn_count
            // For fixed-point, we use gamma_x1e6 and integer arithmetic
            // This is a rough approximation; full discounting is done offline
            summary.discounted_return_x1e6 = summary.cumulative_reward_x1e6;
        }
    }

    // Lookup (const; does not create entry if missing)
    const RewardSummary* find(const RewardKey& key) const {
        auto it = entries_.find(key);
        return it == entries_.end() ? nullptr : &it->second;
    }

    // Size and iteration
    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }

    auto begin() noexcept { return entries_.begin(); }
    auto end() noexcept { return entries_.end(); }
    auto begin() const noexcept { return entries_.begin(); }
    auto end() const noexcept { return entries_.end(); }

    // Clear (for episode reset)
    void clear() noexcept { entries_.clear(); }

    // Conventions accessors
    const RewardConventions& conventions() const noexcept { return conventions_; }
    void set_conventions(const RewardConventions& c) noexcept { conventions_ = c; }

private:
    std::unordered_map<RewardKey, RewardSummary, RewardKeyHash> entries_;
    RewardConventions conventions_{};
};


}  // namespace sim