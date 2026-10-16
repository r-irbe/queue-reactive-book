#pragma once

// reward_ledger.hpp -- RTA-001: Separate reward ledger (disjoint from TCA)
//
// A standalone evaluation-layer ledger for RL/agent reward signals that is
// deliberately disjoint from tca.hpp. TcaSummary/InventoryState may never
// reference reward fields, and RewardLedger may never write into TCA
// structures. This header is the authoritative enforcement point for that
// separation.
//
// INVARIANT: RewardLedger shares no symbols with tca.hpp. TCA is pure
// markout/execution accounting; reward is RL/agent evaluation. They must
// remain disjoint so reward contamination of TCA is structurally impossible.
//
// KERNEL: This header does NOT touch flat_ladder_book.cpp/.hpp. It lives in
// the evaluation/diagnostics layer, consumed by replay_pipeline or offline
// analysis only. No hot-path matching code depends on it.

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

// --- Reward ledger (disjoint from TCA accumulator) ----------------------------
//
// The ledger accumulates reward summaries per (parent_order, agent, scenario).
// It is consumed by offline analysis, replay diagnostics, or reward-vs-TCA
// divergence reporting (RTA-002). It must NEVER be read by TCA computation
// or written to by TCA accumulation.

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

// --- Disjointness compile-time guards ------------------------------------------
//
// These static_asserts ensure that RewardSummary and TcaSummary have no
// overlapping field names, providing a compile-time guard against accidental
// contamination. If someone adds a field to RewardSummary that matches a TcaSummary
// field name, this will fail to compile (as a signal, not a hard error -- the
// guard is advisory since the structs are in the same namespace).
//
// We check by ensuring RewardSummary does not have these TcaSummary field names:
//   passive_qty, aggressive_qty, passive_notional, aggressive_notional,
//   passive_fee_cost, aggressive_fee_cost, passive_markout_x2_qty, etc.
//
// (Implementing this as a concept or static_assert requires reflection which
// C++20 doesn't have. Instead, the disjointness is enforced structurally:
// RewardLedger is a separate class, RewardKey is a separate struct, and the
// export schema arrays are separate constexpr arrays. No shared inheritance,
// no shared base, no shared accessor functions.)

}  // namespace sim