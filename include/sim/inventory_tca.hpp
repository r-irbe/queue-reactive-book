#pragma once

#include "fill_attribution.hpp"
#include "tca.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace sim {

struct InventoryKey {
    AgentId agent = 0;
    ScenarioId source = 0;

    friend bool operator==(const InventoryKey& lhs,
                           const InventoryKey& rhs) noexcept {
        return lhs.agent == rhs.agent && lhs.source == rhs.source;
    }
};

struct InventoryKeyHash {
    std::size_t operator()(const InventoryKey& key) const noexcept {
        std::size_t seed = std::hash<AgentId>{}(key.agent);
        seed ^= std::hash<ScenarioId>{}(key.source) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct InventoryState {
    InventoryKey key{};
    Qty position = 0;
    std::int64_t cash = 0;
    std::int64_t fee_cost = 0;
    Qty bought_qty = 0;
    Qty sold_qty = 0;
};

inline constexpr std::size_t kInventoryStateExportCount = 11;
inline constexpr std::array<ExportColumn, kInventoryStateExportCount>
    kInventoryStateExportSchema = {{{"agent"},
                                    {"source"},
                                    {"position"},
                                    {"cash"},
                                    {"fee_cost"},
                                    {"bought_qty"},
                                    {"sold_qty"},
                                    {"has_mid"},
                                    {"mid_x2"},
                                    {"mark_to_mid_value_x2"},
                                    {"mark_to_mid_value"}}};

inline std::int64_t notional(const FillRecord& fill) noexcept {
    return static_cast<std::int64_t>(fill.price) *
           static_cast<std::int64_t>(fill.qty);
}

inline void apply_inventory_fill(InventoryState& state,
                                 Side side,
                                 Tick price,
                                 Qty qty,
                                 std::int64_t fee_cost = 0) noexcept {
    const std::int64_t fill_notional =
        static_cast<std::int64_t>(price) * static_cast<std::int64_t>(qty);
    if (is_buy(side)) {
        state.position = static_cast<Qty>(state.position + qty);
        state.cash -= fill_notional;
        state.bought_qty = static_cast<Qty>(state.bought_qty + qty);
    } else {
        state.position = static_cast<Qty>(state.position - qty);
        state.cash += fill_notional;
        state.sold_qty = static_cast<Qty>(state.sold_qty + qty);
    }
    state.cash -= fee_cost;
    state.fee_cost += fee_cost;
}

inline std::int64_t mark_to_mid_value_x2(const InventoryState& state,
                                         Tick mid_x2) noexcept {
    return static_cast<std::int64_t>(state.cash) * 2 +
           static_cast<std::int64_t>(state.position) *
               static_cast<std::int64_t>(mid_x2);
}

inline double mark_to_mid_value(const InventoryState& state, Tick mid_x2) noexcept {
    return static_cast<double>(mark_to_mid_value_x2(state, mid_x2)) / 2.0;
}

inline std::array<double, kInventoryStateExportCount>
export_inventory_state(const InventoryState& state,
                       bool has_mid = false,
                       Tick mid_x2 = 0) noexcept {
    const std::int64_t mtm_x2 = has_mid ? mark_to_mid_value_x2(state, mid_x2) : 0;
    return {static_cast<double>(state.key.agent),
            static_cast<double>(state.key.source),
            static_cast<double>(state.position),
            static_cast<double>(state.cash),
            static_cast<double>(state.fee_cost),
            static_cast<double>(state.bought_qty),
            static_cast<double>(state.sold_qty),
            has_mid ? 1.0 : 0.0,
            has_mid ? static_cast<double>(mid_x2) : 0.0,
            static_cast<double>(mtm_x2),
            has_mid ? static_cast<double>(mtm_x2) / 2.0 : 0.0};
}

class InventoryTracker {
public:
    using Map = std::unordered_map<InventoryKey, InventoryState, InventoryKeyHash>;

    void add_maker_fill(const FillRecord& fill,
                        const FeeSchedule& fees = {}) {
        InventoryState& state =
            states_[InventoryKey{.agent = fill.maker, .source = fill.source}];
        state.key = InventoryKey{.agent = fill.maker, .source = fill.source};
        apply_inventory_fill(state, fill.maker_side, fill.price, fill.qty,
                             fees.maker_fee_per_qty *
                                 static_cast<std::int64_t>(fill.qty));
    }

    void add_taker_fill(const FillRecord& fill,
                        const FeeSchedule& fees = {}) {
        InventoryState& state =
            states_[InventoryKey{.agent = fill.taker, .source = fill.source}];
        state.key = InventoryKey{.agent = fill.taker, .source = fill.source};
        apply_inventory_fill(state, fill.taker_side, fill.price, fill.qty,
                             fees.taker_fee_per_qty *
                                 static_cast<std::int64_t>(fill.qty));
    }

    const InventoryState* find(const InventoryKey& key) const {
        const auto it = states_.find(key);
        if (it == states_.end()) return nullptr;
        return &it->second;
    }

    const Map& states() const noexcept { return states_; }
    std::size_t size() const noexcept { return states_.size(); }

private:
    Map states_{};
};

} // namespace sim
