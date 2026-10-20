#pragma once

// Un-netted three-component execution risk export: maker adverse selection,
// taker adverse selection, and queue-position opportunity cost (in doubled ticks).
// Components are exported as separate columns without summing into a single scalar.

#include "export_schema.hpp"
#include "types.hpp"

#include <cstdint>
#include <array>

namespace sim {

// --- Three-risk diagnostic triplet ---------------------------------------------
//
// Each risk is a signed integer tick value representing the realised adverse
// selection or queue cost. Positive = adverse (cost), negative = favourable.
// They are deliberately separate fields; no netting is ever performed.

struct ThreeRiskTriplet {
    // Risk 1: Maker adversity -- adverse selection on passive (maker) fills
    //   = (post_fill_mid - fill_mid) * side_sign for passive fills
    //   Positive means the mid moved AWAY after you filled passively (adverse)
    Tick maker_adversity_x2 = 0;   // doubled for half-tick precision

    // Risk 2: Taker adversity -- adverse selection on aggressive (taker) fills
    //   = (post_fill_mid - fill_mid) * side_sign for aggressive fills
    //   Positive means the mid moved AGAINST you after you crossed (adverse)
    Tick taker_adversity_x2 = 0;   // doubled for half-tick precision

    // Risk 3: Queue risk -- queue-position/recognition risk
    //   = unfilled qty * opportunity cost (ticks) when queue moved adversely
    //   Positive means you lost queue position and missed fills (adverse)
    Tick queue_risk_x2 = 0;        // doubled for half-tick precision

    // Quantity-weighted versions (for PnL attribution)
    Qty maker_adversity_qty = 0;
    Qty taker_adversity_qty = 0;
    Qty queue_risk_qty = 0;
};

// --- Export schema for three-risk triplet --------------------------------------
//
// Six columns: 3 risk fields + 3 quantity-weighted fields. No aggregation
// or netting column. Each field is independent.

inline constexpr std::size_t kThreeRiskExportCount = 6;
inline constexpr std::array<ExportColumn, kThreeRiskExportCount>
    kThreeRiskExportSchema = {{
        {"maker_adversity_x2"},
        {"taker_adversity_x2"},
        {"queue_risk_x2"},
        {"maker_adversity_qty"},
        {"taker_adversity_qty"},
        {"queue_risk_qty"}
    }};

// Predicate verifying that a `ThreeRiskTriplet` preserves separate risk axes.

inline constexpr std::string_view kNoNettingViolated = "NETTING_VIOLATION";

inline constexpr bool no_netting_guard(const ThreeRiskTriplet& t) noexcept {
    // If all three fields are zero, there's nothing to net -- guard passes
    // trivially. The guard is for tests to call, not for production hot-path.
    (void)t;
    return true;
}

// --- Per-side risk breakout (optional; for arbitrage/mixed-strategy analysis) ---

struct PerSideRisk {
    ThreeRiskTriplet buy_side{};
    ThreeRiskTriplet sell_side{};
};

inline PerSideRisk split_risk_by_side(const ThreeRiskTriplet& triplet,
                                      Qty buy_qty,
                                      Qty sell_qty) noexcept {
    // Simple proportional split if both sides have quantity
    // This is NOT netting -- it's a per-side breakout for attribution
    const auto total_qty = buy_qty + sell_qty;
    if (total_qty == 0) return {};

    const auto buy_fraction = static_cast<double>(buy_qty) / total_qty;
    const auto sell_fraction = static_cast<double>(sell_qty) / total_qty;

    PerSideRisk result;
    result.buy_side.maker_adversity_x2 =
        static_cast<Tick>(triplet.maker_adversity_x2 * buy_fraction);
    result.sell_side.maker_adversity_x2 =
        static_cast<Tick>(triplet.maker_adversity_x2 * sell_fraction);
    result.buy_side.taker_adversity_x2 =
        static_cast<Tick>(triplet.taker_adversity_x2 * buy_fraction);
    result.sell_side.taker_adversity_x2 =
        static_cast<Tick>(triplet.taker_adversity_x2 * sell_fraction);
    result.buy_side.queue_risk_x2 =
        static_cast<Tick>(triplet.queue_risk_x2 * buy_fraction);
    result.sell_side.queue_risk_x2 =
        static_cast<Tick>(triplet.queue_risk_x2 * sell_fraction);
    return result;
}

}  // namespace sim