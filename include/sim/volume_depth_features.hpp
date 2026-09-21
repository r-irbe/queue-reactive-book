#pragma once

#include "book_observables.hpp"
#include "export_schema.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sim {

template <std::size_t K>
inline constexpr std::size_t volume_depth_feature_count() noexcept {
    return 4U + 5U * K;
}

template <std::size_t K>
inline std::array<ExportColumn, volume_depth_feature_count<K>()>
volume_depth_feature_schema() noexcept {
    std::array<ExportColumn, volume_depth_feature_count<K>()> out{};
    std::size_t pos = 0;
    out[pos++] = ExportColumn{.name = "ts"};
    out[pos++] = ExportColumn{.name = "seq"};
    out[pos++] = ExportColumn{.name = "bid_count"};
    out[pos++] = ExportColumn{.name = "ask_count"};

    for (std::size_t i = 0; i < K; ++i) {
        out[pos++] = ExportColumn{.name = "cumulative_qty", .side = "bid", .index = i};
        out[pos++] = ExportColumn{.name = "cumulative_qty", .side = "ask", .index = i};
        out[pos++] =
            ExportColumn{.name = "cumulative_notional", .side = "bid", .index = i};
        out[pos++] =
            ExportColumn{.name = "cumulative_notional", .side = "ask", .index = i};
        out[pos++] = ExportColumn{.name = "cumulative_depth_imbalance", .index = i};
    }
    return out;
}

inline std::int64_t level_notional(const DepthLevel& level) noexcept {
    return static_cast<std::int64_t>(level.price) *
           static_cast<std::int64_t>(level.qty);
}

inline double cumulative_depth_imbalance(Qty bid_qty, Qty ask_qty) noexcept {
    const auto total = static_cast<double>(bid_qty) + static_cast<double>(ask_qty);
    if (total == 0.0) return 0.0;
    return (static_cast<double>(bid_qty) - static_cast<double>(ask_qty)) / total;
}

template <std::size_t K>
inline std::array<double, volume_depth_feature_count<K>()>
export_volume_depth_features(const TopKBookSnapshot<K>& snapshot) noexcept {
    std::array<double, volume_depth_feature_count<K>()> out{};
    std::size_t pos = 0;
    out[pos++] = static_cast<double>(snapshot.ts);
    out[pos++] = static_cast<double>(snapshot.seq);
    out[pos++] = static_cast<double>(snapshot.bid_count);
    out[pos++] = static_cast<double>(snapshot.ask_count);

    Qty cumulative_bid_qty = 0;
    Qty cumulative_ask_qty = 0;
    std::int64_t cumulative_bid_notional = 0;
    std::int64_t cumulative_ask_notional = 0;

    for (std::size_t i = 0; i < K; ++i) {
        if (i < snapshot.bid_count) {
            cumulative_bid_qty =
                static_cast<Qty>(cumulative_bid_qty + snapshot.bids[i].qty);
            cumulative_bid_notional += level_notional(snapshot.bids[i]);
        }
        if (i < snapshot.ask_count) {
            cumulative_ask_qty =
                static_cast<Qty>(cumulative_ask_qty + snapshot.asks[i].qty);
            cumulative_ask_notional += level_notional(snapshot.asks[i]);
        }

        out[pos++] = static_cast<double>(cumulative_bid_qty);
        out[pos++] = static_cast<double>(cumulative_ask_qty);
        out[pos++] = static_cast<double>(cumulative_bid_notional);
        out[pos++] = static_cast<double>(cumulative_ask_notional);
        out[pos++] = cumulative_depth_imbalance(cumulative_bid_qty, cumulative_ask_qty);
    }
    return out;
}

} // namespace sim
