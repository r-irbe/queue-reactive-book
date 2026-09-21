#pragma once

#include "book_observables.hpp"
#include "export_schema.hpp"

#include <array>
#include <cstddef>

namespace sim {

inline constexpr std::size_t kObservableFeatureCount = 14;
inline constexpr std::array<ExportColumn, kObservableFeatureCount>
    kObservableFeatureSchema = {{{"ts"},
                                 {"seq"},
                                 {"has_bid"},
                                 {"has_ask"},
                                 {"has_mid"},
                                 {"best_bid"},
                                 {"best_ask"},
                                 {"best_bid_qty"},
                                 {"best_ask_qty"},
                                 {"mid_x2"},
                                 {"spread"},
                                 {"micro_price"},
                                 {"imbalance_l1"},
                                 {"imbalance_total"}}};

template <std::size_t K>
inline constexpr std::size_t top_k_feature_count() noexcept {
    return 4U + 8U * K;
}

template <std::size_t K>
inline std::array<ExportColumn, top_k_feature_count<K>()> top_k_feature_schema()
    noexcept {
    std::array<ExportColumn, top_k_feature_count<K>()> out{};
    std::size_t pos = 0;
    out[pos++] = ExportColumn{.name = "ts"};
    out[pos++] = ExportColumn{.name = "seq"};
    out[pos++] = ExportColumn{.name = "bid_count"};
    out[pos++] = ExportColumn{.name = "ask_count"};

    for (std::size_t i = 0; i < K; ++i) {
        out[pos++] = ExportColumn{.name = "present", .side = "bid", .index = i};
        out[pos++] = ExportColumn{.name = "price", .side = "bid", .index = i};
        out[pos++] = ExportColumn{.name = "qty", .side = "bid", .index = i};
        out[pos++] =
            ExportColumn{.name = "order_count", .side = "bid", .index = i};
    }
    for (std::size_t i = 0; i < K; ++i) {
        out[pos++] = ExportColumn{.name = "present", .side = "ask", .index = i};
        out[pos++] = ExportColumn{.name = "price", .side = "ask", .index = i};
        out[pos++] = ExportColumn{.name = "qty", .side = "ask", .index = i};
        out[pos++] =
            ExportColumn{.name = "order_count", .side = "ask", .index = i};
    }
    return out;
}

inline std::array<double, kObservableFeatureCount>
export_observable_features(const BookObservables& obs) noexcept {
    return {static_cast<double>(obs.ts),
            static_cast<double>(obs.seq),
            obs.has_bid ? 1.0 : 0.0,
            obs.has_ask ? 1.0 : 0.0,
            obs.has_mid ? 1.0 : 0.0,
            static_cast<double>(obs.best_bid),
            static_cast<double>(obs.best_ask),
            static_cast<double>(obs.best_bid_qty),
            static_cast<double>(obs.best_ask_qty),
            static_cast<double>(obs.mid_x2),
            static_cast<double>(obs.spread),
            obs.micro_price,
            obs.imbalance_l1,
            obs.imbalance_total};
}

template <std::size_t K>
inline std::array<double, top_k_feature_count<K>()>
export_top_k_features(const TopKBookSnapshot<K>& snapshot) noexcept {
    std::array<double, top_k_feature_count<K>()> out{};
    std::size_t pos = 0;
    out[pos++] = static_cast<double>(snapshot.ts);
    out[pos++] = static_cast<double>(snapshot.seq);
    out[pos++] = static_cast<double>(snapshot.bid_count);
    out[pos++] = static_cast<double>(snapshot.ask_count);

    for (std::size_t i = 0; i < K; ++i) {
        const bool present = i < snapshot.bid_count;
        out[pos++] = present ? 1.0 : 0.0;
        out[pos++] = static_cast<double>(snapshot.bids[i].price);
        out[pos++] = static_cast<double>(snapshot.bids[i].qty);
        out[pos++] = static_cast<double>(snapshot.bids[i].order_count);
    }
    for (std::size_t i = 0; i < K; ++i) {
        const bool present = i < snapshot.ask_count;
        out[pos++] = present ? 1.0 : 0.0;
        out[pos++] = static_cast<double>(snapshot.asks[i].price);
        out[pos++] = static_cast<double>(snapshot.asks[i].qty);
        out[pos++] = static_cast<double>(snapshot.asks[i].order_count);
    }
    return out;
}

template <std::size_t K>
inline constexpr std::size_t book_feature_count() noexcept {
    return kObservableFeatureCount + top_k_feature_count<K>();
}

template <std::size_t K>
inline std::array<ExportColumn, book_feature_count<K>()> book_feature_schema()
    noexcept {
    std::array<ExportColumn, book_feature_count<K>()> out{};
    const auto depth_schema = top_k_feature_schema<K>();

    std::size_t pos = 0;
    for (const ExportColumn& column : kObservableFeatureSchema) {
        out[pos++] = column;
    }
    for (const ExportColumn& column : depth_schema) {
        out[pos++] = column;
    }
    return out;
}

template <std::size_t K>
inline std::array<double, book_feature_count<K>()>
export_book_features(const BookObservables& obs,
                     const TopKBookSnapshot<K>& snapshot) noexcept {
    std::array<double, book_feature_count<K>()> out{};
    const auto obs_features = export_observable_features(obs);
    const auto depth_features = export_top_k_features(snapshot);

    std::size_t pos = 0;
    for (double feature : obs_features) out[pos++] = feature;
    for (double feature : depth_features) out[pos++] = feature;
    return out;
}

} // namespace sim
