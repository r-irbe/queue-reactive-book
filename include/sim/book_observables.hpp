#pragma once

#include "event_trace.hpp"
#include "flat_ladder_book.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace sim {

struct DepthLevel {
    Tick price = 0;
    Qty qty = 0;
    std::uint32_t order_count = 0;
};

template <std::size_t K>
struct TopKBookSnapshot {
    TimeNs ts = 0;
    EventSeq seq = 0;
    std::array<DepthLevel, K> bids{};
    std::array<DepthLevel, K> asks{};
    std::uint8_t bid_count = 0;
    std::uint8_t ask_count = 0;
};

struct BookObservables {
    TimeNs ts = 0;
    EventSeq seq = 0;
    bool has_bid = false;
    bool has_ask = false;
    Tick best_bid = 0;
    Tick best_ask = 0;
    Qty best_bid_qty = 0;
    Qty best_ask_qty = 0;
    bool has_mid = false;
    Tick mid_x2 = 0;
    Tick spread = 0;
    double micro_price = 0.0;
    Qty total_bid_qty = 0;
    Qty total_ask_qty = 0;
    double imbalance_l1 = 0.0;
    double imbalance_total = 0.0;
};

namespace detail {
inline std::uint32_t clamp_order_count(std::size_t n) noexcept {
    return n > std::numeric_limits<std::uint32_t>::max()
               ? std::numeric_limits<std::uint32_t>::max()
               : static_cast<std::uint32_t>(n);
}

inline double imbalance(Qty bid_qty, Qty ask_qty) noexcept {
    const auto total = static_cast<double>(bid_qty) + static_cast<double>(ask_qty);
    if (total == 0.0) return 0.0;
    return (static_cast<double>(bid_qty) - static_cast<double>(ask_qty)) / total;
}
} // namespace detail

inline BookObservables make_book_observables(const FlatLadderBook& book,
                                             EventSeq seq = 0) {
    BookObservables out{};
    out.ts = book.time();
    out.seq = seq;
    out.has_bid = book.best_bid(out.best_bid, out.best_bid_qty);
    out.has_ask = book.best_ask(out.best_ask, out.best_ask_qty);
    out.total_bid_qty = book.total_volume(Side::Buy);
    out.total_ask_qty = book.total_volume(Side::Sell);
    out.imbalance_l1 = detail::imbalance(out.best_bid_qty, out.best_ask_qty);
    out.imbalance_total = detail::imbalance(out.total_bid_qty, out.total_ask_qty);

    out.has_mid = out.has_bid && out.has_ask;
    if (!out.has_mid) return out;

    out.mid_x2 = static_cast<Tick>(out.best_bid + out.best_ask);
    out.spread = static_cast<Tick>(out.best_ask - out.best_bid);
    const auto touch_qty = static_cast<double>(out.best_bid_qty) +
                           static_cast<double>(out.best_ask_qty);
    if (touch_qty > 0.0) {
        out.micro_price =
            (static_cast<double>(out.best_ask) * static_cast<double>(out.best_bid_qty) +
             static_cast<double>(out.best_bid) * static_cast<double>(out.best_ask_qty)) /
            touch_qty;
    }
    return out;
}

template <std::size_t K>
inline TopKBookSnapshot<K> make_top_k_snapshot(const FlatLadderBook& book,
                                               EventSeq seq = 0) {
    TopKBookSnapshot<K> out{};
    out.ts = book.time();
    out.seq = seq;

    const auto bids = book.levels(Side::Buy, K);
    const auto asks = book.levels(Side::Sell, K);
    const auto bid_count = std::min<std::size_t>(bids.size(), K);
    const auto ask_count = std::min<std::size_t>(asks.size(), K);
    out.bid_count = static_cast<std::uint8_t>(std::min<std::size_t>(
        bid_count, std::numeric_limits<std::uint8_t>::max()));
    out.ask_count = static_cast<std::uint8_t>(std::min<std::size_t>(
        ask_count, std::numeric_limits<std::uint8_t>::max()));

    for (std::size_t i = 0; i < bid_count; ++i) {
        out.bids[i] = {bids[i].price, bids[i].volume,
                       detail::clamp_order_count(bids[i].orders)};
    }
    for (std::size_t i = 0; i < ask_count; ++i) {
        out.asks[i] = {asks[i].price, asks[i].volume,
                       detail::clamp_order_count(asks[i].orders)};
    }
    return out;
}

} // namespace sim
