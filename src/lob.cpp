#include "sim/lob.hpp"

namespace sim {

LimitOrderBook::LimitOrderBook(std::size_t expected_orders,
                               bool enable_compaction,
                               CompactionConfig compaction_cfg,
                               std::size_t arena_bytes)
    : arena_((arena_bytes == 0 ? suggest_arena_bytes(expected_orders) : arena_bytes),
             std::pmr::new_delete_resource()),
      pool_(&arena_),
      bids_(resource()),
      asks_(resource()),
      id_index_(resource()),
      enable_compaction_(enable_compaction),
      compaction_cfg_(compaction_cfg) {
    id_index_.max_load_factor(0.7f);
    id_index_.reserve(expected_orders);
}

ExecutionResult LimitOrderBook::add(Order order, StpMode stp) {
    ExecutionResult result;
    if (order.qty <= 0) {
        return result;
    }

    order.id       = next_id_++;
    order.orig_qty = order.qty;
    const StpMode effective_stp = (stp != StpMode::None)
                                      ? stp
                                      : (order.stp != StpMode::None ? order.stp : default_stp_mode_);

    if (order.type == OrderType::PostOnly) {
        if (is_buy(order.side)) {
            if (!asks_.empty() && asks_.begin()->first <= order.price) {
                return result;
            }
        } else {
            if (!bids_.empty() && bids_.begin()->first >= order.price) {
                return result;
            }
        }
        rest(order);
        result.remaining = order.qty;
        return result;
    }

    if (order.type == OrderType::FOK) {
        Qty available = 0;
        auto can_fill = [&](const auto& book_map, auto price_ok) -> bool {
            for (const auto& [lvl_price, level] : book_map) {
                if (!price_ok(lvl_price)) break;
                if (effective_stp == StpMode::None) {
                    available += level.volume;
                    if (available >= order.qty) return true;
                } else {
                    for (std::size_t i = level.head; i < level.orders.size(); ++i) {
                        const Order& maker = level.orders[i];
                        if (maker.qty <= 0) continue;
                        if (maker.agent == order.agent) {
                            if (effective_stp == StpMode::CancelAggressive ||
                                effective_stp == StpMode::DecrementAndCancel) {
                                return available >= order.qty;
                            }
                            continue;
                        }
                        available += maker.qty;
                        if (available >= order.qty) return true;
                    }
                }
            }
            return available >= order.qty;
        };

        const bool ok = is_buy(order.side)
                            ? can_fill(asks_, [&](Tick p) { return p <= order.price; })
                            : can_fill(bids_, [&](Tick p) { return p >= order.price; });
        if (!ok) {
            return result;
        }
    }

    const bool is_limit = (order.type == OrderType::Limit ||
                           order.type == OrderType::IOC ||
                           order.type == OrderType::FOK);

    if (is_buy(order.side))
        match_against_asks(order, result, is_limit, order.price, effective_stp);
    else
        match_against_bids(order, result, is_limit, order.price, effective_stp);

    if (order.type == OrderType::Limit && order.qty > 0)
        rest(order);

    result.remaining = order.qty;
    return result;
}

void LimitOrderBook::match_against_asks(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp) {
    while (incoming.qty > 0 && !asks_.empty()) {
        auto it = asks_.begin();
        if (is_limit && it->first > limit_price) break;

        Level& level = it->second;
        while (level.head < level.orders.size() && level.orders[level.head].qty == 0)
            ++level.head;

        while (incoming.qty > 0 && level.head < level.orders.size()) {
            Order& maker = level.orders[level.head];
            if (maker.qty == 0) { ++level.head; continue; }

            if (incoming.agent == maker.agent && stp != StpMode::None) {
                if (stp == StpMode::CancelPassive) {
                    level.volume -= maker.qty;
                    id_index_.erase(maker.id);
                    --level.live_orders;
                    maker.qty = 0;
                    ++level.head;
                    continue;
                } else if (stp == StpMode::CancelAggressive) {
                    incoming.qty = 0;
                    break;
                } else if (stp == StpMode::DecrementAndCancel) {
                    const Qty dec = std::min(incoming.qty, maker.qty);
                    maker.qty -= dec;
                    level.volume -= dec;
                    incoming.qty -= dec;
                    if (maker.qty == 0) {
                        id_index_.erase(maker.id);
                        --level.live_orders;
                        ++level.head;
                    }
                    if (incoming.qty == 0) break;
                    continue;
                }
            }

            const Qty fill = std::min(incoming.qty, maker.qty);
            if (out.trades.empty()) out.trades.reserve(4);
            out.trades.push_back({maker.id, incoming.id,
                                  maker.agent, incoming.agent,
                                  incoming.side, it->first, fill, book_time_});
            out.filled   += fill;
            maker.qty    -= fill;
            level.volume -= fill;
            incoming.qty -= fill;

            if (maker.qty == 0) {
                id_index_.erase(maker.id);
                --level.live_orders;
                ++level.head;
            }
        }

        if (level.volume == 0) asks_.erase(it);
        else maybe_compact_level(Side::Sell, it->first, level);
    }
}

void LimitOrderBook::match_against_bids(Order& incoming, ExecutionResult& out, bool is_limit, Tick limit_price, StpMode stp) {
    while (incoming.qty > 0 && !bids_.empty()) {
        auto it = bids_.begin();
        if (is_limit && it->first < limit_price) break;

        Level& level = it->second;
        while (level.head < level.orders.size() && level.orders[level.head].qty == 0)
            ++level.head;

        while (incoming.qty > 0 && level.head < level.orders.size()) {
            Order& maker = level.orders[level.head];
            if (maker.qty == 0) { ++level.head; continue; }

            if (incoming.agent == maker.agent && stp != StpMode::None) {
                if (stp == StpMode::CancelPassive) {
                    level.volume -= maker.qty;
                    id_index_.erase(maker.id);
                    --level.live_orders;
                    maker.qty = 0;
                    ++level.head;
                    continue;
                } else if (stp == StpMode::CancelAggressive) {
                    incoming.qty = 0;
                    break;
                } else if (stp == StpMode::DecrementAndCancel) {
                    const Qty dec = std::min(incoming.qty, maker.qty);
                    maker.qty -= dec;
                    level.volume -= dec;
                    incoming.qty -= dec;
                    if (maker.qty == 0) {
                        id_index_.erase(maker.id);
                        --level.live_orders;
                        ++level.head;
                    }
                    if (incoming.qty == 0) break;
                    continue;
                }
            }

            const Qty fill = std::min(incoming.qty, maker.qty);
            if (out.trades.empty()) out.trades.reserve(4);
            out.trades.push_back({maker.id, incoming.id,
                                  maker.agent, incoming.agent,
                                  incoming.side, it->first, fill, book_time_});
            out.filled   += fill;
            maker.qty    -= fill;
            level.volume -= fill;
            incoming.qty -= fill;

            if (maker.qty == 0) {
                id_index_.erase(maker.id);
                --level.live_orders;
                ++level.head;
            }
        }

        if (level.volume == 0) bids_.erase(it);
        else maybe_compact_level(Side::Buy, it->first, level);
    }
}

void LimitOrderBook::maybe_compact_level(Side side, Tick price, Level& level) {
    if (!enable_compaction_) return;

    const std::size_t total = level.orders.size();
    const std::size_t live  = level.live_orders;
    const std::size_t dead  = total - live;
    const CompactionConfig& cfg = compaction_cfg_;

    if (total < cfg.total_min) return;
    if (dead < cfg.dead_min) return;
    if (dead * cfg.dead_den < total * cfg.dead_num) return;

    std::pmr::vector<Order> compacted(resource());
    compacted.reserve(live);

    std::uint32_t next_idx = 0;
    for (std::size_t i = level.head; i < total; ++i) {
        Order& o = level.orders[i];
        if (o.qty <= 0) continue;
        compacted.push_back(std::move(o));
        id_index_[compacted.back().id] = {side, price, 0, next_idx++};
    }

    level.orders.swap(compacted);
    level.head = 0;
}

void LimitOrderBook::rest(Order& order) {
    order.enqueue_ts = book_time_;
    if (is_buy(order.side)) {
        auto [it, inserted] = bids_.try_emplace(order.price, resource());
        Level& level = it->second;
        if (level.orders.empty()) level.orders.reserve(8);
        const std::uint32_t idx = static_cast<std::uint32_t>(level.orders.size());
        level.push(order);
        level.volume += order.qty;
        id_index_[order.id] = {Side::Buy, order.price, 0, idx};
    } else {
        auto [it, inserted] = asks_.try_emplace(order.price, resource());
        Level& level = it->second;
        if (level.orders.empty()) level.orders.reserve(8);
        const std::uint32_t idx = static_cast<std::uint32_t>(level.orders.size());
        level.push(order);
        level.volume += order.qty;
        id_index_[order.id] = {Side::Sell, order.price, 0, idx};
    }
}

bool LimitOrderBook::cancel(OrderId id) {
    const auto idx_it = id_index_.find(id);
    if (idx_it == id_index_.end()) return false;

    const OrderRef ref = idx_it->second;
    id_index_.erase(idx_it);

    auto tombstone = [&](auto& book_map) {
        const auto it = book_map.find(ref.price);
        if (it == book_map.end()) return;
        Level& level  = it->second;
        Order& o      = level.orders[ref.slot];
        level.volume -= o.qty;
        --level.live_orders;
        o.qty         = 0;
        if (ref.slot == level.head) {
            while (level.head < level.orders.size() && level.orders[level.head].qty == 0) {
                ++level.head;
            }
        }
        if (level.volume == 0) book_map.erase(it);
        else maybe_compact_level(ref.side, ref.price, level);
    };

    if (is_buy(ref.side)) tombstone(bids_);
    else                  tombstone(asks_);
    return true;
}

std::optional<Qty> LimitOrderBook::queue_ahead(OrderId id) const noexcept {
    const auto idx_it = id_index_.find(id);
    if (idx_it == id_index_.end()) return std::nullopt;
    const OrderRef ref = idx_it->second;

    auto check_ahead = [&](const auto& book_map) -> std::optional<Qty> {
        const auto it = book_map.find(ref.price);
        if (it == book_map.end()) return std::nullopt;
        const Level& level = it->second;
        if (ref.slot < level.head || ref.slot >= level.orders.size()) return std::nullopt;
        if (level.orders[ref.slot].qty == 0) return std::nullopt;
        Qty ahead = 0;
        for (std::size_t i = level.head; i < ref.slot; ++i) {
            if (level.orders[i].qty > 0) ahead += level.orders[i].qty;
        }
        return ahead;
    };

    return is_buy(ref.side) ? check_ahead(bids_) : check_ahead(asks_);
}

std::optional<Tick> LimitOrderBook::best_bid() const noexcept {
    return bids_.empty() ? std::nullopt : std::make_optional(bids_.begin()->first);
}

std::optional<Tick> LimitOrderBook::best_ask() const noexcept {
    return asks_.empty() ? std::nullopt : std::make_optional(asks_.begin()->first);
}

Qty LimitOrderBook::volume_at(Side side, Tick price) const noexcept {
    if (side == Side::Buy) {
        const auto it = bids_.find(price);
        return it != bids_.end() ? it->second.volume : 0;
    } else {
        const auto it = asks_.find(price);
        return it != asks_.end() ? it->second.volume : 0;
    }
}

std::vector<BookLevelView> LimitOrderBook::levels(Side side, std::size_t depth) const {
    std::vector<BookLevelView> out;
    auto collect = [&](const auto& book_map) {
        for (const auto& [price, level] : book_map) {
            if (depth > 0 && out.size() >= depth) break;
            if (level.empty()) continue;
            out.push_back({price, level.volume, level.live_orders});
        }
    };
    if (side == Side::Buy) collect(bids_);
    else                   collect(asks_);
    return out;
}

bool LimitOrderBook::empty() const noexcept {
    return bids_.empty() && asks_.empty();
}

} // namespace sim
