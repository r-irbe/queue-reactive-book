#include "sim/flat_ladder_book.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace sim {
namespace {
constexpr std::size_t kNpos = std::numeric_limits<std::size_t>::max();

constexpr std::size_t compute_tick_span(Tick min_tick, Tick max_tick) noexcept {
  if (max_tick < min_tick) {
    return 0;
  }
  return static_cast<std::size_t>(static_cast<std::int64_t>(max_tick) -
                                  static_cast<std::int64_t>(min_tick) + 1);
}
} // namespace

FlatLadderBook::FlatLadderBook(Tick min_tick, Tick max_tick,
                               std::size_t expected_orders,
                               bool /*enable_compaction*/,
                               CompactionConfig /*compaction_cfg*/,
                               std::size_t arena_bytes) noexcept
    : arena_(arena_bytes == 0
                 ? suggest_arena_bytes(expected_orders +
                                       compute_tick_span(min_tick, max_tick))
                 : arena_bytes,
             std::pmr::new_delete_resource()),
      pool_(&arena_), levels_(resource()), node_pool_(resource()),
      level_side_(resource()), level_bitmap_(resource()),
      summary_bitmap_(resource()), id_refs_(resource()), id_live_(resource()),
      min_tick_(min_tick), max_tick_(max_tick),
      tick_count_(compute_tick_span(min_tick, max_tick)),
      is_valid_(max_tick >= min_tick) {
  if (!is_valid_) {
    return;
  }
  levels_.resize(tick_count_);
  level_side_.assign(tick_count_, kNoSide);
  const std::size_t num_level_words = (tick_count_ + 63) / 64;
  level_bitmap_.assign(num_level_words, 0);
  summary_bitmap_.assign((num_level_words + 63) / 64, 0);

  node_pool_.reserve(expected_orders);
  id_refs_.resize(expected_orders + 1);
  id_live_.assign(expected_orders + 1, 0);
}

std::unique_ptr<FlatLadderBook> FlatLadderBook::create(Tick min_tick,
                                                       Tick max_tick,
                                                       std::size_t expected_orders,
                                                       std::size_t arena_bytes) noexcept {
  if (max_tick < min_tick) return nullptr;
  return std::unique_ptr<FlatLadderBook>(new (std::nothrow) FlatLadderBook(
      min_tick, max_tick, expected_orders, false, {}, arena_bytes));
}

bool FlatLadderBook::has_order_ref(OrderId id) const noexcept {
  const auto idx = static_cast<std::size_t>(id);
  return idx < id_live_.size() && id_live_[idx] != 0;
}

void FlatLadderBook::set_order_ref(OrderId id, const OrderRef &ref) {
  const auto idx = static_cast<std::size_t>(id);
  if (idx >= id_refs_.size()) [[unlikely]] {
    const std::size_t next_cap = std::max(idx + 1, id_refs_.size() * 2);
    id_refs_.resize(next_cap);
    id_live_.resize(next_cap, 0);
  }
  id_refs_[idx] = ref;
  if (id_live_[idx] == 0) {
    id_live_[idx] = 1;
    ++live_order_count_;
  }
}

void FlatLadderBook::clear_order_ref(OrderId id) noexcept {
  const auto idx = static_cast<std::size_t>(id);
  if (idx < id_live_.size() && id_live_[idx] != 0) {
    id_live_[idx] = 0;
    --live_order_count_;
  }
}

OrderRef FlatLadderBook::get_order_ref(OrderId id) const noexcept {
  return id_refs_[static_cast<std::size_t>(id)];
}

bool FlatLadderBook::in_range(Tick price) const noexcept {
  return (tick_count_ > 0) && (price >= min_tick_) && (price <= max_tick_);
}

std::size_t FlatLadderBook::index_of(Tick price) const noexcept {
  return static_cast<std::size_t>(static_cast<std::int64_t>(price) -
                                  static_cast<std::int64_t>(min_tick_));
}

Tick FlatLadderBook::price_of(std::size_t idx) const noexcept {
  return static_cast<Tick>(static_cast<std::int64_t>(min_tick_) +
                           static_cast<std::int64_t>(idx));
}

IntrusiveLevel &FlatLadderBook::level_at(std::size_t idx) noexcept {
  return levels_[idx];
}
const IntrusiveLevel &FlatLadderBook::level_at(std::size_t idx) const noexcept {
  return levels_[idx];
}

std::uint32_t FlatLadderBook::alloc_slot() {
  if (free_head_ != kNullSlot) {
    const std::uint32_t slot = free_head_;
    free_head_ = node_pool_[slot].next_slot;
    node_pool_[slot].prev_slot = kNullSlot;
    node_pool_[slot].next_slot = kNullSlot;
    return slot;
  }
  const auto slot = static_cast<std::uint32_t>(node_pool_.size());
  node_pool_.emplace_back();
  return slot;
}

void FlatLadderBook::free_slot(std::uint32_t slot) noexcept {
  node_pool_[slot].id = 0;
  node_pool_[slot].qty = 0;
  node_pool_[slot].prev_slot = kNullSlot;
  node_pool_[slot].next_slot = free_head_;
  free_head_ = slot;
}

bool FlatLadderBook::can_fill_fok(Side side, Tick price, Qty qty,
                                  bool is_market, AgentId taker_agent,
                                  StpMode stp) const noexcept {
  if (!is_valid_ || qty <= 0) {
    return is_valid_ && qty == 0;
  }
  Qty available = 0;

  auto accumulate_level = [&](std::size_t idx, bool &aborted) noexcept -> bool {
    if (stp == StpMode::None) {
      available += levels_[idx].volume;
      return available >= qty;
    }
    for (std::uint32_t curr = levels_[idx].head_slot; curr != kNullSlot;
         curr = node_pool_[curr].next_slot) {
      const IntrusiveOrderNode &maker = node_pool_[curr];
      if (maker.qty <= 0) continue;
      if (maker.agent == taker_agent) {
        if (stp == StpMode::CancelAggressive ||
            stp == StpMode::DecrementAndCancel) {
          aborted = true;
          return available >= qty;
        }
        continue;
      }
      available += maker.qty;
      if (available >= qty) return true;
    }
    return false;
  };

  if (is_buy(side)) {
    for (std::size_t idx = best_ask_idx_; idx != kNpos && idx < tick_count_;) {
      if (!is_market && price_of(idx) > price) break;
      bool aborted = false;
      if (accumulate_level(idx, aborted)) return true;
      if (aborted) return false;

      const std::size_t next_idx = idx + 1;
      if (next_idx >= tick_count_) break;
      std::size_t word = next_idx >> 6;
      const std::size_t bit = next_idx & 63;
      const std::uint64_t mask = (~std::uint64_t{0}) << bit;
      std::uint64_t w = level_bitmap_[word] & mask;
      std::size_t found = kNpos;
      if (w != 0) {
        found = (word << 6) + static_cast<std::size_t>(__builtin_ctzll(w));
        if (found >= tick_count_) found = kNpos;
      } else {
        const std::size_t nw = word + 1;
        if (nw < level_bitmap_.size()) {
          const std::size_t sw = nw >> 6;
          const std::size_t sb = nw & 63;
          const std::uint64_t s_mask = (~std::uint64_t{0}) << sb;
          for (std::size_t s = sw; s < summary_bitmap_.size(); ++s) {
            std::uint64_t s_val = summary_bitmap_[s];
            if (s == sw) s_val &= s_mask;
            if (s_val != 0) {
              const std::size_t tw =
                  (s << 6) + static_cast<std::size_t>(__builtin_ctzll(s_val));
              if (tw < level_bitmap_.size()) {
                found = (tw << 6) +
                        static_cast<std::size_t>(__builtin_ctzll(level_bitmap_[tw]));
                if (found >= tick_count_) found = kNpos;
                break;
              }
            }
          }
        }
      }
      idx = found;
    }
  } else {
    for (std::size_t idx = best_bid_idx_; idx != kNpos;) {
      if (!is_market && price_of(idx) < price) break;
      bool aborted = false;
      if (accumulate_level(idx, aborted)) return true;
      if (aborted) return false;

      if (idx == 0) break;
      const std::size_t prev_idx = idx - 1;
      std::size_t word = prev_idx >> 6;
      const std::size_t bit = prev_idx & 63;
      const std::uint64_t mask = (~std::uint64_t{0}) >> (63 - bit);
      std::uint64_t w = level_bitmap_[word] & mask;
      std::size_t found = kNpos;
      if (w != 0) {
        found = (word << 6) + static_cast<std::size_t>(63 - __builtin_clzll(w));
      } else if (word > 0) {
        const std::size_t pw_idx = word - 1;
        const std::size_t sw = pw_idx >> 6;
        const std::size_t sb = pw_idx & 63;
        const std::uint64_t s_mask = (~std::uint64_t{0}) >> (63 - sb);
        for (std::int64_t s = static_cast<std::int64_t>(sw); s >= 0; --s) {
          std::uint64_t s_val = summary_bitmap_[static_cast<std::size_t>(s)];
          if (static_cast<std::size_t>(s) == sw) s_val &= s_mask;
          if (s_val != 0) {
            const std::size_t tw =
                (static_cast<std::size_t>(s) << 6) +
                static_cast<std::size_t>(63 - __builtin_clzll(s_val));
            if (tw < level_bitmap_.size()) {
              found = (tw << 6) +
                      static_cast<std::size_t>(63 - __builtin_clzll(level_bitmap_[tw]));
              break;
            }
          }
        }
      }
      idx = found;
    }
  }
  return available >= qty;
}

ExecutionResult FlatLadderBook::add(Order order, TimeNs ts, StpMode stp) {
  ExecutionResult result;
  if (!is_valid_ || order.qty <= 0) [[unlikely]] {
    return result;
  }

  book_time_ = ts;
  order.id = next_id_++;
  order.orig_qty = order.qty;
  const StpMode effective_stp =
      (stp != StpMode::None)
          ? stp
          : (order.stp != StpMode::None ? order.stp : default_stp_mode_);

  if (order.type == OrderType::PostOnly) {
    if (is_buy(order.side)) {
      if (best_ask_idx_ != kNpos && price_of(best_ask_idx_) <= order.price) {
        ++event_count_;
        return result;
      }
    } else {
      if (best_bid_idx_ != kNpos && price_of(best_bid_idx_) >= order.price) {
        ++event_count_;
        return result;
      }
    }
    if (!in_range(order.price)) {
      ++event_count_;
      return result;
    }
    rest(order);
    result.remaining = order.qty;
    ++event_count_;
    return result;
  }

  if (order.type == OrderType::FOK) {
    if (!can_fill_fok(order.side, order.price, order.qty, false, order.agent,
                      effective_stp)) {
      ++event_count_;
      return result;
    }
  }

  const bool is_limit = (order.type == OrderType::Limit ||
                         order.type == OrderType::IOC ||
                         order.type == OrderType::FOK);

  auto record_trade = [&](const Trade &tr) {
    if (result.trades.empty()) {
      result.trades.reserve(4);
    }
    result.trades.push_back(tr);
    result.filled += tr.qty;
  };

  if (is_buy(order.side)) {
    if (is_limit)
      match_incoming<Side::Buy, true>(order, record_trade, order.price, ts,
                                      effective_stp);
    else
      match_incoming<Side::Buy, false>(order, record_trade, order.price, ts,
                                       effective_stp);
  } else {
    if (is_limit)
      match_incoming<Side::Sell, true>(order, record_trade, order.price, ts,
                                       effective_stp);
    else
      match_incoming<Side::Sell, false>(order, record_trade, order.price, ts,
                                        effective_stp);
  }

  if (order.type == OrderType::Limit && order.qty > 0 && in_range(order.price)) {
    rest(order);
  }

  result.remaining = order.qty;
  ++event_count_;
  return result;
}

MatchSummary FlatLadderBook::add_into(Order order, TimeNs ts, Trade *trade_buf,
                                      std::size_t trade_cap,
                                      StpMode stp) noexcept {
  MatchSummary summary{};
  if (!is_valid_ || order.qty <= 0) [[unlikely]] {
    return summary;
  }

  book_time_ = ts;
  order.id = next_id_++;
  order.orig_qty = order.qty;
  summary.id = order.id;
  const StpMode effective_stp =
      (stp != StpMode::None)
          ? stp
          : (order.stp != StpMode::None ? order.stp : default_stp_mode_);

  if (order.type == OrderType::PostOnly) {
    if (is_buy(order.side)) {
      if (best_ask_idx_ != kNpos && price_of(best_ask_idx_) <= order.price) {
        ++event_count_;
        return summary;
      }
    } else {
      if (best_bid_idx_ != kNpos && price_of(best_bid_idx_) >= order.price) {
        ++event_count_;
        return summary;
      }
    }
    if (!in_range(order.price)) {
      ++event_count_;
      return summary;
    }
    rest(order);
    summary.remaining = order.qty;
    ++event_count_;
    return summary;
  }

  if (order.type == OrderType::FOK) {
    if (!can_fill_fok(order.side, order.price, order.qty, false, order.agent,
                      effective_stp)) {
      ++event_count_;
      return summary;
    }
  }

  const bool is_limit = (order.type == OrderType::Limit ||
                         order.type == OrderType::IOC ||
                         order.type == OrderType::FOK);

  auto record_trade = [&](const Trade &tr) noexcept {
    if (trade_buf != nullptr && summary.trade_count < trade_cap) {
      trade_buf[summary.trade_count] = tr;
    }
    ++summary.trade_count;
    summary.filled += tr.qty;
  };

  if (is_buy(order.side)) {
    if (is_limit)
      match_incoming<Side::Buy, true>(order, record_trade, order.price, ts,
                                      effective_stp);
    else
      match_incoming<Side::Buy, false>(order, record_trade, order.price, ts,
                                       effective_stp);
  } else {
    if (is_limit)
      match_incoming<Side::Sell, true>(order, record_trade, order.price, ts,
                                       effective_stp);
    else
      match_incoming<Side::Sell, false>(order, record_trade, order.price, ts,
                                        effective_stp);
  }

  if (order.type == OrderType::Limit && order.qty > 0 && in_range(order.price)) {
    rest(order);
  }

  summary.remaining = order.qty;
  ++event_count_;
  return summary;
}

template <Side TakerSide, bool IsLimit, typename TradeSink>
[[gnu::hot]]
void FlatLadderBook::match_incoming(Order &incoming, TradeSink &&on_trade,
                                    Tick limit_price, TimeNs ts, StpMode stp) {
  constexpr bool kBuy = (TakerSide == Side::Buy);

  while (incoming.qty > 0) {
    const std::size_t idx = kBuy ? best_ask_idx_ : best_bid_idx_;
    if (idx == kNpos) [[unlikely]] break;

    const Tick maker_price = price_of(idx);
    if constexpr (IsLimit) {
      if constexpr (kBuy) {
        if (maker_price > limit_price) break;
      } else {
        if (maker_price < limit_price) break;
      }
    }

    IntrusiveLevel &level = level_at(idx);

    while (incoming.qty > 0 && level.head_slot != kNullSlot) {
      const std::uint32_t slot = level.head_slot;
      IntrusiveOrderNode &maker = node_pool_[slot];

      if (incoming.agent == maker.agent && stp != StpMode::None) {
        if (stp == StpMode::CancelPassive) {
          const Qty c_qty = maker.qty;
          level.volume -= c_qty;
          constexpr std::uint8_t kOppositeSide = kBuy ? 1 : 0;
          total_vol_[kOppositeSide] -= c_qty;

          clear_order_ref(maker.id);
          --level.live_orders;

          level.head_slot = maker.next_slot;
          if (level.head_slot != kNullSlot) {
            node_pool_[level.head_slot].prev_slot = kNullSlot;
          } else {
            level.tail_slot = kNullSlot;
          }
          free_slot(slot);
          continue;
        } else if (stp == StpMode::CancelAggressive) {
          incoming.qty = 0;
          break;
        } else if (stp == StpMode::DecrementAndCancel) {
          const Qty dec = std::min(incoming.qty, maker.qty);
          maker.qty -= dec;
          level.volume -= dec;
          incoming.qty -= dec;
          constexpr std::uint8_t kOppositeSide = kBuy ? 1 : 0;
          total_vol_[kOppositeSide] -= dec;

          if (maker.qty == 0) {
            clear_order_ref(maker.id);
            --level.live_orders;
            level.head_slot = maker.next_slot;
            if (level.head_slot != kNullSlot) {
              node_pool_[level.head_slot].prev_slot = kNullSlot;
            } else {
              level.tail_slot = kNullSlot;
            }
            free_slot(slot);
          }
          if (incoming.qty == 0) break;
          continue;
        }
      }

      const Qty fill = std::min(incoming.qty, maker.qty);
      on_trade(Trade{.maker_id   = maker.id,
                     .taker_id   = incoming.id,
                     .maker      = maker.agent,
                     .taker      = incoming.agent,
                     .taker_side = incoming.side,
                     .price      = maker_price,
                     .qty        = fill,
                     .ts         = ts});

      maker.qty -= fill;
      level.volume -= fill;
      incoming.qty -= fill;
      constexpr std::uint8_t kOppositeSide = kBuy ? 1 : 0;
      total_vol_[kOppositeSide] -= fill;

      if (maker.qty == 0) {
        clear_order_ref(maker.id);
        --level.live_orders;
        level.head_slot = maker.next_slot;
        if (level.head_slot != kNullSlot) {
          node_pool_[level.head_slot].prev_slot = kNullSlot;
        } else {
          level.tail_slot = kNullSlot;
        }
        free_slot(slot);
      }
    }

    if (level.volume == 0) {
      level.head_slot = kNullSlot;
      level.tail_slot = kNullSlot;
      level_side_[idx] = kNoSide;
      clear_level_occupied(idx);
      if constexpr (kBuy) {
        refresh_best_ask_from(idx + 1);
      } else {
        refresh_best_bid_from(idx == 0 ? kNpos : idx - 1);
      }
    }
  }
}

void FlatLadderBook::rest(Order &order) {
  order.enqueue_ts = book_time_;
  const std::size_t idx = index_of(order.price);
  IntrusiveLevel &level = level_at(idx);

  const std::uint32_t slot = alloc_slot();
  IntrusiveOrderNode &node = node_pool_[slot];
  node.id = order.id;
  node.enqueue_ts = order.enqueue_ts;
  node.qty = order.qty;
  node.agent = order.agent;
  node.prev_slot = level.tail_slot;
  node.next_slot = kNullSlot;

  if (level.tail_slot != kNullSlot) {
    node_pool_[level.tail_slot].next_slot = slot;
  } else {
    level.head_slot = slot;
  }
  level.tail_slot = slot;

  ++level.live_orders;
  level.volume += order.qty;
  level_side_[idx] = side_tag(order.side);
  mark_level_occupied(idx);

  set_order_ref(order.id, {order.side, 0, static_cast<std::uint32_t>(idx), slot});

  total_vol_[static_cast<std::uint8_t>(order.side) & 1] += order.qty;
  if (is_buy(order.side)) {
    if (best_bid_idx_ == kNpos || idx > best_bid_idx_)
      best_bid_idx_ = idx;
  } else {
    if (best_ask_idx_ == kNpos || idx < best_ask_idx_)
      best_ask_idx_ = idx;
  }
}

bool FlatLadderBook::cancel(OrderId id, TimeNs ts) noexcept {
  book_time_ = ts;
  if (!has_order_ref(id)) [[unlikely]]
    return false;

  const OrderRef ref = get_order_ref(id);
  clear_order_ref(id);

  const std::size_t level_idx = ref.level_idx;
  const std::uint32_t slot = ref.slot;
  IntrusiveLevel &level = level_at(level_idx);
  IntrusiveOrderNode &node = node_pool_[slot];

  const Qty cancelled_qty = node.qty;
  level.volume -= cancelled_qty;
  total_vol_[static_cast<std::uint8_t>(ref.side) & 1] -= cancelled_qty;
  --level.live_orders;

  if (node.prev_slot != kNullSlot) {
    node_pool_[node.prev_slot].next_slot = node.next_slot;
  } else {
    level.head_slot = node.next_slot;
  }

  if (node.next_slot != kNullSlot) {
    node_pool_[node.next_slot].prev_slot = node.prev_slot;
  } else {
    level.tail_slot = node.prev_slot;
  }

  free_slot(slot);

  if (level.volume == 0) {
    level.head_slot = kNullSlot;
    level.tail_slot = kNullSlot;
    level_side_[level_idx] = kNoSide;
    clear_level_occupied(level_idx);
    if (is_buy(ref.side) && best_bid_idx_ == level_idx)
      refresh_best_bid_from(level_idx == 0 ? kNpos : level_idx - 1);
    if (is_sell(ref.side) && best_ask_idx_ == level_idx)
      refresh_best_ask_from(level_idx + 1);
  }

  ++event_count_;
  return true;
}

std::optional<Qty> FlatLadderBook::queue_ahead(OrderId id) const noexcept {
  if (!has_order_ref(id)) return std::nullopt;
  const OrderRef ref = get_order_ref(id);
  if (ref.level_idx >= tick_count_) return std::nullopt;
  const IntrusiveLevel &lvl = levels_[ref.level_idx];
  const std::uint32_t target_slot = ref.slot;
  if (target_slot >= node_pool_.size()) return std::nullopt;
  if (node_pool_[target_slot].qty == 0) return std::nullopt;

  Qty ahead = 0;
  for (std::uint32_t curr = lvl.head_slot;
       curr != kNullSlot && curr != target_slot;
       curr = node_pool_[curr].next_slot) {
    ahead += node_pool_[curr].qty;
  }
  return ahead;
}

bool FlatLadderBook::best_bid(Tick &price, Qty &volume) const noexcept {
  if (best_bid_idx_ == kNpos || tick_count_ == 0) {
    price = min_tick_;
    volume = 0;
    return false;
  }
  price = price_of(best_bid_idx_);
  volume = levels_[best_bid_idx_].volume;
  return true;
}

bool FlatLadderBook::best_ask(Tick &price, Qty &volume) const noexcept {
  if (best_ask_idx_ == kNpos || tick_count_ == 0) {
    price = min_tick_;
    volume = 0;
    return false;
  }
  price = price_of(best_ask_idx_);
  volume = levels_[best_ask_idx_].volume;
  return true;
}

bool FlatLadderBook::mid_tick_x2(Tick &sum) const noexcept {
  if (best_bid_idx_ == kNpos || best_ask_idx_ == kNpos || tick_count_ == 0) {
    sum = 0;
    return false;
  }
  sum = static_cast<Tick>(price_of(best_bid_idx_) + price_of(best_ask_idx_));
  return true;
}

Qty FlatLadderBook::volume_at(Side side, Tick price) const noexcept {
  if (!in_range(price)) {
    return 0;
  }
  const std::size_t idx = index_of(price);
  return level_side_[idx] == side_tag(side) ? levels_[idx].volume : 0;
}

bool FlatLadderBook::mid_price(double &price) const noexcept {
  Tick bb = 0;
  Tick ba = 0;
  Qty  bv = 0;
  Qty  av = 0;
  if (!best_bid(bb, bv) || !best_ask(ba, av)) return false;
  price = 0.5 * (static_cast<double>(bb) + static_cast<double>(ba));
  return true;
}

std::vector<BookLevelView> FlatLadderBook::levels(Side side, std::size_t depth) const {
  std::vector<BookLevelView> out;
  if (side == Side::Buy) {
    const std::uint8_t tag = kBuySide;
    for (std::size_t i = best_bid_idx_; i != kNpos;) {
      const IntrusiveLevel &level = level_at(i);
      if (!level.empty() && level_side_[i] == tag)
        out.push_back({price_of(i), level.volume, level.live_orders});
      if (depth > 0 && out.size() >= depth)
        break;
      if (i == 0)
        break;
      --i;
    }
  } else {
    const std::uint8_t tag = kSellSide;
    for (std::size_t i = best_ask_idx_; i != kNpos && i < tick_count_; ++i) {
      const IntrusiveLevel &level = level_at(i);
      if (!level.empty() && level_side_[i] == tag)
        out.push_back({price_of(i), level.volume, level.live_orders});
      if (depth > 0 && out.size() >= depth)
        break;
    }
  }
  return out;
}

bool FlatLadderBook::is_empty() const noexcept {
  return best_bid_idx_ == kNpos && best_ask_idx_ == kNpos;
}

InvariantReport FlatLadderBook::check_invariants_report() const noexcept {
  Qty bid_sum = 0;
  Qty ask_sum = 0;
  std::size_t live_sum = 0;
  std::size_t observed_best_bid = kNpos;
  std::size_t observed_best_ask = kNpos;

  for (std::size_t i = 0; i < tick_count_; ++i) {
    const IntrusiveLevel &level = level_at(i);
    Qty qty_sum = 0;
    std::uint32_t live = 0;

    for (std::uint32_t curr = level.head_slot; curr != kNullSlot; curr = node_pool_[curr].next_slot) {
      const IntrusiveOrderNode &node = node_pool_[curr];
      if (node.qty <= 0) {
        return InvariantReport{false, "flat: intrusive node with non-positive qty found (tombstone violation)"};
      }
      qty_sum += node.qty;
      ++live;
      if (live > live_order_count_ + 1) {
        return InvariantReport{false, "flat: cycle detected in intrusive slotmap"};
      }
    }

    if (qty_sum != level.volume)
      return InvariantReport{false, "flat: level volume mismatch"};
    if (live != level.live_orders)
      return InvariantReport{false, "flat: level live_orders mismatch"};

    const std::uint8_t tag = level_side_[i];
    const bool bit_set = (level_bitmap_[i / 64] & (std::uint64_t{1} << (i % 64))) != 0;
    if (qty_sum == 0) {
      if (tag != kNoSide)
        return InvariantReport{false, "flat: empty level with side tag set"};
      if (bit_set)
        return InvariantReport{false, "flat: empty level with bitmap bit set"};
      continue;
    }
    if (!bit_set)
      return InvariantReport{false, "flat: non-empty level missing bitmap bit"};
    if (tag == kBuySide) {
      bid_sum += qty_sum;
      observed_best_bid = i;
    } else if (tag == kSellSide) {
      ask_sum += qty_sum;
      if (observed_best_ask == kNpos) observed_best_ask = i;
    } else {
      return InvariantReport{false, "flat: non-empty level missing side tag"};
    }
    live_sum += live;
  }

  if (bid_sum != total_vol_[0])
    return InvariantReport{false, "flat: total bid volume mismatch"};
  if (ask_sum != total_vol_[1])
    return InvariantReport{false, "flat: total ask volume mismatch"};
  if (live_sum != live_order_count_)
    return InvariantReport{false, "flat: live order count mismatch"};
  if (observed_best_bid != best_bid_idx_)
    return InvariantReport{false, "flat: best_bid_idx out of sync"};
  if (observed_best_ask != best_ask_idx_)
    return InvariantReport{false, "flat: best_ask_idx out of sync"};
  if (best_bid_idx_ != kNpos && best_ask_idx_ != kNpos &&
      best_bid_idx_ >= best_ask_idx_)
    return InvariantReport{false, "flat: crossed book"};

  return InvariantReport{true, "ok"};
}

void FlatLadderBook::check_invariants() const noexcept {
  const auto report = check_invariants_report();
  (void)report;
  assert(report.ok && report.message);
}

void FlatLadderBook::refresh_best_bid_from(std::size_t start) noexcept {
  if (tick_count_ == 0 || start == kNpos) {
    best_bid_idx_ = kNpos;
    return;
  }
  const std::size_t clamped = std::min(start, tick_count_ - 1);
  std::size_t word = clamped >> 6;
  const std::size_t bit_in_word = clamped & 63;
  const std::uint64_t head_mask = (~std::uint64_t{0}) >> (63 - bit_in_word);
  std::uint64_t w = level_bitmap_[word] & head_mask;
  if (w != 0) {
    const int hi = 63 - __builtin_clzll(w);
    best_bid_idx_ = (word << 6) + static_cast<std::size_t>(hi);
    return;
  }

  if (word == 0) {
    best_bid_idx_ = kNpos;
    return;
  }
  const std::size_t target_word = word - 1;
  const std::size_t s_word = target_word >> 6;
  const std::size_t s_bit = target_word & 63;
  const std::uint64_t s_head_mask = (~std::uint64_t{0}) >> (63 - s_bit);

  for (std::int64_t sw = static_cast<std::int64_t>(s_word); sw >= 0; --sw) {
    std::uint64_t sw_val = summary_bitmap_[static_cast<std::size_t>(sw)];
    if (static_cast<std::size_t>(sw) == s_word) {
      sw_val &= s_head_mask;
    }
    if (sw_val != 0) {
      const int hi_w = 63 - __builtin_clzll(sw_val);
      const std::size_t prev_word = (static_cast<std::size_t>(sw) << 6) + static_cast<std::size_t>(hi_w);
      const std::uint64_t target_w = level_bitmap_[prev_word];
      const int hi = 63 - __builtin_clzll(target_w);
      best_bid_idx_ = (prev_word << 6) + static_cast<std::size_t>(hi);
      return;
    }
  }
  best_bid_idx_ = kNpos;
}

void FlatLadderBook::refresh_best_ask_from(std::size_t start) noexcept {
  if (start >= tick_count_) {
    best_ask_idx_ = kNpos;
    return;
  }
  std::size_t word = start >> 6;
  const std::size_t bit_in_word = start & 63;
  const std::uint64_t tail_mask = (~std::uint64_t{0}) << bit_in_word;
  std::uint64_t w = level_bitmap_[word] & tail_mask;
  if (w != 0) {
    const int lo = __builtin_ctzll(w);
    best_ask_idx_ = (word << 6) + static_cast<std::size_t>(lo);
    if (best_ask_idx_ >= tick_count_) [[unlikely]] {
      best_ask_idx_ = kNpos;
    }
    return;
  }

  const std::size_t next_word = word + 1;
  if (next_word >= level_bitmap_.size()) {
    best_ask_idx_ = kNpos;
    return;
  }
  const std::size_t s_word = next_word >> 6;
  const std::size_t s_bit = next_word & 63;
  const std::uint64_t s_tail_mask = (~std::uint64_t{0}) << s_bit;
  const std::size_t num_s_words = summary_bitmap_.size();

  for (std::size_t sw = s_word; sw < num_s_words; ++sw) {
    std::uint64_t sw_val = summary_bitmap_[sw];
    if (sw == s_word) {
      sw_val &= s_tail_mask;
    }
    if (sw_val != 0) {
      const int lo_w = __builtin_ctzll(sw_val);
      const std::size_t target_w_idx = (sw << 6) + static_cast<std::size_t>(lo_w);
      if (target_w_idx < level_bitmap_.size()) {
        const std::uint64_t target_w = level_bitmap_[target_w_idx];
        const int lo = __builtin_ctzll(target_w);
        best_ask_idx_ = (target_w_idx << 6) + static_cast<std::size_t>(lo);
        if (best_ask_idx_ >= tick_count_) [[unlikely]] {
          best_ask_idx_ = kNpos;
        }
        return;
      }
    }
  }
  best_ask_idx_ = kNpos;
}

void FlatLadderBook::mark_level_occupied(std::size_t idx) noexcept {
  const std::size_t word = idx >> 6;
  const std::uint64_t prev = level_bitmap_[word];
  level_bitmap_[word] = prev | (std::uint64_t{1} << (idx & 63));
  if (prev == 0) [[unlikely]] {
    summary_bitmap_[word >> 6] |= (std::uint64_t{1} << (word & 63));
  }
}

void FlatLadderBook::clear_level_occupied(std::size_t idx) noexcept {
  const std::size_t word = idx >> 6;
  level_bitmap_[word] &= ~(std::uint64_t{1} << (idx & 63));
  if (level_bitmap_[word] == 0) [[unlikely]] {
    summary_bitmap_[word >> 6] &= ~(std::uint64_t{1} << (word & 63));
  }
}

} // namespace sim
