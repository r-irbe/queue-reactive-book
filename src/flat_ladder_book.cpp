// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#include "sim/flat_ladder_book.hpp"
#include <algorithm>
#include <bit>
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

inline std::uint64_t fnv1a_mix64(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v;
  h *= 0x100000001B3ULL;
  return h;
}
} // namespace

FlatLadderBook::FlatLadderBook(Tick min_tick, Tick max_tick,
                               std::size_t expected_orders,
                               bool /*enable_compaction*/,
                               CompactionConfig /*compaction_cfg*/,
                               std::size_t arena_bytes) noexcept
    : arena_(arena_bytes == 0
                 ? suggest_arena_bytes(expected_orders +
                                       compute_tick_span(min_tick, max_tick) +
                                       16384)
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

  const std::size_t prealloc_orders = std::max<std::size_t>(expected_orders + 4096, 8192);
  node_pool_.reserve(prealloc_orders);
  id_refs_.resize(prealloc_orders + 1);
  id_live_.assign(prealloc_orders + 1, 0);
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

std::size_t FlatLadderBook::find_next_occupied_from(std::size_t start) const noexcept {
  if (start >= tick_count_) {
    return kNpos;
  }
  const std::size_t word = start >> 6;
  const std::size_t bit_in_word = start & 63;
  const std::uint64_t tail_mask = (~std::uint64_t{0}) << bit_in_word;
  const std::uint64_t w = level_bitmap_[word] & tail_mask;
  if (w != 0) {
    const std::size_t idx = (word << 6) + static_cast<std::size_t>(std::countr_zero(w));
    return (idx < tick_count_) ? idx : kNpos;
  }

  const std::size_t next_word = word + 1;
  if (next_word >= level_bitmap_.size()) {
    return kNpos;
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
      const std::size_t target_w_idx =
          (sw << 6) + static_cast<std::size_t>(std::countr_zero(sw_val));
      if (target_w_idx < level_bitmap_.size()) {
        const std::uint64_t target_w = level_bitmap_[target_w_idx];
        const std::size_t idx =
            (target_w_idx << 6) + static_cast<std::size_t>(std::countr_zero(target_w));
        return (idx < tick_count_) ? idx : kNpos;
      }
      break;
    }
  }
  return kNpos;
}

std::size_t FlatLadderBook::find_prev_occupied_from(std::size_t start) const noexcept {
  if (tick_count_ == 0 || start == kNpos) {
    return kNpos;
  }
  const std::size_t clamped = std::min(start, tick_count_ - 1);
  const std::size_t word = clamped >> 6;
  const std::size_t bit_in_word = clamped & 63;
  const std::uint64_t head_mask = (~std::uint64_t{0}) >> (63 - bit_in_word);
  const std::uint64_t w = level_bitmap_[word] & head_mask;
  if (w != 0) {
    const int hi = 63 - std::countl_zero(w);
    return (word << 6) + static_cast<std::size_t>(hi);
  }

  if (word == 0) {
    return kNpos;
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
      const int hi_w = 63 - std::countl_zero(sw_val);
      const std::size_t prev_word =
          (static_cast<std::size_t>(sw) << 6) + static_cast<std::size_t>(hi_w);
      if (prev_word < level_bitmap_.size()) {
        const std::uint64_t target_w = level_bitmap_[prev_word];
        const int hi = 63 - std::countl_zero(target_w);
        return (prev_word << 6) + static_cast<std::size_t>(hi);
      }
      break;
    }
  }
  return kNpos;
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
    for (std::size_t idx = best_ask_idx_; idx != kNpos && idx < tick_count_;
         idx = find_next_occupied_from(idx + 1)) {
      if (!is_market && price_of(idx) > price) break;
      bool aborted = false;
      if (accumulate_level(idx, aborted)) return true;
      if (aborted) return false;
    }
  } else {
    for (std::size_t idx = best_bid_idx_; idx != kNpos;
         idx = (idx == 0 ? kNpos : find_prev_occupied_from(idx - 1))) {
      if (!is_market && price_of(idx) < price) break;
      bool aborted = false;
      if (accumulate_level(idx, aborted)) return true;
      if (aborted) return false;
    }
  }
  return available >= qty;
}

ExecutionResult FlatLadderBook::add(Order order, TimeNs ts, StpMode stp) {
  ExecutionResult result;
  if (!is_valid_ || order.qty <= 0) [[unlikely]] {
    result.status = OrderStatus::RejectedInvalid;
    return result;
  }

  book_time_ = ts;
  order.id = next_id_++;
  order.orig_qty = order.qty;
  result.id = order.id;
  const StpMode effective_stp =
      (stp != StpMode::None)
          ? stp
          : (order.stp != StpMode::None ? order.stp : default_stp_mode_);

  if (order.type == OrderType::PostOnly) {
    if (is_buy(order.side)) {
      if (best_ask_idx_ != kNpos && price_of(best_ask_idx_) <= order.price) {
        result.status = OrderStatus::RejectedPostOnly;
        ++event_count_;
        return result;
      }
    } else {
      if (best_bid_idx_ != kNpos && price_of(best_bid_idx_) >= order.price) {
        result.status = OrderStatus::RejectedPostOnly;
        ++event_count_;
        return result;
      }
    }
    if (!in_range(order.price)) {
      result.status = OrderStatus::RejectedOutOfRange;
      ++event_count_;
      return result;
    }
    if (active_undo_log_ == nullptr) [[likely]] {
      rest_impl<false>(order);
    } else {
      rest_impl<true>(order);
    }
    result.remaining = order.qty;
    result.status = OrderStatus::AcceptedRested;
    ++event_count_;
    return result;
  }

  if (order.type == OrderType::FOK) {
    if (!can_fill_fok(order.side, order.price, order.qty, false, order.agent,
                      effective_stp)) {
      result.status = OrderStatus::RejectedFok;
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

  std::uint8_t stp_cancels = 0;
  bool stp_cancelled_taker = false;
  if (active_undo_log_ == nullptr) [[likely]] {
    if (is_buy(order.side)) {
      if (is_limit)
        match_incoming<Side::Buy, true, false>(order, record_trade, order.price, ts,
                                               effective_stp, stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Buy, false, false>(order, record_trade, order.price, ts,
                                                effective_stp, stp_cancels, stp_cancelled_taker);
    } else {
      if (is_limit)
        match_incoming<Side::Sell, true, false>(order, record_trade, order.price, ts,
                                                effective_stp, stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Sell, false, false>(order, record_trade, order.price, ts,
                                                 effective_stp, stp_cancels, stp_cancelled_taker);
    }
  } else {
    if (is_buy(order.side)) {
      if (is_limit)
        match_incoming<Side::Buy, true, true>(order, record_trade, order.price, ts,
                                              effective_stp, stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Buy, false, true>(order, record_trade, order.price, ts,
                                               effective_stp, stp_cancels, stp_cancelled_taker);
    } else {
      if (is_limit)
        match_incoming<Side::Sell, true, true>(order, record_trade, order.price, ts,
                                               effective_stp, stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Sell, false, true>(order, record_trade, order.price, ts,
                                                effective_stp, stp_cancels, stp_cancelled_taker);
    }
  }

  if (order.type == OrderType::Limit && order.qty > 0) {
    if (in_range(order.price)) {
      if (active_undo_log_ == nullptr) [[likely]] {
        rest_impl<false>(order);
      } else {
        rest_impl<true>(order);
      }
      result.status = (result.filled > 0) ? OrderStatus::FilledPartialRest
                                          : OrderStatus::AcceptedRested;
    } else {
      result.status = (result.filled > 0) ? OrderStatus::FilledPartialIoc
                                          : OrderStatus::RejectedOutOfRange;
    }
  } else {
    if (result.filled == order.orig_qty) {
      result.status = OrderStatus::FilledComplete;
    } else if (stp_cancelled_taker) {
      result.status = OrderStatus::CancelledByStp;
    } else if (result.filled > 0) {
      result.status = OrderStatus::FilledPartialIoc;
    } else {
      result.status = OrderStatus::ExpiredIocZero;
    }
  }

  result.remaining = order.qty;
  ++event_count_;
  return result;
}

MatchOutcome FlatLadderBook::add_outcome_into(const Order &in_order, TimeNs ts,
                                              Trade *trade_buf,
                                              std::size_t trade_cap,
                                              StpMode stp) noexcept {
  MatchOutcome outcome{};
  Order order = in_order;
  if (!is_valid_ || order.qty <= 0) [[unlikely]] {
    outcome.status = OrderStatus::RejectedInvalid;
    return outcome;
  }

  book_time_ = ts;
  order.id = next_id_++;
  order.orig_qty = order.qty;
  outcome.id = order.id;
  const StpMode effective_stp =
      (stp != StpMode::None)
          ? stp
          : (order.stp != StpMode::None ? order.stp : default_stp_mode_);

  if (order.type == OrderType::PostOnly) {
    if (is_buy(order.side)) {
      if (best_ask_idx_ != kNpos && price_of(best_ask_idx_) <= order.price) {
        outcome.status = OrderStatus::RejectedPostOnly;
        ++event_count_;
        return outcome;
      }
    } else {
      if (best_bid_idx_ != kNpos && price_of(best_bid_idx_) >= order.price) {
        outcome.status = OrderStatus::RejectedPostOnly;
        ++event_count_;
        return outcome;
      }
    }
    if (!in_range(order.price)) {
      outcome.status = OrderStatus::RejectedOutOfRange;
      ++event_count_;
      return outcome;
    }
    if (active_undo_log_ == nullptr) [[likely]] {
      rest_impl<false>(order);
    } else {
      rest_impl<true>(order);
    }
    outcome.status = OrderStatus::AcceptedRested;
    ++event_count_;
    return outcome;
  }

  if (order.type == OrderType::FOK) {
    if (!can_fill_fok(order.side, order.price, order.qty, false, order.agent,
                      effective_stp)) {
      outcome.status = OrderStatus::RejectedFok;
      ++event_count_;
      return outcome;
    }
  }

  const bool is_limit = (order.type == OrderType::Limit ||
                         order.type == OrderType::IOC ||
                         order.type == OrderType::FOK);

  auto record_trade = [&](const Trade &tr) noexcept {
    if (trade_buf != nullptr && outcome.trade_count < trade_cap) {
      trade_buf[outcome.trade_count] = tr;
    }
    if (outcome.trade_count < std::numeric_limits<std::uint16_t>::max()) {
      ++outcome.trade_count;
    }
    outcome.filled += tr.qty;
  };

  bool stp_cancelled_taker = false;
  if (active_undo_log_ == nullptr) [[likely]] {
    if (is_buy(order.side)) {
      if (is_limit)
        match_incoming<Side::Buy, true, false>(order, record_trade, order.price, ts,
                                               effective_stp, outcome.stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Buy, false, false>(order, record_trade, order.price, ts,
                                                effective_stp, outcome.stp_cancels, stp_cancelled_taker);
    } else {
      if (is_limit)
        match_incoming<Side::Sell, true, false>(order, record_trade, order.price, ts,
                                                effective_stp, outcome.stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Sell, false, false>(order, record_trade, order.price, ts,
                                                 effective_stp, outcome.stp_cancels, stp_cancelled_taker);
    }
  } else {
    if (is_buy(order.side)) {
      if (is_limit)
        match_incoming<Side::Buy, true, true>(order, record_trade, order.price, ts,
                                              effective_stp, outcome.stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Buy, false, true>(order, record_trade, order.price, ts,
                                               effective_stp, outcome.stp_cancels, stp_cancelled_taker);
    } else {
      if (is_limit)
        match_incoming<Side::Sell, true, true>(order, record_trade, order.price, ts,
                                               effective_stp, outcome.stp_cancels, stp_cancelled_taker);
      else
        match_incoming<Side::Sell, false, true>(order, record_trade, order.price, ts,
                                                effective_stp, outcome.stp_cancels, stp_cancelled_taker);
    }
  }

  if (order.type == OrderType::Limit && order.qty > 0) {
    if (in_range(order.price)) {
      if (active_undo_log_ == nullptr) [[likely]] {
        rest_impl<false>(order);
      } else {
        rest_impl<true>(order);
      }
      outcome.status = (outcome.filled > 0) ? OrderStatus::FilledPartialRest
                                            : OrderStatus::AcceptedRested;
    } else {
      // Out-of-Range Limit Order Fix: when !in_range(order.price), do NOT report remaining > 0;
      // set status = RejectedOutOfRange (if 0 filled) or FilledPartialIoc (if partially filled),
      // so MatchOutcome::remaining() / MatchSummary::remaining is 0!
      outcome.status = (outcome.filled > 0) ? OrderStatus::FilledPartialIoc
                                            : OrderStatus::RejectedOutOfRange;
    }
  } else {
    if (outcome.filled == order.orig_qty) {
      outcome.status = OrderStatus::FilledComplete;
    } else if (stp_cancelled_taker) {
      outcome.status = OrderStatus::CancelledByStp;
    } else if (outcome.filled > 0) {
      outcome.status = OrderStatus::FilledPartialIoc;
    } else {
      outcome.status = OrderStatus::ExpiredIocZero;
    }
  }

  ++event_count_;
  return outcome;
}

template <bool RecordUndo>
void FlatLadderBook::unlink_maker_node_impl(std::size_t level_idx, std::uint32_t slot, Side maker_side) noexcept {
  IntrusiveLevel &level = level_at(level_idx);
  IntrusiveOrderNode &node = node_pool_[slot];
  const OrderId id = node.id;
  const Qty removed_qty = node.qty;
  const std::size_t side_idx = static_cast<std::size_t>(maker_side) & 1U;
  const std::uint32_t prev_s = node.prev_slot;
  const std::uint32_t next_s = node.next_slot;

  if constexpr (!RecordUndo) {
    clear_order_ref(id);
    level.volume -= removed_qty;
    total_vol_[side_idx] -= removed_qty;
    --level.live_orders;

    if (prev_s != kNullSlot) {
      node_pool_[prev_s].next_slot = next_s;
    } else {
      level.head_slot = next_s;
    }

    if (next_s != kNullSlot) {
      node_pool_[next_s].prev_slot = prev_s;
    } else {
      level.tail_slot = prev_s;
    }

    free_slot(slot);

    if (level.volume == 0) {
      level.head_slot = kNullSlot;
      level.tail_slot = kNullSlot;
      level_side_[level_idx] = kNoSide;
      clear_level_occupied(level_idx);
    }
  } else {
    const std::uint32_t old_head = level.head_slot;
    const std::uint32_t old_tail = level.tail_slot;
    const std::uint32_t old_live = level.live_orders;
    const std::uint32_t old_vol = static_cast<std::uint32_t>(level.volume);
    const std::uint32_t old_tot = static_cast<std::uint32_t>(total_vol_[side_idx]);
    const std::uint8_t old_side = level_side_[level_idx];
    const std::uint32_t old_free = free_head_;
    const std::uint64_t old_l_bm = level_bitmap_[level_idx >> 6];
    const std::uint64_t old_s_bm = summary_bitmap_[level_idx >> 12];
    const std::size_t id_idx = static_cast<std::size_t>(id);
    const std::uint8_t old_id_live = (id_idx < id_live_.size()) ? id_live_[id_idx] : 0U;
    const OrderRef old_ref = (id_idx < id_refs_.size()) ? id_refs_[id_idx] : OrderRef{};
    const IntrusiveOrderNode saved_node = node;
    const std::uint32_t old_prev_next = (prev_s != kNullSlot) ? node_pool_[prev_s].next_slot : 0U;
    const std::uint32_t old_next_prev = (next_s != kNullSlot) ? node_pool_[next_s].prev_slot : 0U;

    clear_order_ref(id);
    level.volume -= removed_qty;
    total_vol_[side_idx] -= removed_qty;
    --level.live_orders;

    if (prev_s != kNullSlot) {
      node_pool_[prev_s].next_slot = next_s;
    } else {
      level.head_slot = next_s;
    }

    if (next_s != kNullSlot) {
      node_pool_[next_s].prev_slot = prev_s;
    } else {
      level.tail_slot = prev_s;
    }

    free_slot(slot);

    if (level.volume == 0) {
      level.head_slot = kNullSlot;
      level.tail_slot = kNullSlot;
      level_side_[level_idx] = kNoSide;
      clear_level_occupied(level_idx);
    }

    CompactDeltaToken tok{};
    tok.kind = LadderDeltaKind::UnlinkNode;
    tok.order_side = maker_side;
    tok.side_xor = static_cast<std::uint8_t>(old_side ^ level_side_[level_idx]);
    tok.id_live_xor = static_cast<std::uint8_t>(old_id_live ^ ((id_idx < id_live_.size()) ? id_live_[id_idx] : 0U));
    tok.level_idx = static_cast<std::uint32_t>(level_idx);
    tok.slot = slot;
    tok.vol_xor = old_vol ^ static_cast<std::uint32_t>(level.volume);
    tok.tot_vol_xor = old_tot ^ static_cast<std::uint32_t>(total_vol_[side_idx]);
    tok.live_xor = old_live ^ level.live_orders;
    tok.head_xor = old_head ^ level.head_slot;

    StructuralUndoPayload payload{};
    payload.tail_xor = old_tail ^ level.tail_slot;
    payload.free_head_xor = old_free ^ free_head_;
    payload.prev_next_xor = (prev_s != kNullSlot) ? (old_prev_next ^ node_pool_[prev_s].next_slot) : 0U;
    payload.next_prev_xor = (next_s != kNullSlot) ? (old_next_prev ^ node_pool_[next_s].prev_slot) : 0U;
    payload.l_bitmap_xor = old_l_bm ^ level_bitmap_[level_idx >> 6];
    payload.s_bitmap_xor = old_s_bm ^ summary_bitmap_[level_idx >> 12];
    payload.order_id = id;
    payload.saved_ref = old_ref;
    payload.saved_node = saved_node;
    active_undo_log_->push_structural(tok, payload);
  }
}

void FlatLadderBook::unlink_maker_node(std::size_t level_idx, std::uint32_t slot, Side maker_side) noexcept {
  if (active_undo_log_ == nullptr) [[likely]] {
    unlink_maker_node_impl<false>(level_idx, slot, maker_side);
  } else {
    unlink_maker_node_impl<true>(level_idx, slot, maker_side);
  }
}

template <Side TakerSide, bool IsLimit, bool RecordUndo, typename TradeSink>
[[gnu::hot]]
void FlatLadderBook::match_incoming(Order &incoming, TradeSink &&on_trade,
                                    Tick limit_price, TimeNs ts, StpMode stp,
                                    std::uint8_t &stp_cancels,
                                    bool &stp_cancelled_taker) {
  constexpr bool kBuy = (TakerSide == Side::Buy);
  constexpr Side kMakerSide = kBuy ? Side::Sell : Side::Buy;
  constexpr std::size_t kOppositeSide = kBuy ? 1U : 0U;

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
        if (stp_cancels < 255U) ++stp_cancels;
        if (stp == StpMode::CancelPassive) {
          unlink_maker_node_impl<RecordUndo>(idx, slot, kMakerSide);
          continue;
        } else if (stp == StpMode::CancelAggressive) {
          incoming.qty = 0;
          stp_cancelled_taker = true;
          break;
        } else if (stp == StpMode::DecrementAndCancel) {
          const Qty dec = std::min(incoming.qty, maker.qty);
          incoming.qty -= dec;
          if (dec == maker.qty) {
            unlink_maker_node_impl<RecordUndo>(idx, slot, kMakerSide);
          } else {
            if constexpr (!RecordUndo) {
              maker.qty -= dec;
              level.volume -= dec;
              total_vol_[kOppositeSide] -= dec;
            } else {
              const std::uint32_t old_vol = static_cast<std::uint32_t>(level.volume);
              const std::uint32_t old_tot = static_cast<std::uint32_t>(total_vol_[kOppositeSide]);
              const std::uint32_t old_node_qty = static_cast<std::uint32_t>(maker.qty);
              maker.qty -= dec;
              level.volume -= dec;
              total_vol_[kOppositeSide] -= dec;
              active_undo_log_->push_partial(
                  kMakerSide, static_cast<std::uint32_t>(idx), slot,
                  old_vol ^ static_cast<std::uint32_t>(level.volume),
                  old_tot ^ static_cast<std::uint32_t>(total_vol_[kOppositeSide]),
                  old_node_qty ^ static_cast<std::uint32_t>(maker.qty));
            }
          }
          if (incoming.qty == 0) {
            stp_cancelled_taker = true;
            break;
          }
          continue;
        }
      }

      const Qty fill = std::min(incoming.qty, maker.qty);
      on_trade(Trade{maker.id, incoming.id, maker.agent, incoming.agent,
                     TakerSide, maker_price, fill, ts});
      incoming.qty -= fill;

      if (fill == maker.qty) {
        unlink_maker_node_impl<RecordUndo>(idx, slot, kMakerSide);
      } else {
        if constexpr (!RecordUndo) {
          maker.qty -= fill;
          level.volume -= fill;
          total_vol_[kOppositeSide] -= fill;
        } else {
          const std::uint32_t old_vol = static_cast<std::uint32_t>(level.volume);
          const std::uint32_t old_tot = static_cast<std::uint32_t>(total_vol_[kOppositeSide]);
          const std::uint32_t old_node_qty = static_cast<std::uint32_t>(maker.qty);
          maker.qty -= fill;
          level.volume -= fill;
          total_vol_[kOppositeSide] -= fill;
          active_undo_log_->push_partial(
              kMakerSide, static_cast<std::uint32_t>(idx), slot,
              old_vol ^ static_cast<std::uint32_t>(level.volume),
              old_tot ^ static_cast<std::uint32_t>(total_vol_[kOppositeSide]),
              old_node_qty ^ static_cast<std::uint32_t>(maker.qty));
        }
      }
    }

    if (level.volume == 0) {
      if constexpr (kBuy) {
        refresh_best_ask_from(idx + 1);
      } else {
        refresh_best_bid_from(idx == 0 ? kNpos : idx - 1);
      }
    }
  }
}

template <bool RecordUndo>
void FlatLadderBook::rest_impl(Order &order) {
  order.enqueue_ts = book_time_;
  const std::size_t idx = index_of(order.price);
  IntrusiveLevel &level = level_at(idx);
  const std::size_t side_idx = static_cast<std::size_t>(order.side) & 1U;
  const std::uint32_t prev_tail = level.tail_slot;

  if constexpr (!RecordUndo) {
    const std::uint32_t slot = alloc_slot();
    IntrusiveOrderNode &node = node_pool_[slot];
    node.id = order.id;
    node.enqueue_ts = order.enqueue_ts;
    node.qty = order.qty;
    node.agent = order.agent;
    node.prev_slot = prev_tail;
    node.next_slot = kNullSlot;

    if (prev_tail != kNullSlot) {
      node_pool_[prev_tail].next_slot = slot;
    } else {
      level.head_slot = slot;
    }
    level.tail_slot = slot;

    ++level.live_orders;
    level.volume += order.qty;
    level_side_[idx] = side_tag(order.side);
    mark_level_occupied(idx);

    set_order_ref(order.id, {order.side, 0, static_cast<std::uint32_t>(idx), slot});

    total_vol_[side_idx] += order.qty;
    if (is_buy(order.side)) {
      if (best_bid_idx_ == kNpos || idx > best_bid_idx_)
        best_bid_idx_ = idx;
    } else {
      if (best_ask_idx_ == kNpos || idx < best_ask_idx_)
        best_ask_idx_ = idx;
    }
  } else {
    const std::uint32_t old_head = level.head_slot;
    const std::uint32_t old_tail = level.tail_slot;
    const std::uint32_t old_live = level.live_orders;
    const std::uint32_t old_vol = static_cast<std::uint32_t>(level.volume);
    const std::uint32_t old_tot = static_cast<std::uint32_t>(total_vol_[side_idx]);
    const std::uint8_t old_side = level_side_[idx];
    const std::uint64_t old_l_bm = level_bitmap_[idx >> 6];
    const std::uint64_t old_s_bm = summary_bitmap_[idx >> 12];
    const std::uint32_t old_prev_next = (prev_tail != kNullSlot) ? node_pool_[prev_tail].next_slot : 0U;

    const bool reused_from_free = (free_head_ != kNullSlot);
    const std::uint32_t old_free = free_head_;
    const IntrusiveOrderNode saved_free_node = reused_from_free ? node_pool_[free_head_] : IntrusiveOrderNode{};

    const std::uint32_t slot = alloc_slot();
    IntrusiveOrderNode &node = node_pool_[slot];
    node.id = order.id;
    node.enqueue_ts = order.enqueue_ts;
    node.qty = order.qty;
    node.agent = order.agent;
    node.prev_slot = prev_tail;
    node.next_slot = kNullSlot;

    if (prev_tail != kNullSlot) {
      node_pool_[prev_tail].next_slot = slot;
    } else {
      level.head_slot = slot;
    }
    level.tail_slot = slot;

    ++level.live_orders;
    level.volume += order.qty;
    level_side_[idx] = side_tag(order.side);
    mark_level_occupied(idx);

    const std::size_t id_idx = static_cast<std::size_t>(order.id);
    const std::uint8_t old_id_live = (id_idx < id_live_.size()) ? id_live_[id_idx] : 0U;
    const OrderRef old_ref = (id_idx < id_refs_.size()) ? id_refs_[id_idx] : OrderRef{};

    set_order_ref(order.id, {order.side, 0, static_cast<std::uint32_t>(idx), slot});

    total_vol_[side_idx] += order.qty;
    if (is_buy(order.side)) {
      if (best_bid_idx_ == kNpos || idx > best_bid_idx_)
        best_bid_idx_ = idx;
    } else {
      if (best_ask_idx_ == kNpos || idx < best_ask_idx_)
        best_ask_idx_ = idx;
    }

    CompactDeltaToken tok{};
    tok.kind = LadderDeltaKind::InsertRest;
    tok.order_side = order.side;
    tok.side_xor = static_cast<std::uint8_t>(old_side ^ level_side_[idx]);
    tok.id_live_xor = static_cast<std::uint8_t>(old_id_live ^ id_live_[id_idx]);
    tok.level_idx = static_cast<std::uint32_t>(idx);
    tok.slot = slot;
    tok.vol_xor = old_vol ^ static_cast<std::uint32_t>(level.volume);
    tok.tot_vol_xor = old_tot ^ static_cast<std::uint32_t>(total_vol_[side_idx]);
    tok.live_xor = old_live ^ level.live_orders;
    tok.head_xor = old_head ^ level.head_slot;

    StructuralUndoPayload payload{};
    payload.tail_xor = old_tail ^ level.tail_slot;
    payload.free_head_xor = old_free ^ free_head_;
    payload.prev_next_xor = (prev_tail != kNullSlot) ? (old_prev_next ^ node_pool_[prev_tail].next_slot) : 0U;
    payload.l_bitmap_xor = old_l_bm ^ level_bitmap_[idx >> 6];
    payload.s_bitmap_xor = old_s_bm ^ summary_bitmap_[idx >> 12];
    payload.order_id = order.id;
    payload.saved_ref = old_ref;
    payload.saved_node = saved_free_node;
    payload.saved_node.prev_slot = prev_tail;
    payload.saved_node.agent = reused_from_free ? 1U : 0U;
    active_undo_log_->push_structural(tok, payload);
  }
}

void FlatLadderBook::rest(Order &order) {
  if (active_undo_log_ == nullptr) [[likely]] {
    rest_impl<false>(order);
  } else {
    rest_impl<true>(order);
  }
}

CancelOutcome FlatLadderBook::cancel_outcome(OrderId id, TimeNs ts) noexcept {
  book_time_ = ts;
  CancelOutcome out{};
  if (!is_valid_ || id == 0 || id >= next_id_) [[unlikely]] {
    out.status = CancelStatus::NotFound;
    return out;
  }
  if (!has_order_ref(id)) [[unlikely]] {
    out.status = CancelStatus::AlreadyInactive;
    return out;
  }

  const OrderRef ref = get_order_ref(id);
  const std::size_t level_idx = ref.level_idx;
  const std::uint32_t slot = ref.slot;
  out.cancelled_qty = node_pool_[slot].qty;
  const auto clamped_offset = std::clamp<std::int64_t>(
      static_cast<std::int64_t>(level_idx),
      std::numeric_limits<std::int16_t>::min(),
      std::numeric_limits<std::int16_t>::max());
  out.price_offset = static_cast<std::int16_t>(clamped_offset);
  out.side = ref.side;
  out.status = CancelStatus::Cancelled;

  if (active_undo_log_ == nullptr) [[likely]] {
    unlink_maker_node_impl<false>(level_idx, slot, ref.side);
  } else {
    unlink_maker_node_impl<true>(level_idx, slot, ref.side);
  }

  const IntrusiveLevel &level = level_at(level_idx);
  if (level.volume == 0) {
    if (is_buy(ref.side) && best_bid_idx_ == level_idx)
      refresh_best_bid_from(level_idx == 0 ? kNpos : level_idx - 1);
    if (is_sell(ref.side) && best_ask_idx_ == level_idx)
      refresh_best_ask_from(level_idx + 1);
  }

  ++event_count_;
  return out;
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

OrderId FlatLadderBook::peek_head_order_id(Side side, Tick price) const noexcept {
  if (!is_valid_ || !in_range(price)) return 0ULL;
  const std::size_t idx = index_of(price);
  if (level_side_[idx] != side_tag(side)) return 0ULL;
  const std::uint32_t slot = levels_[idx].head_slot;
  if (slot == kNullSlot || slot >= node_pool_.size()) return 0ULL;
  return node_pool_[slot].id;
}

OrderId FlatLadderBook::peek_tail_order_id(Side side, Tick price) const noexcept {
  if (!is_valid_ || !in_range(price)) return 0ULL;
  const std::size_t idx = index_of(price);
  if (level_side_[idx] != side_tag(side)) return 0ULL;
  const std::uint32_t slot = levels_[idx].tail_slot;
  if (slot == kNullSlot || slot >= node_pool_.size()) return 0ULL;
  return node_pool_[slot].id;
}

OrderId FlatLadderBook::peek_nth_order_id(Side side, Tick price, std::size_t n) const noexcept {
  if (!is_valid_ || !in_range(price)) return 0ULL;
  const std::size_t idx = index_of(price);
  if (level_side_[idx] != side_tag(side)) return 0ULL;
  std::uint32_t curr = levels_[idx].head_slot;
  std::size_t step = 0;
  while (curr != kNullSlot && curr < node_pool_.size()) {
    if (step == n) return node_pool_[curr].id;
    curr = node_pool_[curr].next_slot;
    ++step;
  }
  return 0ULL;
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

std::size_t FlatLadderBook::levels_into(Side side, BookLevelView *out,
                                        std::size_t max_levels) const noexcept {
  if (!is_valid_ || tick_count_ == 0 || out == nullptr || max_levels == 0) {
    return 0;
  }
  std::size_t count = 0;
  if (side == Side::Buy) {
    const std::uint8_t tag = kBuySide;
    for (std::size_t idx = best_bid_idx_; idx != kNpos;
         idx = (idx == 0 ? kNpos : find_prev_occupied_from(idx - 1))) {
      const IntrusiveLevel &level = level_at(idx);
      if (!level.empty() && level_side_[idx] == tag) {
        out[count++] = BookLevelView{price_of(idx), level.volume, level.live_orders};
        if (count >= max_levels) {
          break;
        }
      }
    }
  } else {
    const std::uint8_t tag = kSellSide;
    for (std::size_t idx = best_ask_idx_; idx != kNpos && idx < tick_count_;
         idx = find_next_occupied_from(idx + 1)) {
      const IntrusiveLevel &level = level_at(idx);
      if (!level.empty() && level_side_[idx] == tag) {
        out[count++] = BookLevelView{price_of(idx), level.volume, level.live_orders};
        if (count >= max_levels) {
          break;
        }
      }
    }
  }
  return count;
}

std::vector<BookLevelView> FlatLadderBook::levels(Side side, std::size_t depth) const {
  std::vector<BookLevelView> out;
  if (!is_valid_ || tick_count_ == 0) {
    return out;
  }
  if (depth > 0 && depth <= 64) {
    BookLevelView stack_buf[64];
    const std::size_t n = levels_into(side, stack_buf, depth);
    out.assign(stack_buf, stack_buf + n);
    return out;
  }
  if (depth > 0) {
    out.reserve(std::min<std::size_t>(depth, 256));
  }
  if (side == Side::Buy) {
    const std::uint8_t tag = kBuySide;
    for (std::size_t idx = best_bid_idx_; idx != kNpos;
         idx = (idx == 0 ? kNpos : find_prev_occupied_from(idx - 1))) {
      const IntrusiveLevel &level = level_at(idx);
      if (!level.empty() && level_side_[idx] == tag) {
        out.push_back({price_of(idx), level.volume, level.live_orders});
        if (depth > 0 && out.size() >= depth) {
          break;
        }
      }
    }
  } else {
    const std::uint8_t tag = kSellSide;
    for (std::size_t idx = best_ask_idx_; idx != kNpos && idx < tick_count_;
         idx = find_next_occupied_from(idx + 1)) {
      const IntrusiveLevel &level = level_at(idx);
      if (!level.empty() && level_side_[idx] == tag) {
        out.push_back({price_of(idx), level.volume, level.live_orders});
        if (depth > 0 && out.size() >= depth) {
          break;
        }
      }
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
  best_bid_idx_ = find_prev_occupied_from(start);
}

void FlatLadderBook::refresh_best_ask_from(std::size_t start) noexcept {
  best_ask_idx_ = find_next_occupied_from(start);
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

void FlatLadderBook::attach_undo_log(LadderUndoLog *log) noexcept {
  if (log != nullptr) {
    log->reset();
    log->saved_best_bid_idx = best_bid_idx_;
    log->saved_best_ask_idx = best_ask_idx_;
    log->saved_live_order_count = live_order_count_;
    log->saved_event_count = event_count_;
    log->saved_next_id = next_id_;
    log->saved_book_time = book_time_;
  }
  active_undo_log_ = log;
}

void FlatLadderBook::rollback_undo_log_to(LadderUndoLog &log,
                                          std::size_t target_count,
                                          std::size_t saved_best_bid,
                                          std::size_t saved_best_ask,
                                          std::size_t saved_live_orders,
                                          std::uint64_t saved_events,
                                          OrderId saved_next,
                                          TimeNs saved_time) noexcept {
  while (log.size() > target_count) {
    const CompactDeltaToken &tok = log.pop_token();
    const std::size_t idx = tok.level_idx;
    IntrusiveLevel &level = level_at(idx);
    const std::size_t side_idx = static_cast<std::size_t>(tok.order_side) & 1U;

    if (tok.kind == LadderDeltaKind::PartialFill) {
      const std::uint32_t cur_vol = static_cast<std::uint32_t>(level.volume);
      const std::uint32_t cur_tot = static_cast<std::uint32_t>(total_vol_[side_idx]);
      const std::uint32_t cur_node_qty = static_cast<std::uint32_t>(node_pool_[tok.slot].qty);
      level.volume = static_cast<Qty>(cur_vol ^ tok.vol_xor);
      total_vol_[side_idx] = static_cast<Qty>(cur_tot ^ tok.tot_vol_xor);
      node_pool_[tok.slot].qty = static_cast<Qty>(cur_node_qty ^ tok.live_xor);
    } else if (tok.kind == LadderDeltaKind::UnlinkNode) {
      const StructuralUndoPayload &p = log.payload_at(tok.aux_idx);
      level.head_slot ^= tok.head_xor;
      level.tail_slot ^= p.tail_xor;
      level.live_orders ^= tok.live_xor;
      level.volume = static_cast<Qty>(static_cast<std::uint32_t>(level.volume) ^ tok.vol_xor);
      total_vol_[side_idx] = static_cast<Qty>(static_cast<std::uint32_t>(total_vol_[side_idx]) ^ tok.tot_vol_xor);
      level_side_[idx] ^= tok.side_xor;
      level_bitmap_[idx >> 6] ^= p.l_bitmap_xor;
      summary_bitmap_[idx >> 12] ^= p.s_bitmap_xor;
      free_head_ ^= p.free_head_xor;

      node_pool_[tok.slot] = p.saved_node;
      const std::uint32_t prev_s = p.saved_node.prev_slot;
      const std::uint32_t next_s = p.saved_node.next_slot;
      if (prev_s != kNullSlot) {
        node_pool_[prev_s].next_slot ^= p.prev_next_xor;
      }
      if (next_s != kNullSlot) {
        node_pool_[next_s].prev_slot ^= p.next_prev_xor;
      }

      const std::size_t id_idx = static_cast<std::size_t>(p.order_id);
      if (id_idx < id_live_.size()) {
        id_live_[id_idx] ^= tok.id_live_xor;
        id_refs_[id_idx] = p.saved_ref;
      }
    } else if (tok.kind == LadderDeltaKind::InsertRest) {
      const StructuralUndoPayload &p = log.payload_at(tok.aux_idx);
      level.head_slot ^= tok.head_xor;
      level.tail_slot ^= p.tail_xor;
      level.live_orders ^= tok.live_xor;
      level.volume = static_cast<Qty>(static_cast<std::uint32_t>(level.volume) ^ tok.vol_xor);
      total_vol_[side_idx] = static_cast<Qty>(static_cast<std::uint32_t>(total_vol_[side_idx]) ^ tok.tot_vol_xor);
      level_side_[idx] ^= tok.side_xor;
      level_bitmap_[idx >> 6] ^= p.l_bitmap_xor;
      summary_bitmap_[idx >> 12] ^= p.s_bitmap_xor;

      const std::uint32_t prev_tail = p.saved_node.prev_slot;
      if (prev_tail != kNullSlot) {
        node_pool_[prev_tail].next_slot ^= p.prev_next_xor;
      }

      const std::size_t id_idx = static_cast<std::size_t>(p.order_id);
      if (id_idx < id_live_.size()) {
        id_live_[id_idx] ^= tok.id_live_xor;
        id_refs_[id_idx] = p.saved_ref;
      }

      free_head_ ^= p.free_head_xor;
      if (p.saved_node.agent != 0U) {
        IntrusiveOrderNode restored = p.saved_node;
        restored.prev_slot = kNullSlot;
        node_pool_[tok.slot] = restored;
      } else if (tok.slot + 1U == static_cast<std::uint32_t>(node_pool_.size())) {
        node_pool_.pop_back();
      } else {
        node_pool_[tok.slot] = IntrusiveOrderNode{};
      }
    }
  }

  best_bid_idx_ = saved_best_bid;
  best_ask_idx_ = saved_best_ask;
  live_order_count_ = saved_live_orders;
  event_count_ = saved_events;
  next_id_ = saved_next;
  book_time_ = saved_time;
}

void FlatLadderBook::rollback_undo_log(LadderUndoLog &log) noexcept {
  rollback_undo_log_to(log, 0U,
                       log.saved_best_bid_idx, log.saved_best_ask_idx,
                       log.saved_live_order_count, log.saved_event_count,
                       log.saved_next_id, log.saved_book_time);
}

std::uint64_t FlatLadderBook::state_digest() const noexcept {
  std::uint64_t h = 0xCBF29CE484222325ULL;
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(best_bid_idx_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(best_ask_idx_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(live_order_count_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(total_vol_[0]));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(total_vol_[1]));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(event_count_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(next_id_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(book_time_));
  h = fnv1a_mix64(h, static_cast<std::uint64_t>(free_head_));

  for (std::uint64_t w : summary_bitmap_) {
    h = fnv1a_mix64(h, w);
  }
  for (std::size_t sw = 0; sw < summary_bitmap_.size(); ++sw) {
    std::uint64_t s_val = summary_bitmap_[sw];
    while (s_val != 0) {
      const int lo_w = std::countr_zero(s_val);
      s_val &= (s_val - 1ULL);
      const std::size_t lw_idx = (sw << 6) + static_cast<std::size_t>(lo_w);
      if (lw_idx >= level_bitmap_.size()) continue;
      std::uint64_t l_val = level_bitmap_[lw_idx];
      h = fnv1a_mix64(h, lw_idx);
      h = fnv1a_mix64(h, l_val);
      while (l_val != 0) {
        const int lo_b = std::countr_zero(l_val);
        l_val &= (l_val - 1ULL);
        const std::size_t idx = (lw_idx << 6) + static_cast<std::size_t>(lo_b);
        if (idx >= tick_count_) continue;
        const IntrusiveLevel &lvl = levels_[idx];
        h = fnv1a_mix64(h, idx);
        h = fnv1a_mix64(h, level_side_[idx]);
        h = fnv1a_mix64(h, lvl.head_slot);
        h = fnv1a_mix64(h, lvl.tail_slot);
        h = fnv1a_mix64(h, lvl.live_orders);
        h = fnv1a_mix64(h, static_cast<std::uint64_t>(lvl.volume));
        for (std::uint32_t curr = lvl.head_slot; curr != kNullSlot; curr = node_pool_[curr].next_slot) {
          const IntrusiveOrderNode &n = node_pool_[curr];
          h = fnv1a_mix64(h, curr);
          h = fnv1a_mix64(h, n.id);
          h = fnv1a_mix64(h, static_cast<std::uint64_t>(n.enqueue_ts));
          h = fnv1a_mix64(h, static_cast<std::uint64_t>(n.qty));
          h = fnv1a_mix64(h, static_cast<std::uint64_t>(n.agent));
          h = fnv1a_mix64(h, n.prev_slot);
          h = fnv1a_mix64(h, n.next_slot);
        }
      }
    }
  }
  return h;
}

bool FlatLadderBook::state_equals(const FlatLadderBook &other) const noexcept {
  if (min_tick_ != other.min_tick_ || max_tick_ != other.max_tick_ ||
      tick_count_ != other.tick_count_ || best_bid_idx_ != other.best_bid_idx_ ||
      best_ask_idx_ != other.best_ask_idx_ || live_order_count_ != other.live_order_count_ ||
      total_vol_[0] != other.total_vol_[0] || total_vol_[1] != other.total_vol_[1] ||
      event_count_ != other.event_count_ || next_id_ != other.next_id_ ||
      book_time_ != other.book_time_ || free_head_ != other.free_head_) {
    return false;
  }
  if (level_bitmap_ != other.level_bitmap_ ||
      summary_bitmap_ != other.summary_bitmap_ ||
      level_side_ != other.level_side_) {
    return false;
  }
  for (std::size_t i = 0; i < tick_count_; ++i) {
    const IntrusiveLevel &a = levels_[i];
    const IntrusiveLevel &b = other.levels_[i];
    if (a.head_slot != b.head_slot || a.tail_slot != b.tail_slot ||
        a.live_orders != b.live_orders || a.volume != b.volume) {
      return false;
    }
  }
  if (node_pool_.size() != other.node_pool_.size()) return false;
  std::uint32_t fa = free_head_;
  std::uint32_t fb = other.free_head_;
  std::size_t free_steps = 0;
  while (fa != kNullSlot || fb != kNullSlot) {
    if (fa != fb || free_steps > node_pool_.size()) return false;
    fa = node_pool_[fa].next_slot;
    fb = other.node_pool_[fb].next_slot;
    ++free_steps;
  }
  const std::size_t max_id = std::min(id_live_.size(), other.id_live_.size());
  for (std::size_t id = 0; id < max_id; ++id) {
    if (id_live_[id] != other.id_live_[id]) return false;
    if (id_live_[id] != 0) {
      const OrderRef &ra = id_refs_[id];
      const OrderRef &rb = other.id_refs_[id];
      if (ra.side != rb.side || ra.level_idx != rb.level_idx || ra.slot != rb.slot) {
        return false;
      }
    }
  }
  return state_digest() == other.state_digest();
}

} // namespace sim
