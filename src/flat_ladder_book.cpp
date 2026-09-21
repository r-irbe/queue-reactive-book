#include "sim/flat_ladder_book.hpp"
#include <cstddef>

namespace sim {
namespace {
// Sentinel for "no best price index cached". Compares != to any valid index.
constexpr std::size_t kNpos = std::numeric_limits<std::size_t>::max();
} // namespace

FlatLadderBook::FlatLadderBook(Tick min_tick, Tick max_tick,
                               std::size_t expected_orders,
                               bool enable_compaction,
                               CompactionConfig compaction_cfg,
                               std::size_t arena_bytes)
    : arena_(arena_bytes == 0
                 ? suggest_arena_bytes(
                       expected_orders +
                       static_cast<std::size_t>(max_tick >= min_tick
                                                    ? (max_tick - min_tick + 1)
                                                    : 0))
                 : arena_bytes,
             std::pmr::new_delete_resource()),
      pool_(&arena_), levels_(resource()), level_side_(resource()),
      level_bitmap_(resource()), id_refs_(resource()), id_live_(resource()),
      min_tick_(min_tick), max_tick_(max_tick),
      tick_count_(max_tick >= min_tick
                      ? static_cast<std::size_t>(max_tick - min_tick + 1)
                      : 0),
      enable_compaction_(enable_compaction), compaction_cfg_(compaction_cfg) {
  // Construction and test/debug invariant checks may throw; production paths
  // return data (bool / ExecutionResult) to keep hot paths noexcept-friendly.
  if (max_tick < min_tick) {
    throw std::invalid_argument("FlatLadderBook: max_tick < min_tick");
  }
  levels_.reserve(tick_count_);
  level_side_.assign(tick_count_, kNoSide);
  // Bitmap: 1 bit per ladder slot; words are 64-bit so we round up.
  level_bitmap_.assign((tick_count_ + 63) / 64, 0);
  for (std::size_t i = 0; i < tick_count_; ++i) {
    levels_.emplace_back(resource());
  }
  // OrderId 0 is a reserved sentinel; live storage indexes 1..N. Reserve
  // slot 0 so id-as-index lookups never need an off-by-one adjustment.
  id_refs_.reserve(expected_orders + 1);
  id_live_.reserve(expected_orders + 1);
  id_refs_.resize(1);
  id_live_.resize(1, 0);
}

bool FlatLadderBook::has_order_ref(OrderId id) const noexcept {
  const auto idx = static_cast<std::size_t>(id);
  return idx < id_live_.size() && id_live_[idx] != 0;
}

void FlatLadderBook::set_order_ref(OrderId id, const OrderRef &ref) {
  const auto idx = static_cast<std::size_t>(id);
  if (idx >= id_refs_.size()) {
    id_refs_.resize(idx + 1);
    id_live_.resize(idx + 1, 0);
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
  return tick_count_ > 0 && price >= min_tick_ && price <= max_tick_;
}

std::size_t FlatLadderBook::index_of(Tick price) const noexcept {
  return static_cast<std::size_t>(price - min_tick_);
}

Tick FlatLadderBook::price_of(std::size_t idx) const noexcept {
  return static_cast<Tick>(min_tick_ + static_cast<Tick>(idx));
}

Level &FlatLadderBook::level_at(std::size_t idx) noexcept {
  return levels_[idx];
}
const Level &FlatLadderBook::level_at(std::size_t idx) const noexcept {
  return levels_[idx];
}

// Single entry point for both Market and Limit; matching dispatch and
// rest-residual policy diverge only by `order.type`.
ExecutionResult FlatLadderBook::add(Order order, TimeNs ts) {
  // Validation belongs at the boundary (here) so the inner loops stay branch-light.
  book_time_ = ts;
  order.id = next_id_++;
  order.orig_qty = order.qty;
  const bool is_limit = (order.type == OrderType::Limit);

  ExecutionResult result;
  // Heuristic reserve: avoids small grow-resizes in common crosses; measured
  // maker-count and latency-tail evidence is still benchmark/replay gated.
  result.trades.reserve(8);

  if (is_buy(order.side)) {
    if (is_limit) match_incoming<Side::Buy,  true >(order, result, order.price, ts);
    else          match_incoming<Side::Buy,  false>(order, result, order.price, ts);
  } else {
    if (is_limit) match_incoming<Side::Sell, true >(order, result, order.price, ts);
    else          match_incoming<Side::Sell, false>(order, result, order.price, ts);
  }

  if (is_limit && order.qty > 0 && in_range(order.price)) {
    rest(order);
  }

  result.remaining = order.qty;
  result.filled = order.orig_qty - order.qty;
  ++event_count_; // cheap public health metric; never used on the hot path itself.
  return result;
}

// Match `incoming` against the opposite side. All side- and limit-specific
// logic resolves at compile time via `if constexpr`; the body is identical
// otherwise to avoid divergence bugs between buy/sell and limit/market paths.
template <Side TakerSide, bool IsLimit>
[[gnu::hot]]
void FlatLadderBook::match_incoming(Order &incoming, ExecutionResult &out,
                                    Tick limit_price, TimeNs ts) {
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

    Level &level = level_at(idx);
    while (level.head < level.orders.size() &&
           level.orders[level.head].qty == 0) {
      ++level.head;
    }

    while (incoming.qty > 0 && level.head < level.orders.size()) {
      Order &maker = level.orders[level.head];
      if (maker.qty == 0) {
        ++level.head;
        continue;
      }

      const Qty fill = std::min(incoming.qty, maker.qty);
      out.trades.push_back(Trade{.maker_id   = maker.id,
                                 .taker_id   = incoming.id,
                                 .maker      = maker.agent,
                                 .taker      = incoming.agent,
                                 .taker_side = incoming.side,
                                 .price      = maker_price,
                                 .qty        = fill,
                                 .ts         = ts});
      maker.qty       -= fill;
      level.volume    -= fill;
      incoming.qty    -= fill;
      if constexpr (kBuy) total_ask_vol_ -= fill;
      else                total_bid_vol_ -= fill;

      if (maker.qty == 0) {
        clear_order_ref(maker.id);
        --level.live_orders;
        ++level.head;
      }
    }

    if (level.volume == 0) {
      level.head = 0;
      level_side_[idx] = kNoSide;
      clear_level_occupied(idx);
      if constexpr (kBuy) {
        refresh_best_ask_from(idx + 1);
      } else {
        refresh_best_bid_from(idx == 0 ? kNpos : idx - 1);
      }
    } else {
      maybe_compact_level(idx, level);
    }
  }
}

// Explicit instantiations — keeps the template body in this TU.
// 4 = Side(Buy/Sell) × IsLimit(true/false).
template void FlatLadderBook::match_incoming<Side::Buy,  true >(Order &, ExecutionResult &, Tick, TimeNs);
template void FlatLadderBook::match_incoming<Side::Buy,  false>(Order &, ExecutionResult &, Tick, TimeNs);
template void FlatLadderBook::match_incoming<Side::Sell, true >(Order &, ExecutionResult &, Tick, TimeNs);
template void FlatLadderBook::match_incoming<Side::Sell, false>(Order &, ExecutionResult &, Tick, TimeNs);

// Compaction is independent of order time; it rewrites a level's storage
// in-place when tombstone density crosses configured thresholds.
void FlatLadderBook::maybe_compact_level(std::size_t idx, Level &level) {
  if (!enable_compaction_)
    return;

  const std::size_t total = level.orders.size();
  const std::size_t live = level.live_orders;
  const std::size_t dead = total - live;
  const CompactionConfig &cfg = compaction_cfg_;

  if (total < cfg.total_min)
    return;
  if (dead < cfg.dead_min)
    return;
  if (dead * cfg.dead_den < total * cfg.dead_num)
    return;

  std::pmr::vector<Order> compacted(resource());
  compacted.reserve(live);

  std::uint32_t next_idx = 0;
  for (std::size_t i = level.head; i < total; ++i) {
    Order &o = level.orders[i];
    if (o.qty <= 0)
      continue;
    compacted.push_back(std::move(o));
    set_order_ref(compacted.back().id,
                  {compacted.back().side, 0, static_cast<std::uint32_t>(idx),
                   next_idx++});
  }

  level.orders.swap(compacted);
  level.head = 0;
}

void FlatLadderBook::rest(Order &order) {
  order.enqueue_ts = book_time_;
  const std::size_t idx = index_of(order.price);
  Level &level = level_at(idx);
  if (level.orders.empty())
    level.orders.reserve(8);

  const std::uint32_t slot = static_cast<std::uint32_t>(level.orders.size());
  level.push(order);
  level.volume += order.qty;
  level_side_[idx] = side_tag(order.side);
  mark_level_occupied(idx);
  set_order_ref(order.id,
                {order.side, 0, static_cast<std::uint32_t>(idx), slot});

  if (is_buy(order.side)) {
    total_bid_vol_ += order.qty;
    if (best_bid_idx_ == kNpos || idx > best_bid_idx_)
      best_bid_idx_ = idx;
  } else {
    total_ask_vol_ += order.qty;
    if (best_ask_idx_ == kNpos || idx < best_ask_idx_)
      best_ask_idx_ = idx;
  }
}

bool FlatLadderBook::cancel(OrderId id, TimeNs ts) {
  book_time_ = ts;
  if (!has_order_ref(id)) [[unlikely]]
    return false;
  const OrderRef ref = get_order_ref(id);
  // No id-index erase needed: out-of-band orders never rest, so we never have
  // a stray ref to an unbacked level.
  clear_order_ref(id);
  const std::size_t level_idx = ref.level_idx;
  const std::size_t slot = ref.slot;
  Level &level = level_at(level_idx);
  Order &o = level.orders[slot];
  const Qty cancelled_qty = o.qty;
  level.volume -= cancelled_qty;
  if (is_buy(ref.side)) total_bid_vol_ -= cancelled_qty;
  else                  total_ask_vol_ -= cancelled_qty;
  --level.live_orders;
  o.qty = 0;

  if (level.volume == 0) {
    level.head = 0;
    level_side_[level_idx] = kNoSide;
    clear_level_occupied(level_idx);
    if (is_buy(ref.side) && best_bid_idx_ == level_idx)
      refresh_best_bid_from(level_idx == 0 ? kNpos : level_idx - 1);
    if (is_sell(ref.side) && best_ask_idx_ == level_idx)
      refresh_best_ask_from(level_idx + 1);
  } else {
    maybe_compact_level(level_idx, level);
  }
  ++event_count_;
  return true;
}

bool FlatLadderBook::best_bid(Tick &price, Qty &volume) const noexcept {
  if (best_bid_idx_ == kNpos) [[unlikely]] return false;
  price = price_of(best_bid_idx_);
  volume = volume_at(Side::Buy, price);
  return true;
}

bool FlatLadderBook::best_ask(Tick &price, Qty &volume) const noexcept {
  if (best_ask_idx_ == kNpos) [[unlikely]] return false;
  price = price_of(best_ask_idx_);
  volume = volume_at(Side::Sell, price);
  return true;
}

// Integer mid × 2 for the hot path. No FP, fully deterministic.
bool FlatLadderBook::mid_tick_x2(Tick &sum) const noexcept {
  if (best_bid_idx_ == kNpos || best_ask_idx_ == kNpos) [[unlikely]]
    return false;
  sum = static_cast<Tick>(price_of(best_bid_idx_) + price_of(best_ask_idx_));
  return true;
}

Qty FlatLadderBook::volume_at(Side side, Tick price) const noexcept {
  if (!in_range(price))
    return 0;
  const std::size_t idx = index_of(price);
  const Level &level = level_at(idx);
  return level_side_[idx] == side_tag(side) ? level.volume : 0;
}

bool FlatLadderBook::mid_price(double &price) const {
  Tick bb = 0;
  Tick ba = 0;
  Qty  bv = 0;
  Qty  av = 0;
  if (!best_bid(bb, bv) || !best_ask(ba, av)) return false;
  price = 0.5 * (static_cast<double>(bb) + static_cast<double>(ba));
  return true;
}

std::vector<BookLevelView> FlatLadderBook::levels(Side side,
                                                  std::size_t depth) const {
  std::vector<BookLevelView> out;
  if (side == Side::Buy) {
    const std::uint8_t tag = kBuySide;
    for (std::size_t i = best_bid_idx_; i != kNpos;) {
      const Level &level = level_at(i);
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
      const Level &level = level_at(i);
      if (!level.empty() && level_side_[i] == tag)
        out.push_back({price_of(i), level.volume, level.live_orders});
      if (depth > 0 && out.size() >= depth)
        break;
    }
  }
  return out;
}

// True when both sides are empty (no cached best on either side).
bool FlatLadderBook::is_empty() const noexcept {
  return best_bid_idx_ == kNpos && best_ask_idx_ == kNpos;
}

// O(N + sum(level.orders)). Test/debug only — never invoked on the hot path.
// Walks each level once, validating per-level qty/live counts and overall
// totals/cached bests against the running state.
void FlatLadderBook::check_invariants() const {
  Qty bid_sum = 0;
  Qty ask_sum = 0;
  std::size_t live_sum = 0;
  std::size_t observed_best_bid = kNpos; // highest bid index seen
  std::size_t observed_best_ask = kNpos; // lowest ask index seen

  for (std::size_t i = 0; i < tick_count_; ++i) {
    const Level &level = level_at(i);
    Qty qty_sum = 0;
    std::uint32_t live = 0;
    for (std::size_t k = level.head; k < level.orders.size(); ++k) {
      const Order &o = level.orders[k];
      if (o.qty > 0) {
        qty_sum += o.qty;
        ++live;
      }
    }
    if (qty_sum != level.volume)
      throw std::logic_error("flat: level volume mismatch");
    if (live != level.live_orders)
      throw std::logic_error("flat: level live_orders mismatch");

    const std::uint8_t tag = level_side_[i];
    const bool bit_set = (level_bitmap_[i / 64] & (std::uint64_t{1} << (i % 64))) != 0;
    if (qty_sum == 0) {
      if (tag != kNoSide)
        throw std::logic_error("flat: empty level with side tag set");
      if (bit_set)
        throw std::logic_error("flat: empty level with bitmap bit set");
      continue;
    }
    if (!bit_set)
      throw std::logic_error("flat: non-empty level missing bitmap bit");
    if (tag == kBuySide) {
      bid_sum += qty_sum;
      observed_best_bid = i; // last seen wins -> highest index
    } else if (tag == kSellSide) {
      ask_sum += qty_sum;
      if (observed_best_ask == kNpos) observed_best_ask = i; // first seen -> lowest
    } else {
      throw std::logic_error("flat: non-empty level missing side tag");
    }
    live_sum += live;
  }

  if (bid_sum != total_bid_vol_)
    throw std::logic_error("flat: total bid volume mismatch");
  if (ask_sum != total_ask_vol_)
    throw std::logic_error("flat: total ask volume mismatch");
  if (live_sum != live_order_count_)
    throw std::logic_error("flat: live order count mismatch");
  if (observed_best_bid != best_bid_idx_)
    throw std::logic_error("flat: best_bid_idx out of sync");
  if (observed_best_ask != best_ask_idx_)
    throw std::logic_error("flat: best_ask_idx out of sync");
  if (best_bid_idx_ != kNpos && best_ask_idx_ != kNpos &&
      best_bid_idx_ >= best_ask_idx_)
    throw std::logic_error("flat: crossed book");
}
// O(N/64) bitmap-accelerated highest-set-bit search at or below `start`.
void FlatLadderBook::refresh_best_bid_from(std::size_t start) noexcept {
  if (tick_count_ == 0 || start == kNpos) {
    best_bid_idx_ = kNpos;
    return;
  }
  const std::size_t clamped = std::min(start, tick_count_ - 1);
  std::size_t word = clamped / 64;
  const std::size_t bit_in_word = clamped % 64;
  // Mask off bits above `bit_in_word` in the starting word.
  const std::uint64_t head_mask = (bit_in_word == 63)
                                      ? ~std::uint64_t{0}
                                      : ((std::uint64_t{1} << (bit_in_word + 1)) - 1);
  std::uint64_t w = level_bitmap_[word] & head_mask;
  while (true) {
    if (w != 0) {
      const int hi = 63 - __builtin_clzll(w);
      best_bid_idx_ = word * 64 + static_cast<std::size_t>(hi);
      return;
    }
    if (word == 0) break;
    --word;
    w = level_bitmap_[word];
  }
  best_bid_idx_ = kNpos;
}

// O(N/64) bitmap-accelerated lowest-set-bit search at or above `start`.
void FlatLadderBook::refresh_best_ask_from(std::size_t start) noexcept {
  if (start >= tick_count_) {
    best_ask_idx_ = kNpos;
    return;
  }
  std::size_t word = start / 64;
  const std::size_t bit_in_word = start % 64;
  // Mask off bits below `bit_in_word` in the starting word.
  const std::uint64_t tail_mask = ~((std::uint64_t{1} << bit_in_word) - 1);
  std::uint64_t w = level_bitmap_[word] & tail_mask;
  const std::size_t nwords = level_bitmap_.size();
  while (true) {
    if (w != 0) {
      const int lo = __builtin_ctzll(w);
      best_ask_idx_ = word * 64 + static_cast<std::size_t>(lo);
      if (best_ask_idx_ >= tick_count_) [[unlikely]] {
        // Trailing bits beyond ladder — should never trigger if invariants hold.
        best_ask_idx_ = kNpos;
      }
      return;
    }
    ++word;
    if (word >= nwords) break;
    w = level_bitmap_[word];
  }
  best_ask_idx_ = kNpos;
}

void FlatLadderBook::mark_level_occupied(std::size_t idx) noexcept {
  level_bitmap_[idx / 64] |= (std::uint64_t{1} << (idx % 64));
}

void FlatLadderBook::clear_level_occupied(std::size_t idx) noexcept {
  level_bitmap_[idx / 64] &= ~(std::uint64_t{1} << (idx % 64));
}

} // namespace sim
