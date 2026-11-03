// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#include "catch2/catch_test_macros.hpp"
#define QRLOB_NO_GLOBAL_NEW_OVERRIDE 1
#include "qr/alloc_guard.hpp"
#include "sim/sim.hpp"
#include "sim/bitwise_veb_codings.hpp"
#include "sim/veb_static_search.hpp"
#include "sim/ofi_sparsity_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using namespace sim;

TEST_CASE("Pillar 4.1: FlatLadderBook::levels() O(1) bitmap-accelerated sparse traversal", "[pillar4][levels]") {
  FlatLadderBook book(100000, 300000, 2048);
  REQUIRE(book.is_valid());

  // Empty book edge cases
  REQUIRE(book.levels(Side::Buy).empty());
  REQUIRE(book.levels(Side::Sell).empty());
  REQUIRE(book.levels(Side::Buy, 5).empty());
  REQUIRE(book.levels(Side::Sell, 5).empty());

  // Single bid and single ask across 200,000-tick sparse universe
  Order ob1{}; ob1.agent = 1; ob1.side = Side::Buy; ob1.type = OrderType::Limit; ob1.price = 199500; ob1.qty = 25;
  Order oa1{}; oa1.agent = 2; oa1.side = Side::Sell; oa1.type = OrderType::Limit; oa1.price = 200500; oa1.qty = 40;
  const auto b1 = book.add(ob1, 10);
  const auto a1 = book.add(oa1, 20);
  (void)b1;
  (void)a1;

  auto bids_single = book.levels(Side::Buy);
  auto asks_single = book.levels(Side::Sell);
  REQUIRE(bids_single.size() == 1);
  REQUIRE(bids_single[0].price == 199500);
  REQUIRE(bids_single[0].volume == 25);
  REQUIRE(bids_single[0].orders == 1);
  REQUIRE(asks_single.size() == 1);
  REQUIRE(asks_single[0].price == 200500);
  REQUIRE(asks_single[0].volume == 40);
  REQUIRE(asks_single[0].orders == 1);

  // Populate multiple sparse levels across distant 64-word & 4096-summary boundaries
  const std::vector<Tick> bid_ticks = {100000, 100001, 100063, 100064, 104095, 104096, 150000, 199500};
  const std::vector<Tick> ask_ticks = {200500, 210000, 240959, 240960, 280000, 299935, 299936, 300000};

  TimeNs ts = 100;
  for (Tick t : bid_ticks) {
    if (t != 199500) {
      { Order o{}; o.agent = 1; o.side = Side::Buy; o.type = OrderType::Limit; o.price = t; o.qty = 10; book.add(o, ts++); }
    }
  }
  for (Tick t : ask_ticks) {
    if (t != 200500) {
      { Order o{}; o.agent = 2; o.side = Side::Sell; o.type = OrderType::Limit; o.price = t; o.qty = 15; book.add(o, ts++); }
    }
  }

  // Full depth check: bids descending from best_bid, asks ascending from best_ask
  const auto all_bids = book.levels(Side::Buy, 0);
  const auto all_asks = book.levels(Side::Sell, 0);
  REQUIRE(all_bids.size() == bid_ticks.size());
  REQUIRE(all_asks.size() == ask_ticks.size());

  for (std::size_t i = 0; i < bid_ticks.size(); ++i) {
    REQUIRE(all_bids[i].price == bid_ticks[bid_ticks.size() - 1 - i]);
  }
  for (std::size_t i = 0; i < ask_ticks.size(); ++i) {
    REQUIRE(all_asks[i].price == ask_ticks[i]);
  }

  // Truncated depth check
  const auto top3_bids = book.levels(Side::Buy, 3);
  const auto top3_asks = book.levels(Side::Sell, 3);
  REQUIRE(top3_bids.size() == 3);
  REQUIRE(top3_asks.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(top3_bids[i].price == all_bids[i].price);
    REQUIRE(top3_asks[i].price == all_asks[i].price);
  }
  REQUIRE(book.check_invariants_report().ok);
}

TEST_CASE("Pillar 4.2: BitwiseVeb4096 and SparseBitwiseVeb16M O(1) operations & F2 XOR rollback", "[pillar4][veb]") {
  BitwiseVeb4096 veb{};
  REQUIRE(veb.empty());
  REQUIRE(veb.min_tick() == kVebNullTick);
  REQUIRE(veb.max_tick() == kVebNullTick);
  REQUIRE(veb.next_higher(0) == kVebNullTick);
  REQUIRE(veb.prev_lower(4095) == kVebNullTick);

  // Out-of-bounds safety
  veb.insert(4096);
  veb.insert(999999);
  REQUIRE(veb.empty());
  REQUIRE(!veb.contains(4096));

  std::set<std::uint32_t> oracle;
  Xoshiro256StarStar rng(0xBEEF1234ULL);
  for (int i = 0; i < 400; ++i) {
    const std::uint32_t t = static_cast<std::uint32_t>(rng.next_u64() % 4096ULL);
    veb.insert(t);
    oracle.insert(t);
  }
  // Explicitly include boundaries
  for (std::uint32_t b : {0U, 1U, 63U, 64U, 65U, 2047U, 2048U, 4031U, 4032U, 4094U, 4095U}) {
    veb.insert(b);
    oracle.insert(b);
  }

  REQUIRE(!veb.empty());
  REQUIRE(veb.min_tick() == *oracle.begin());
  REQUIRE(veb.max_tick() == *oracle.rbegin());

  for (std::uint32_t q = 0; q < 4096; ++q) {
    REQUIRE(veb.contains(q) == (oracle.count(q) > 0));
    auto it_hi = oracle.upper_bound(q);
    const std::uint32_t exp_hi = (it_hi == oracle.end()) ? kVebNullTick : *it_hi;
    REQUIRE(veb.next_higher(q) == exp_hi);

    auto it_lo = oracle.lower_bound(q);
    const std::uint32_t exp_lo = (it_lo == oracle.begin()) ? kVebNullTick : *std::prev(it_lo);
    REQUIRE(veb.prev_lower(q) == exp_lo);
  }

  // Test ReversibleVebDelta F2 XOR rollback law: old ^ delta ^ delta == old
  const BitwiseVeb4096 snapshot = veb;
  std::vector<ReversibleVebDelta> deltas;
  for (int i = 0; i < 128; ++i) {
    const std::uint32_t t = static_cast<std::uint32_t>(rng.next_u64() % 4096ULL);
    if ((i & 1) == 0) {
      deltas.push_back(veb.insert_reversible(t));
    } else {
      deltas.push_back(veb.remove_reversible(t));
    }
  }
  for (auto it = deltas.rbegin(); it != deltas.rend(); ++it) {
    veb.rollback_delta(*it);
  }
  REQUIRE(veb.root == snapshot.root);
  for (std::size_t c = 0; c < 64; ++c) {
    REQUIRE(veb.l2[c] == snapshot.l2[c]);
  }

  // Remove all elements and verify empty
  for (std::uint32_t t : oracle) {
    veb.remove(t);
  }
  REQUIRE(veb.empty());
  REQUIRE(veb.min_tick() == kVebNullTick);
  REQUIRE(veb.max_tick() == kVebNullTick);

  // 4-Level SparseBitwiseVeb16M across 16,777,216 ticks
  SparseBitwiseVeb16M veb16m;
  REQUIRE(veb16m.empty());
  REQUIRE(veb16m.allocated_pages() == 0);

  const std::vector<std::uint32_t> wide_ticks = {
      0U, 1U, 63U, 64U, 4095U, 4096U, 65535U, 65536U,
      1234567U, 8388607U, 8388608U, 16773119U, 16773120U, 16777214U, 16777215U};
  for (std::uint32_t t : wide_ticks) {
    veb16m.insert(t);
  }
  REQUIRE(!veb16m.empty());
  REQUIRE(veb16m.min_tick() == wide_ticks.front());
  REQUIRE(veb16m.max_tick() == wide_ticks.back());

  for (std::size_t i = 0; i < wide_ticks.size(); ++i) {
    const std::uint32_t t = wide_ticks[i];
    REQUIRE(veb16m.contains(t));
    const std::uint32_t exp_next = (i + 1 < wide_ticks.size()) ? wide_ticks[i + 1] : kVebNullTick;
    const std::uint32_t exp_prev = (i > 0) ? wide_ticks[i - 1] : kVebNullTick;
    REQUIRE(veb16m.next_higher(t) == exp_next);
    REQUIRE(veb16m.prev_lower(t) == exp_prev);
  }
  for (std::uint32_t t : wide_ticks) {
    veb16m.remove(t);
    REQUIRE(!veb16m.contains(t));
  }
  REQUIRE(veb16m.empty());
  REQUIRE(veb16m.min_tick() == kVebNullTick);
  REQUIRE(veb16m.max_tick() == kVebNullTick);
}

TEST_CASE("Pillar 4.3: SWAR Queue-Reactive LUTs, branchless bit_mux, and Q48.16 Horner Hawkes decay", "[pillar4][swar][hawkes]") {
  // Verify SWAR 4-bit nibble lookup tables
  const std::uint8_t exp_add[8]    = {2, 3, 4, 5, 6, 8, 5, 5};
  const std::uint8_t exp_cancel[8] = {0, 1, 2, 2, 3, 4, 0, 4};
  const std::uint8_t exp_exec[8]   = {0, 0, 15, 4, 15, 9, 15, 15};
  for (std::uint32_t r = 0; r < 8; ++r) {
    REQUIRE(SwarQueueReactiveRegimeLut::decode_add(r) == exp_add[r]);
    REQUIRE(SwarQueueReactiveRegimeLut::decode_cancel(r) == exp_cancel[r]);
    REQUIRE(SwarQueueReactiveRegimeLut::decode_exec(r) == exp_exec[r]);
  }

  // Branchless bit_mux and is_zero_mask
  REQUIRE(is_zero_mask(0ULL) == ~0ULL);
  REQUIRE(is_zero_mask(1ULL) == 0ULL);
  REQUIRE(is_zero_mask(0x8000000000000000ULL) == 0ULL);
  REQUIRE(bit_mux(~0ULL, 0xAAAA5555ULL, 0x12345678ULL) == 0xAAAA5555ULL);
  REQUIRE(bit_mux(0ULL, 0xAAAA5555ULL, 0x12345678ULL) == 0x12345678ULL);
  REQUIRE(verified_cfr_branchless_relu64(-42) == 0);
  REQUIRE(verified_cfr_branchless_relu64(0) == 0);
  REQUIRE(verified_cfr_branchless_relu64(12345) == 12345);

  // Verify Horner Q48.16 exponential decay < 0.15% max relative error on [0, 4*ln(2)]
  REQUIRE(eval_hawkes_q16_horner_decay(0ULL) == 65536ULL);
  const double max_x = 4.0 * 0.6931471805599453;
  double max_rel_err = 0.0;
  for (int step = 0; step <= 2000; ++step) {
    const double x = max_x * (static_cast<double>(step) / 2000.0);
    const auto x_q16 = static_cast<std::uint64_t>(std::llround(x * 65536.0));
    const std::uint64_t approx_q16 = eval_hawkes_q16_horner_decay(x_q16);
    const double approx = static_cast<double>(approx_q16) / 65536.0;
    const double exact = std::exp(-static_cast<double>(x_q16) / 65536.0);
    const double rel_err = std::abs(approx - exact) / exact;
    if (rel_err > max_rel_err) max_rel_err = rel_err;
  }
  REQUIRE(max_rel_err < 0.0015);

  // OnlineHawkesQ16 deterministic fixed-point intensity evolution
  OnlineHawkesQ16 hawkes(65536ULL, 32768ULL, 65536ULL); // mu=1.0, alpha=0.5, beta=1.0
  REQUIRE(hawkes.intensity_q16() == 65536ULL);
  const std::uint64_t after_jump = hawkes.record_event(0ULL);
  REQUIRE(after_jump == 65536ULL + 32768ULL);
  const std::uint64_t decayed = hawkes.decay_dt(65536ULL); // dt = 1.0s
  const double expected_lambda = 1.0 + 0.5 * std::exp(-1.0);
  const double actual_lambda = OnlineHawkesQ16::q16_to_double(decayed);
  REQUIRE(std::abs(actual_lambda - expected_lambda) < 0.002);
}

TEST_CASE("Pillar 4.4: ReversibleSubgameScope 1,000-step L3 rollout + zero-allocation bit-exact rollback", "[pillar4][reversible]") {
  FlatLadderBook book(9000, 11000, 8192);
  FlatLadderBook oracle(9000, 11000, 8192);
  Xoshiro256StarStar rng(0xCAFEBABE12345678ULL);

  std::vector<OrderId> base_orders;
  TimeNs ts = 1000;
  for (int i = 0; i < 150; ++i) {
    const Side side = (i & 1) == 0 ? Side::Buy : Side::Sell;
    const Tick price = (side == Side::Buy)
                           ? static_cast<Tick>(9900 + static_cast<int>(rng.next_u64() % 95ULL))
                           : static_cast<Tick>(10005 + static_cast<int>(rng.next_u64() % 95ULL));
    const Qty qty = static_cast<Qty>(1 + (rng.next_u64() % 25ULL));
    Order o{}; o.agent = static_cast<AgentId>(1 + (i % 7)); o.side = side; o.type = OrderType::Limit; o.price = price; o.qty = qty;
    Trade buf[8];
    const auto s1 = book.add_into(o, ts, buf, 8);
    const auto s2 = oracle.add_into(o, ts, buf, 8);
    REQUIRE(s1.id == s2.id);
    if (s1.remaining > 0) base_orders.push_back(s1.id);
    ts += 10;
  }

  REQUIRE(book.check_invariants_report().ok);
  REQUIRE(book.state_equals(oracle));
  const std::uint64_t digest_before = book.state_digest();

  // Pre-allocate undo log outside the zero-allocation guard
  LadderUndoLog undo_log;
  Trade trade_buf[16];
  std::vector<OrderId> rollout_orders = base_orders;
  rollout_orders.reserve(base_orders.size() + 2048);

  {
    sim::diag::ScopedAllocCounter alloc_guard;
    {
      ReversibleSubgameScope scope(book, undo_log);

      for (int step = 0; step < 1000; ++step) {
        const std::uint64_t r = rng.next_u64() % 100ULL;
        if (r < 55ULL) {
          // Resting or crossing limit order
          const Side side = ((rng.next_u64() & 1ULL) == 0ULL) ? Side::Buy : Side::Sell;
          const Tick price = (side == Side::Buy)
                                 ? static_cast<Tick>(9850 + static_cast<int>(rng.next_u64() % 170ULL))
                                 : static_cast<Tick>(9980 + static_cast<int>(rng.next_u64() % 170ULL));
          const Qty qty = static_cast<Qty>(1 + (rng.next_u64() % 30ULL));
          Order o{}; o.agent = static_cast<AgentId>(10 + (step % 5)); o.side = side; o.type = OrderType::Limit; o.price = price; o.qty = qty;
          const MatchSummary ms = scope.add(o, ts++, trade_buf, 16);
          if (ms.remaining > 0) rollout_orders.push_back(ms.id);
        } else if (r < 80ULL && !rollout_orders.empty()) {
          // Cancel random order
          const std::size_t pick = static_cast<std::size_t>(rng.next_u64() % rollout_orders.size());
          const OrderId cid = rollout_orders[pick];
          scope.cancel(cid, ts++);
        } else {
          // Market execution
          const Side side = ((rng.next_u64() & 1ULL) == 0ULL) ? Side::Buy : Side::Sell;
          const Qty qty = static_cast<Qty>(1 + (rng.next_u64() % 20ULL));
          scope.execute(side, qty, ts++, 99, trade_buf, 16);
        }
      }
      REQUIRE(!scope.overflowed());
      REQUIRE(scope.delta_count() > 0);
      scope.rollback();
    }
    REQUIRE(alloc_guard.delta().allocations == 0);
  }

  REQUIRE(book.check_invariants_report().ok);
  REQUIRE(book.state_digest() == digest_before);
  REQUIRE(book.state_equals(oracle));
}

TEST_CASE("Pillar 4.5: EytzingerTickGrid branchless lower_bound parity across N in [1, 4096]", "[pillar4][eytzinger]") {
  EytzingerTickGrid<std::int64_t> empty_grid;
  REQUIRE(empty_grid.empty());
  REQUIRE(empty_grid.lower_bound_rank(100) == 0);
  REQUIRE(empty_grid.lower_bound_ptr(100) == nullptr);

  // Exhaustive check across key sizes including powers of 2, Mersenne sizes, and arbitrary sizes up to 4096
  const std::vector<std::size_t> test_sizes = {
      1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
      127, 128, 129, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 3000, 4095, 4096};

  for (std::size_t n : test_sizes) {
    std::vector<std::int64_t> sorted(n);
    for (std::size_t i = 0; i < n; ++i) {
      sorted[i] = static_cast<std::int64_t>(1000 + i * 5 + (i % 3));
    }
    EytzingerTickGrid<std::int64_t> grid(sorted);
    REQUIRE(grid.size() == n);

    // Test below minimum, exact hits, between-element gaps, and above maximum
    REQUIRE(grid.lower_bound_rank(sorted.front() - 10) == 0);
    REQUIRE(grid.lower_bound_rank(sorted.back() + 10) == n);
    REQUIRE(grid.lower_bound_ptr(sorted.back() + 10) == nullptr);

    for (std::size_t i = 0; i < n; ++i) {
      const std::int64_t key = sorted[i];
      REQUIRE(grid.lower_bound_rank(key) == i);
      REQUIRE(grid.contains(key));
      const auto* ptr = grid.lower_bound_ptr(key);
      REQUIRE(ptr != nullptr);
      REQUIRE(*ptr == key);

      const std::int64_t probe = key - 1;
      const auto it = std::lower_bound(sorted.begin(), sorted.end(), probe);
      const auto exp_rank = static_cast<std::size_t>(std::distance(sorted.begin(), it));
      REQUIRE(grid.lower_bound_rank(probe) == exp_rank);
    }
  }
}

TEST_CASE("Pillar 4.6: AVX2 Multi-Level OFI over TopKBookSnapshot<8> parity vs scalar", "[pillar4][ofi][avx2]") {
  Xoshiro256StarStar rng(0x9876543210ABCDEFULL);
  const auto custom_weights = default_ofi_decay_weights<8>(0.35);

  for (int trial = 0; trial < 200; ++trial) {
    TopKBookSnapshot<8> prev{};
    TopKBookSnapshot<8> curr{};
    prev.ts = 1000 + trial * 10;
    curr.ts = prev.ts + 5; prev.bid_count = 8; prev.ask_count = 8; curr.bid_count = 8; curr.ask_count = 8;

    for (std::size_t m = 0; m < 8; ++m) {
      prev.bids[m].price = static_cast<Tick>(10000 - static_cast<int>(m) * 2 - static_cast<int>(rng.next_u64() % 2ULL));
      prev.bids[m].qty = static_cast<Qty>(10 + (rng.next_u64() % 500ULL));
      prev.asks[m].price = static_cast<Tick>(10010 + static_cast<int>(m) * 2 + static_cast<int>(rng.next_u64() % 2ULL));
      prev.asks[m].qty = static_cast<Qty>(10 + (rng.next_u64() % 500ULL));

      // Exercise all three price transition branches: p_t > p_{t-1}, p_t == p_{t-1}, p_t < p_{t-1}
      const int delta_bp = static_cast<int>(rng.next_u64() % 3ULL) - 1;
      const int delta_ap = static_cast<int>(rng.next_u64() % 3ULL) - 1;
      curr.bids[m].price = prev.bids[m].price + delta_bp;
      curr.bids[m].qty = static_cast<Qty>(10 + (rng.next_u64() % 500ULL));
      curr.asks[m].price = prev.asks[m].price + delta_ap;
      curr.asks[m].qty = static_cast<Qty>(10 + (rng.next_u64() % 500ULL));
    }

    const auto scalar_out = compute_multilevel_ofi_scalar<8>(prev, curr, custom_weights);
    const auto avx2_out = compute_multilevel_ofi_avx2<8>(prev, curr, custom_weights);

    for (std::size_t m = 0; m < 8; ++m) {
      REQUIRE(avx2_out.level_ofi[m] == scalar_out.level_ofi[m]);
    }
    REQUIRE(avx2_out.l1_ofi == scalar_out.l1_ofi);
    REQUIRE(std::abs(avx2_out.weighted_ofi - scalar_out.weighted_ofi) < 1e-9);
    REQUIRE(std::abs(avx2_out.normalized_ofi - scalar_out.normalized_ofi) < 1e-9);

    const double integrated = compute_integrated_ofi_avx2<8>(prev, curr, custom_weights);
    REQUIRE(std::abs(integrated - scalar_out.weighted_ofi) < 1e-9);
  }
}

TEST_CASE("Pillar 4.4b: Nested ReversibleSubgameScope (depth >= 2), STP DecrementAndCancel, and free-list reuse", "[pillar4][reversible][nested]") {
  // 1. Nested scopes (both shared undo_log and default local_log) with inner rollback + outer continue + outer rollback
  {
    FlatLadderBook book(9900, 10100, 1024);
    FlatLadderBook oracle(9900, 10100, 1024);
    Order o1{}; o1.agent = 1; o1.side = Side::Buy; o1.type = OrderType::Limit; o1.price = 9990; o1.qty = 10;
    book.add_into(o1, 1, nullptr, 0);
    oracle.add_into(o1, 1, nullptr, 0);

    LadderUndoLog shared_log;
    {
      ReversibleSubgameScope outer(book, shared_log);
      Order o2{}; o2.agent = 2; o2.side = Side::Sell; o2.type = OrderType::Limit; o2.price = 10010; o2.qty = 20;
      outer.add(o2, 2);
      const std::uint64_t mid_digest = book.state_digest();

      {
        ReversibleSubgameScope inner(book, shared_log);
        Order o3{}; o3.agent = 3; o3.side = Side::Buy; o3.type = OrderType::Limit; o3.price = 10010; o3.qty = 5;
        inner.add(o3, 3);
        inner.rollback();
      }
      REQUIRE(book.state_digest() == mid_digest);
      REQUIRE(book.check_invariants_report().ok);

      // Outer continues after inner rollback, plus a committed inner scope
      {
        ReversibleSubgameScope inner_commit(book);
        Order o4{}; o4.agent = 4; o4.side = Side::Buy; o4.type = OrderType::Limit; o4.price = 9995; o4.qty = 15;
        inner_commit.add(o4, 4);
        inner_commit.commit();
      }
      outer.rollback();
    }
    REQUIRE(book.state_equals(oracle));
    REQUIRE(book.check_invariants_report().ok);
  }

  // 2. Free-list multi-slot reuse before, inside, and after ReversibleSubgameScope rollback
  {
    FlatLadderBook book(9900, 10100, 1024);
    FlatLadderBook oracle(9900, 10100, 1024);
    Order o1{}; o1.agent = 1; o1.side = Side::Buy; o1.type = OrderType::Limit; o1.price = 9990; o1.qty = 10;
    Order o2{}; o2.agent = 1; o2.side = Side::Buy; o2.type = OrderType::Limit; o2.price = 9991; o2.qty = 15;
    auto s1 = book.add_into(o1, 1, nullptr, 0); oracle.add_into(o1, 1, nullptr, 0);
    auto s2 = book.add_into(o2, 2, nullptr, 0); oracle.add_into(o2, 2, nullptr, 0);
    book.cancel(s1.id, 3); oracle.cancel(s1.id, 3);
    book.cancel(s2.id, 4); oracle.cancel(s2.id, 4);

    {
      ReversibleSubgameScope scope(book);
      scope.add(o1, 5);
      scope.add(o2, 6);
      scope.rollback();
    }
    REQUIRE(book.state_equals(oracle));
    book.add_into(o1, 10, nullptr, 0); oracle.add_into(o1, 10, nullptr, 0);
    book.add_into(o2, 11, nullptr, 0); oracle.add_into(o2, 11, nullptr, 0);
    REQUIRE(book.state_equals(oracle));
    REQUIRE(book.check_invariants_report().ok);
  }

  // 3. StpMode::DecrementAndCancel inside ReversibleSubgameScope (unlink + partial decrement at same level)
  {
    FlatLadderBook book(9900, 10100, 1024);
    FlatLadderBook oracle(9900, 10100, 1024);
    Order m1{}; m1.agent = 7; m1.side = Side::Sell; m1.type = OrderType::Limit; m1.price = 10000; m1.qty = 10;
    Order m2{}; m2.agent = 7; m2.side = Side::Sell; m2.type = OrderType::Limit; m2.price = 10000; m2.qty = 20;
    Order m3{}; m3.agent = 8; m3.side = Side::Sell; m3.type = OrderType::Limit; m3.price = 10000; m3.qty = 15;
    book.add_into(m1, 1, nullptr, 0); oracle.add_into(m1, 1, nullptr, 0);
    book.add_into(m2, 2, nullptr, 0); oracle.add_into(m2, 2, nullptr, 0);
    book.add_into(m3, 3, nullptr, 0); oracle.add_into(m3, 3, nullptr, 0);

    {
      ReversibleSubgameScope scope(book);
      Order taker{}; taker.agent = 7; taker.side = Side::Buy; taker.type = OrderType::Limit; taker.price = 10000; taker.qty = 16;
      Trade tbuf[4];
      const auto ms = scope.add(taker, 4, tbuf, 4, StpMode::DecrementAndCancel);
      REQUIRE(ms.filled == 0);
      REQUIRE(book.volume_at(Side::Sell, 10000) == 29); // m1 unlinked (-10), m2 decremented by 6 (14 left) + m3 (15) = 29
      scope.rollback();
    }
    REQUIRE(book.state_equals(oracle));
    REQUIRE(book.check_invariants_report().ok);
  }

  // 4. Verify 4-arg verified_veb3_4096_encode and SparseBitwiseVeb16M::reserve_page zero-alloc
  REQUIRE(verified_veb3_4096_encode(1ULL, 2ULL, 3ULL, 4ULL) == ((1ULL << 36ULL) | (2ULL << 24ULL) | (3ULL << 12ULL) | 4ULL));
  SparseBitwiseVeb16M veb16m;
  REQUIRE(veb16m.reserve_page(100U));
  {
    sim::diag::ScopedAllocCounter guard;
    veb16m.insert((100U << 12U) | 42U);
    REQUIRE(veb16m.contains((100U << 12U) | 42U));
    veb16m.remove((100U << 12U) | 42U);
    REQUIRE(guard.delta().allocations == 0);
  }
}

TEST_CASE("Pillar 4.7: Register-Packed MatchOutcome (16B) and CancelOutcome (8B) FOOP contracts", "[pillar4][foop][outcome]") {
  static_assert(sizeof(MatchOutcome) == 16);
  static_assert(alignof(MatchOutcome) == 16);
  static_assert(sizeof(CancelOutcome) == 8);
  static_assert(alignof(CancelOutcome) == 8);
  static_assert(sizeof(CompactDeltaToken) == 32);
  static_assert(sizeof(ReversibleSubgameScope) <= 32);

  FlatLadderBook flat(100, 200, 1024);
  LimitOrderBook tree(1024);
  Trade tbuf[8];

  // 1. RejectedInvalid (qty <= 0 or invalid book)
  {
    Order bad{}; bad.agent = 1; bad.side = Side::Buy; bad.type = OrderType::Limit; bad.price = 150; bad.qty = 0;
    const MatchOutcome m_flat = flat.add_outcome(bad, 1);
    const MatchOutcome m_tree = tree.add_outcome(bad);
    REQUIRE(m_flat.status == OrderStatus::RejectedInvalid);
    REQUIRE(m_flat.is_rejected());
    REQUIRE_FALSE(m_flat.is_accepted());
    REQUIRE(m_flat.remaining(10) == 0);
    REQUIRE(m_tree.status == OrderStatus::RejectedInvalid);

    FlatLadderBook invalid_book(200, 100);
    Order good{}; good.agent = 1; good.side = Side::Buy; good.type = OrderType::Limit; good.price = 150; good.qty = 10;
    const MatchOutcome m_inv = invalid_book.add_outcome(good, 1);
    REQUIRE(m_inv.status == OrderStatus::RejectedInvalid);
  }

  // 2. RejectedOutOfRange on out-of-bounds Limit and PostOnly orders (fixes flat_ladder_book.cpp:409 bug!)
  {
    Order oob_limit{}; oob_limit.agent = 1; oob_limit.side = Side::Buy; oob_limit.type = OrderType::Limit; oob_limit.price = 95; oob_limit.qty = 25;
    const MatchOutcome mo = flat.add_outcome(oob_limit, 2);
    REQUIRE(mo.status == OrderStatus::RejectedOutOfRange);
    REQUIRE(mo.filled == 0);
    REQUIRE(mo.remaining(25) == 0);
    REQUIRE(mo.is_rejected());
    REQUIRE_FALSE(mo.has_rested());

    const MatchSummary ms = flat.add_into(oob_limit, 3, tbuf, 8);
    REQUIRE(ms.status == OrderStatus::RejectedOutOfRange);
    REQUIRE(ms.filled == 0);
    REQUIRE(ms.remaining == 0);
    REQUIRE(flat.order_count() == 0);
  }

  // 3. AcceptedRested
  Order sell_rest{}; sell_rest.agent = 2; sell_rest.side = Side::Sell; sell_rest.type = OrderType::Limit; sell_rest.price = 150; sell_rest.qty = 40;
  const MatchOutcome mo_rest_flat = flat.add_outcome(sell_rest, 10);
  const MatchOutcome mo_rest_tree = tree.add_outcome(sell_rest);
  REQUIRE(mo_rest_flat.status == OrderStatus::AcceptedRested);
  REQUIRE(mo_rest_flat.has_rested());
  REQUIRE(mo_rest_flat.remaining(40) == 40);
  REQUIRE(mo_rest_tree.status == OrderStatus::AcceptedRested);

  // 4. RejectedPostOnly & AcceptedRested PostOnly
  {
    Order po_cross{}; po_cross.agent = 3; po_cross.side = Side::Buy; po_cross.type = OrderType::PostOnly; po_cross.price = 150; po_cross.qty = 10;
    REQUIRE(flat.add_outcome(po_cross, 11).status == OrderStatus::RejectedPostOnly);
    REQUIRE(tree.add_outcome(po_cross).status == OrderStatus::RejectedPostOnly);

    Order po_passive{}; po_passive.agent = 3; po_passive.side = Side::Buy; po_passive.type = OrderType::PostOnly; po_passive.price = 140; po_passive.qty = 10;
    const auto po_ok_f = flat.add_outcome(po_passive, 12);
    const auto po_ok_t = tree.add_outcome(po_passive);
    REQUIRE(po_ok_f.status == OrderStatus::AcceptedRested);
    REQUIRE(po_ok_t.status == OrderStatus::AcceptedRested);
    REQUIRE(flat.cancel_outcome(po_ok_f.id, 13).status == CancelStatus::Cancelled);
    REQUIRE(tree.cancel_outcome(po_ok_t.id).status == CancelStatus::Cancelled);
  }

  // 5. RejectedFok
  {
    Order fok_big{}; fok_big.agent = 3; fok_big.side = Side::Buy; fok_big.type = OrderType::FOK; fok_big.price = 150; fok_big.qty = 100;
    REQUIRE(flat.add_outcome(fok_big, 14).status == OrderStatus::RejectedFok);
    REQUIRE(tree.add_outcome(fok_big).status == OrderStatus::RejectedFok);
  }

  // 6. ExpiredIocZero & FilledPartialIoc
  {
    Order ioc_miss{}; ioc_miss.agent = 3; ioc_miss.side = Side::Buy; ioc_miss.type = OrderType::IOC; ioc_miss.price = 145; ioc_miss.qty = 10;
    REQUIRE(flat.add_outcome(ioc_miss, 15).status == OrderStatus::ExpiredIocZero);
    REQUIRE(tree.add_outcome(ioc_miss).status == OrderStatus::ExpiredIocZero);

    Order ioc_part{}; ioc_part.agent = 3; ioc_part.side = Side::Buy; ioc_part.type = OrderType::IOC; ioc_part.price = 150; ioc_part.qty = 60;
    // Wait: let us add another 30 at 155 first so 150 only has 40
    Order ask_155{}; ask_155.agent = 2; ask_155.side = Side::Sell; ask_155.type = OrderType::Limit; ask_155.price = 155; ask_155.qty = 30;
    flat.add_outcome(ask_155, 16);
    tree.add_outcome(ask_155);

    const auto ioc_f = flat.add_outcome_into(ioc_part, 17, tbuf, 8);
    const auto ioc_t = tree.add_outcome_into(ioc_part, tbuf, 8);
    REQUIRE(ioc_f.status == OrderStatus::FilledPartialIoc);
    REQUIRE(ioc_f.filled == 40);
    REQUIRE(ioc_f.trade_count == 1);
    REQUIRE(ioc_f.remaining(60) == 0);
    REQUIRE(ioc_t.status == OrderStatus::FilledPartialIoc);
    REQUIRE(ioc_t.filled == 40);
  }

  // 7. FilledPartialRest & FilledComplete
  {
    // Currently ask_155 has 30 at 155. Submit Limit Buy at 155 for 50 -> fills 30, rests 20 at 155!
    Order lim_part{}; lim_part.agent = 4; lim_part.side = Side::Buy; lim_part.type = OrderType::Limit; lim_part.price = 155; lim_part.qty = 50;
    const auto lp_f = flat.add_outcome_into(lim_part, 18, tbuf, 8);
    const auto lp_t = tree.add_outcome_into(lim_part, tbuf, 8);
    REQUIRE(lp_f.status == OrderStatus::FilledPartialRest);
    REQUIRE(lp_f.filled == 30);
    REQUIRE(lp_f.remaining(50) == 20);
    REQUIRE(lp_t.status == OrderStatus::FilledPartialRest);
    REQUIRE(lp_t.remaining(50) == 20);

    // Now Sell 20 at 155 -> FilledComplete!
    Order sell_comp{}; sell_comp.agent = 5; sell_comp.side = Side::Sell; sell_comp.type = OrderType::Limit; sell_comp.price = 155; sell_comp.qty = 20;
    const auto sc_f = flat.add_outcome_into(sell_comp, 19, tbuf, 8);
    const auto sc_t = tree.add_outcome_into(sell_comp, tbuf, 8);
    REQUIRE(sc_f.status == OrderStatus::FilledComplete);
    REQUIRE(sc_f.filled == 20);
    REQUIRE(sc_f.remaining(20) == 0);
    REQUIRE(sc_t.status == OrderStatus::FilledComplete);
  }

  // 8. CancelledByStp (CancelAggressive and DecrementAndCancel)
  {
    Order self_ask{}; self_ask.agent = 77; self_ask.side = Side::Sell; self_ask.type = OrderType::Limit; self_ask.price = 160; self_ask.qty = 25;
    flat.add_outcome(self_ask, 20);
    tree.add_outcome(self_ask);

    Order self_buy{}; self_buy.agent = 77; self_buy.side = Side::Buy; self_buy.type = OrderType::Limit; self_buy.price = 160; self_buy.qty = 25;
    const auto stp_f = flat.add_outcome(self_buy, 21, StpMode::CancelAggressive);
    const auto stp_t = tree.add_outcome(self_buy, StpMode::CancelAggressive);
    REQUIRE(stp_f.status == OrderStatus::CancelledByStp);
    REQUIRE(stp_f.stp_cancels == 1);
    REQUIRE(stp_f.filled == 0);
    REQUIRE(stp_t.status == OrderStatus::CancelledByStp);
    REQUIRE(stp_t.stp_cancels == 1);
  }

  // 9. CancelOutcome: Cancelled, AlreadyInactive, NotFound
  {
    Order rest_bid{}; rest_bid.agent = 9; rest_bid.side = Side::Buy; rest_bid.type = OrderType::Limit; rest_bid.price = 120; rest_bid.qty = 33;
    const auto rb_f = flat.add_outcome(rest_bid, 30);
    const auto rb_t = tree.add_outcome(rest_bid);

    const CancelOutcome c1_f = flat.cancel_outcome(rb_f.id, 31);
    const CancelOutcome c1_t = tree.cancel_outcome(rb_t.id);
    REQUIRE(static_cast<bool>(c1_f));
    REQUIRE(c1_f.status == CancelStatus::Cancelled);
    REQUIRE(c1_f.cancelled_qty == 33);
    REQUIRE(c1_f.side == Side::Buy);
    REQUIRE(c1_f.price_offset == 20); // 120 - 100
    REQUIRE(static_cast<bool>(c1_t));
    REQUIRE(c1_t.status == CancelStatus::Cancelled);
    REQUIRE(c1_t.cancelled_qty == 33);

    // Second cancel of same OrderId -> AlreadyInactive
    const CancelOutcome c2_f = flat.cancel_outcome(rb_f.id, 32);
    const CancelOutcome c2_t = tree.cancel_outcome(rb_t.id);
    REQUIRE_FALSE(static_cast<bool>(c2_f));
    REQUIRE(c2_f.status == CancelStatus::AlreadyInactive);
    REQUIRE_FALSE(static_cast<bool>(c2_t));
    REQUIRE(c2_t.status == CancelStatus::AlreadyInactive);

    // Cancel of never-issued OrderId -> NotFound
    const CancelOutcome c3_f = flat.cancel_outcome(999999ULL, 33);
    const CancelOutcome c3_t = tree.cancel_outcome(999999ULL);
    REQUIRE_FALSE(static_cast<bool>(c3_f));
    REQUIRE(c3_f.status == CancelStatus::NotFound);
    REQUIRE_FALSE(static_cast<bool>(c3_t));
    REQUIRE(c3_t.status == CancelStatus::NotFound);
  }
}

TEST_CASE("Pillar 4.8: Zero-Allocation levels_into, make_top_k_snapshot<K>, and QueueReactiveProcess Q16 path", "[pillar4][levels_into][q16]") {
  FlatLadderBook flat(1000, 2000, 1024);
  LimitOrderBook tree(1024);

  for (int i = 0; i < 12; ++i) {
    Order b{}; b.agent = 1; b.side = Side::Buy; b.type = OrderType::Limit; b.price = 1400 - i * 3; b.qty = 10 + i;
    Order a{}; a.agent = 2; a.side = Side::Sell; a.type = OrderType::Limit; a.price = 1410 + i * 3; a.qty = 15 + i;
    flat.add_outcome(b, i * 2 + 1);
    flat.add_outcome(a, i * 2 + 2);
    tree.add_outcome(b);
    tree.add_outcome(a);
  }

  BookLevelView f_bids[8]{};
  BookLevelView f_asks[8]{};
  BookLevelView t_bids[8]{};
  BookLevelView t_asks[8]{};
  TopKBookSnapshot<8> snap{};

  {
    sim::diag::ScopedAllocCounter guard;
    const std::size_t nb_f = flat.levels_into(Side::Buy, f_bids, 8);
    const std::size_t na_f = flat.levels_into(Side::Sell, f_asks, 8);
    const std::size_t nb_t = tree.levels_into(Side::Buy, t_bids, 8);
    const std::size_t na_t = tree.levels_into(Side::Sell, t_asks, 8);
    snap = make_top_k_snapshot<8>(flat, 42);
    REQUIRE(guard.delta().allocations == 0);
    REQUIRE(nb_f == 8);
    REQUIRE(na_f == 8);
    REQUIRE(nb_t == 8);
    REQUIRE(na_t == 8);
  }

  const auto vec_bids = flat.levels(Side::Buy, 8);
  const auto vec_asks = flat.levels(Side::Sell, 8);
  for (std::size_t i = 0; i < 8; ++i) {
    REQUIRE(f_bids[i].price == vec_bids[i].price);
    REQUIRE(f_bids[i].volume == vec_bids[i].volume);
    REQUIRE(t_bids[i].price == vec_bids[i].price);
    REQUIRE(snap.bids[i].price == vec_bids[i].price);
    REQUIRE(snap.asks[i].price == vec_asks[i].price);
  }

  // QueueReactiveProcess Q16 zero-libm fast path verification
  QueueReactiveProcess qrp;
  {
    sim::diag::ScopedAllocCounter guard;
    for (int step = 0; step < 256; ++step) {
      const auto ev = qrp.next_event_q16(1400, 120, 1410, 150);
      REQUIRE(ev.qty >= 1);
      REQUIRE(ev.inter_arrival_ns >= 1);
    }
    REQUIRE(guard.delta().allocations == 0);
  }
  REQUIRE(qrp.current_time() > 0);
  REQUIRE(qrp.hawkes_q16().event_count() == 256);
}

TEST_CASE("Pillar 4.9: Nested ReversibleSubgameScope commit() with external logs does not double-merge on destruction", "[pillar4][undo][commit]") {
  FlatLadderBook book(1000, 1200, 1024);

  Order seed_bid{}; seed_bid.agent = 1; seed_bid.side = Side::Buy; seed_bid.type = OrderType::Limit; seed_bid.price = 1090; seed_bid.qty = 50;
  Order seed_ask{}; seed_ask.agent = 2; seed_ask.side = Side::Sell; seed_ask.type = OrderType::Limit; seed_ask.price = 1110; seed_ask.qty = 50;
  book.add_outcome(seed_bid, 1);
  book.add_outcome(seed_ask, 2);

  const std::uint64_t root_digest = book.state_digest();

  LadderUndoLog outer_log;
  LadderUndoLog inner_log;

  {
    ReversibleSubgameScope outer(book, outer_log);
    Order b1{}; b1.agent = 3; b1.side = Side::Buy; b1.type = OrderType::Limit; b1.price = 1095; b1.qty = 20;
    outer.add_outcome(b1, 3);
    REQUIRE(outer.delta_count() == 1);

    {
      ReversibleSubgameScope inner(book, inner_log);
      // Partial fill + rest inside inner scope using distinct external log
      Order cross_buy{}; cross_buy.agent = 4; cross_buy.side = Side::Buy; cross_buy.type = OrderType::Limit; cross_buy.price = 1110; cross_buy.qty = 30;
      const auto mo = inner.add_outcome(cross_buy, 4);
      REQUIRE(mo.status == OrderStatus::FilledComplete);
      REQUIRE(mo.filled == 30);

      Order rest_ask{}; rest_ask.agent = 5; rest_ask.side = Side::Sell; rest_ask.type = OrderType::Limit; rest_ask.price = 1105; rest_ask.qty = 15;
      inner.add_outcome(rest_ask, 5);
      REQUIRE(inner.delta_count() == 2);

      // Explicitly commit inner scope: merges inner_log into outer_log once.
      inner.commit();
      REQUIRE(outer.delta_count() == 3);
    }
    // After inner scope destructor runs, outer.delta_count() must still be 3 (not 5!).
    REQUIRE(outer.delta_count() == 3);
    // Now outer scope rolls back on destruction, restoring exact root state!
  }

  REQUIRE(book.state_digest() == root_digest);
  REQUIRE(book.volume_at(Side::Buy, 1090) == 50);
  REQUIRE(book.volume_at(Side::Sell, 1110) == 50);
  REQUIRE(book.volume_at(Side::Buy, 1095) == 0);
  REQUIRE(book.volume_at(Side::Sell, 1105) == 0);
}
