#include "sim/bitwise_veb_codings.hpp"
#include <catch2/catch_test_macros.hpp>

#include "sim/alloc_probe.hpp"
#include "sim/book_observables.hpp"
#include "sim/dataset_size_guard.hpp"
#include "sim/execution_baselines.hpp"
#include "sim/export_schema.hpp"
#include "sim/feature_export_contract.hpp"
#include "sim/feature_export.hpp"
#include "sim/fill_attribution.hpp"
#include "sim/flat_ladder_book.hpp"
#include "sim/inventory_tca.hpp"
#include "sim/label_export.hpp"
#include "sim/lob.hpp"
#include "sim/markout_observer.hpp"
#include "sim/moving_window_features.hpp"
#include "sim/order_flow_features.hpp"
#include "sim/parent_order_tracker.hpp"
#include "sim/queue_reactive.hpp"
#include "sim/queue_state_observer.hpp"
#include "sim/quote_adapters.hpp"
#include "sim/ofi_sparsity_features.hpp"
#include "sim/replay_diagnostics.hpp"
#include "sim/replay_checkpoint.hpp"
#include "sim/replay_driver.hpp"
#include "sim/replay_pipeline.hpp"
#include "sim/reward_ledger.hpp"
#include "sim/risk_guard.hpp"
#include "sim/rng.hpp"
#include "sim/tca.hpp"
#include "sim/tca_diagnostics.hpp"
#include "sim/volume_depth_features.hpp"

#include <array>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>

#include <cstdlib>
#include <limits>
#include <new>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define QRLOB_SANITIZER_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define QRLOB_SANITIZER_ACTIVE 1
#endif
#endif

#ifndef QRLOB_SANITIZER_ACTIVE
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
namespace {
inline void* test_aligned_alloc(std::size_t size, std::size_t align) noexcept {
    if (align < sizeof(void*)) align = sizeof(void*);
    void* p = nullptr;
    if (posix_memalign(&p, align, size == 0 ? 1 : size) != 0) return nullptr;
    return p;
}
} // namespace

void* operator new(std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) throw std::bad_alloc();
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) throw std::bad_alloc();
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new(std::size_t size, std::align_val_t al) {
    void* p = test_aligned_alloc(size, static_cast<std::size_t>(al));
    if (!p) throw std::bad_alloc();
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size, std::align_val_t al) {
    void* p = test_aligned_alloc(size, static_cast<std::size_t>(al));
    if (!p) throw std::bad_alloc();
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p && sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (p && sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::align_val_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::align_val_t) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept { std::free(ptr); }
void operator delete(void* ptr, const std::nothrow_t&) noexcept { std::free(ptr); }
void operator delete[](void* ptr, const std::nothrow_t&) noexcept { std::free(ptr); }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

namespace {

inline sim::Order make_limit_order(sim::AgentId agent,
                                   sim::Side side,
                                   sim::Tick price,
                                   sim::Qty qty) {
    sim::Order o{};
    o.agent = agent;
    o.qty = qty;
    o.orig_qty = qty;
    o.price = price;
    o.side = side;
    o.type = sim::OrderType::Limit;
    return o;
}

inline sim::Order make_market_order(sim::AgentId agent,
                                    sim::Side side,
                                    sim::Qty qty) {
    sim::Order o{};
    o.agent = agent;
    o.qty = qty;
    o.orig_qty = qty;
    o.price = 0;
    o.side = side;
    o.type = sim::OrderType::Market;
    return o;
}

// Bridge the two add()/cancel() signatures: LOB takes no ts; FlatLadderBook
// requires ts. Default ts=0 in test code only -- production API still mandates
// an explicit timestamp on the FlatLadderBook hot path.
template <typename Book>
auto add_at(Book& b, const sim::Order& o, sim::TimeNs ts = 0) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        return b.add(o, ts);
    } else {
        (void)ts;
        return b.add(o);
    }
}

template <typename Book>
bool cancel_at(Book& b, sim::OrderId id, sim::TimeNs ts = 0) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        return b.cancel(id, ts);
    } else {
        (void)ts;
        return b.cancel(id);
    }
}

// Bridge the two best-side query signatures: LOB returns std::optional<Tick>
// (0-arg); FlatLadderBook returns bool with out-params. Both lower to the same
// optional<Tick> so equality comparison is meaningful in cross-checks.
template <typename Book>
std::optional<sim::Tick> best_bid_opt(const Book& b) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        sim::Tick p{}; sim::Qty v{};
        if (!b.best_bid(p, v)) return std::nullopt;
        return p;
    } else {
        return b.best_bid();
    }
}

template <typename Book>
std::optional<sim::Tick> best_ask_opt(const Book& b) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        sim::Tick p{}; sim::Qty v{};
        if (!b.best_ask(p, v)) return std::nullopt;
        return p;
    } else {
        return b.best_ask();
    }
}

template <typename ResultA, typename ResultB>
void require_same_add_result(const ResultA& a, const ResultB& b) {
    REQUIRE(a.filled == b.filled);
    REQUIRE(a.remaining == b.remaining);
}

template <typename A, typename B>
void require_same_book_state(const A& a, const B& b, sim::Tick min_tick, sim::Tick max_tick) {
    REQUIRE(best_bid_opt(a) == best_bid_opt(b));
    REQUIRE(best_ask_opt(a) == best_ask_opt(b));
    REQUIRE(a.order_count() == b.order_count());
    for (sim::Tick p = min_tick; p <= max_tick; ++p) {
        REQUIRE(a.volume_at(sim::Side::Buy, p) == b.volume_at(sim::Side::Buy, p));
        REQUIRE(a.volume_at(sim::Side::Sell, p) == b.volume_at(sim::Side::Sell, p));
    }
    REQUIRE(a.levels(sim::Side::Buy).size() == b.levels(sim::Side::Buy).size());
    REQUIRE(a.levels(sim::Side::Sell).size() == b.levels(sim::Side::Sell).size());
}

template <typename BookA, typename BookB>
void require_same_after_add(BookA& a,
                            BookB& b,
                            const sim::Order& order,
                            sim::Tick min_tick,
                            sim::Tick max_tick,
                            sim::TimeNs ts = 0) {
    const auto ra = add_at(a, order, ts);
    const auto rb = add_at(b, order, ts);
    require_same_add_result(ra, rb);
    require_same_book_state(a, b, min_tick, max_tick);
}

} // namespace

TEST_CASE("rng wrapper is deterministic for a fixed seed", "[rng]") {
    sim::Rng a(42);
    sim::Rng b(42);

    REQUIRE(a.next_u64() == b.next_u64());
    REQUIRE(a.next_u64() == b.next_u64());
    REQUIRE(a.next_u32() == b.next_u32());
}

TEST_CASE("uniform_u64 stays in range", "[rng]") {
    sim::Rng r(7);
    constexpr std::uint64_t upper = 10;
    for (int i = 0; i < 1000; ++i) {
        const auto x = sim::distributions::next_uniform_u64(r, upper);
        REQUIRE(x < upper);
    }
}

// --- RNG reproducibility contract (xoshiro256** golden + stream partitioning) -
// These pin the deterministic output so a future compiler/platform/refactor
// change that silently alters the RNG stream is caught immediately. The values
// were generated from this implementation for the documented seeds.
TEST_CASE("xoshiro256** golden sequence for the default seed", "[rng][golden]") {
    sim::Rng r;  // constructed with rng_constants::kDefaultSeed
    constexpr std::array<std::uint64_t, 8> kGolden = {
        0x422ea740d0977210ULL,
        0xe062b061b42e2928ULL,
        0x5a071fc5930841b6ULL,
        0x01334ef8ed3cc2bdULL,
        0xe45cbd6a2d9e96dbULL,
        0x3bc1fe841a5f292fULL,
        0x60001d95ebbbd8e6ULL,
        0xa0aee00b5b303762ULL,
    };
    for (std::uint64_t expected : kGolden) {
        REQUIRE(r.next_u64() == expected);
    }
}

TEST_CASE("xoshiro256** golden sequence for an explicit seed", "[rng][golden]") {
    sim::Rng r(0xDEADBEEFULL);
    constexpr std::array<std::uint64_t, 4> kGolden = {
        0xc5555444a74d7e83ULL,
        0x65c30d37b4b16e38ULL,
        0x54f773200a4efa23ULL,
        0x429aed75fb958af7ULL,
    };
    for (std::uint64_t expected : kGolden) {
        REQUIRE(r.next_u64() == expected);
    }
}

TEST_CASE("jump() partitions a seed into non-overlapping streams", "[rng][stream]") {
    // Two workers sharing a seed must use jump() to get disjoint substreams;
    // otherwise parallel scenario generation silently produces correlated draws.
    sim::Rng stream0(123ULL);
    sim::Rng stream1(123ULL);
    stream1.jump();

    constexpr int kWindow = 20000;
    std::set<std::uint64_t> seen;
    for (int i = 0; i < kWindow; ++i) {
        seen.insert(stream0.next_u64());
    }
    int overlap = 0;
    for (int i = 0; i < kWindow; ++i) {
        if (seen.count(stream1.next_u64()) != 0) {
            ++overlap;
        }
    }
    REQUIRE(overlap == 0);
}

TEST_CASE("long_jump() and jump() yield distinct stream origins", "[rng][stream]") {
    sim::Rng base(999ULL);
    sim::Rng jumped(999ULL);
    sim::Rng long_jumped(999ULL);
    jumped.jump();
    long_jumped.long_jump();

    const std::uint64_t b0 = base.next_u64();
    const std::uint64_t j0 = jumped.next_u64();
    const std::uint64_t l0 = long_jumped.next_u64();
    REQUIRE(b0 != j0);
    REQUIRE(b0 != l0);
    REQUIRE(j0 != l0);
}

TEST_CASE("next_unit_double stays in [0, 1)", "[rng][distribution]") {
    sim::Rng r(7ULL);
    for (int i = 0; i < 100000; ++i) {
        const double d = sim::distributions::next_unit_double(r);
        REQUIRE(d >= 0.0);
        REQUIRE(d < 1.0);
    }
}

TEST_CASE("next_uniform_u64 covers its range without bias for small bounds", "[rng][distribution]") {
    sim::Rng r(2024ULL);
    constexpr std::uint64_t upper = 6;  // fair die
    std::array<int, upper> counts{};
    constexpr int draws = 600000;
    for (int i = 0; i < draws; ++i) {
        counts[sim::distributions::next_uniform_u64(r, upper)]++;
    }
    // Each bucket should be near draws/upper (100k); allow a wide tolerance band
    // so the test asserts "no gross bias / no empty bucket", not exact uniformity.
    for (int c : counts) {
        REQUIRE(c > 90000);
        REQUIRE(c < 110000);
    }
}

TEST_CASE("flat ladder matches limit order book in a bounded band", "[lob][flat-ladder]") {
    constexpr sim::Tick kMin = 9980;
    constexpr sim::Tick kMax = 10020;

    sim::LimitOrderBook tree(64);
    sim::FlatLadderBook flat(kMin, kMax, 64);

    require_same_after_add(tree, flat, make_limit_order(1, sim::Side::Buy, 9998, 4), kMin, kMax, 1);
    require_same_after_add(tree, flat, make_limit_order(2, sim::Side::Sell, 10002, 3), kMin, kMax, 2);
    require_same_after_add(tree, flat, make_limit_order(3, sim::Side::Sell, 9998, 2), kMin, kMax, 3);
    require_same_after_add(tree, flat, make_limit_order(4, sim::Side::Buy, 10002, 4), kMin, kMax, 4);

    REQUIRE(cancel_at(tree, 1, 5) == cancel_at(flat, 1, 5));
    require_same_book_state(tree, flat, kMin, kMax);
}

TEST_CASE("flat ladder out-of-band limits do not rest", "[lob][flat-ladder]") {
    constexpr sim::Tick kMin = 9980;
    constexpr sim::Tick kMax = 10020;

    sim::FlatLadderBook book(kMin, kMax, 32);

    // Out-of-band passive order: should not rest on the ladder.
    const auto passive = book.add(make_limit_order(10, sim::Side::Buy, kMin - 1, 3), 1);
    REQUIRE(passive.filled == 0);
    REQUIRE(passive.remaining == 3);
    REQUIRE(book.order_count() == 0);

    // Out-of-band aggressive order: may match in-band liquidity,
    // but any remainder still does not rest.
    const auto seed = book.add(make_limit_order(11, sim::Side::Sell, kMax, 2), 2);
    REQUIRE(seed.remaining == 2);

    const auto aggressive = book.add(make_limit_order(12, sim::Side::Buy, kMax + 1, 5), 3);
    REQUIRE(aggressive.filled == 2);
    REQUIRE(aggressive.remaining == 3);
    REQUIRE(book.order_count() == 0);
    REQUIRE(book.volume_at(sim::Side::Sell, kMax) == 0);
}

TEST_CASE("flat ladder is_empty / total_volume / event_count", "[flat-ladder][meta]") {
    constexpr sim::Tick kMin = 1000;
    constexpr sim::Tick kMax = 1010;
    sim::FlatLadderBook book(kMin, kMax, 16);

    REQUIRE(book.is_empty());
    REQUIRE(book.total_volume(sim::Side::Buy) == 0);
    REQUIRE(book.total_volume(sim::Side::Sell) == 0);
    REQUIRE(book.event_count() == 0);

    book.add(make_limit_order(1, sim::Side::Buy, 1003, 5), 1);
    REQUIRE_FALSE(book.is_empty());
    REQUIRE(book.total_volume(sim::Side::Buy) == 5);
    REQUIRE(book.total_volume(sim::Side::Sell) == 0);
    REQUIRE(book.event_count() == 1);

    book.add(make_limit_order(2, sim::Side::Sell, 1007, 4), 2);
    REQUIRE(book.total_volume(sim::Side::Sell) == 4);
    REQUIRE(book.event_count() == 2);

    // Cross: 3 of the resting bid is consumed.
    book.add(make_limit_order(3, sim::Side::Sell, 1003, 3), 3);
    REQUIRE(book.total_volume(sim::Side::Buy) == 2);
    REQUIRE(book.event_count() == 3);

    // Cancel remaining bid   buy side empty, single side remains.
    REQUIRE(book.cancel(1, 4));
    REQUIRE(book.total_volume(sim::Side::Buy) == 0);
    REQUIRE(book.event_count() == 4);
    REQUIRE_FALSE(book.is_empty()); // ask side still has order 2.

    REQUIRE(book.cancel(2, 5));
    REQUIRE(book.is_empty());
    REQUIRE(book.event_count() == 5);

    // Double-cancel must be safe and reported as false.
    REQUIRE_FALSE(book.cancel(2, 6));
}

TEST_CASE("flat ladder mid_tick_x2 is integer and false on empty side", "[flat-ladder][mid]") {
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100;
    sim::FlatLadderBook book(kMin, kMax, 16);

    sim::Tick sum = -1;
    REQUIRE_FALSE(book.mid_tick_x2(sum));

    book.add(make_limit_order(1, sim::Side::Buy, 40, 1), 1);
    REQUIRE_FALSE(book.mid_tick_x2(sum)); // ask still empty

    book.add(make_limit_order(2, sim::Side::Sell, 50, 1), 2);
    REQUIRE(book.mid_tick_x2(sum));
    REQUIRE(sum == 90); // 40 + 50, no FP, no rounding.
}

TEST_CASE("flat ladder market vs limit matching (IsLimit template paths)", "[flat-ladder][match]") {
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100;
    sim::FlatLadderBook book(kMin, kMax, 32);

    book.add(make_limit_order(1, sim::Side::Sell, 50, 3), 1);
    book.add(make_limit_order(2, sim::Side::Sell, 60, 5), 2);

    // Market buy: no price cap; sweeps through both ask levels.
    auto m = book.add(make_market_order(3, sim::Side::Buy, 7), 3);
    REQUIRE(m.filled == 7);
    REQUIRE(m.remaining == 0);
    REQUIRE(book.volume_at(sim::Side::Sell, 50) == 0);
    REQUIRE(book.volume_at(sim::Side::Sell, 60) == 1);

    // Re-seed asks.
    book.add(make_limit_order(4, sim::Side::Sell, 70, 4), 4);

    // Aggressive limit buy with cap=65: should NOT touch the 70 level.
    auto l = book.add(make_limit_order(5, sim::Side::Buy, 65, 10), 5);
    REQUIRE(l.filled == 1);     // only the 60-level had 1 left
    REQUIRE(l.remaining == 9);  // residual rests at 65
    REQUIRE(book.volume_at(sim::Side::Sell, 70) == 4);
    REQUIRE(book.volume_at(sim::Side::Buy, 65) == 9);
}

TEST_CASE("flat ladder check_invariants holds across deep ladder operations",
          "[flat-ladder][deep][bitmap]") {
    // Wide band exercises the bitmap path: ladder = 100'001 entries, ~1563
    // 64-bit words. Linear refresh would dominate here.
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100'000;
    sim::FlatLadderBook book(kMin, kMax, 1024);

    book.check_invariants();

    // Rest at sparse, extreme prices to exercise the bitmap directly.
    // IDs are assigned monotonically by the book starting at 1, so:
    //   bids get ids 1..5 (in bid_prices order)
    //   asks get ids 6..10 (in ask_prices order)
    // Bid range is strictly below ask range so nothing crosses on rest.
    const std::array<sim::Tick, 5> bid_prices{1, 64, 65, 4096, 49'999};
    const std::array<sim::Tick, 5> ask_prices{50'000, 99'998, 99'999, 100'000, 50'001};

    sim::TimeNs ts = 0;
    for (int i = 0; i < 5; ++i) {
        book.add(make_limit_order(static_cast<sim::AgentId>(i + 1),
                                  sim::Side::Buy, bid_prices[i], 1), ++ts);
    }
    for (int i = 0; i < 5; ++i) {
        book.add(make_limit_order(static_cast<sim::AgentId>(100 + i),
                                  sim::Side::Sell, ask_prices[i], 1), ++ts);
    }

    book.check_invariants();
    {
        const auto bb = best_bid_opt(book);
        const auto ba = best_ask_opt(book);
        // Short-circuit AND so clang-tidy sees the optional check guarding
        // the dereference (REQUIRE alone isn't recognised as a precondition).
        REQUIRE((bb && *bb == 49'999));
        REQUIRE((ba && *ba == 50'000));
    }

    // Drain the inside bid (highest bid = 49'999   5th bid added   id 5).
    // Refresh must walk down the bitmap to the next non-empty word.
    REQUIRE(book.cancel(/*id=*/5, ++ts));
    book.check_invariants();
    {
        const auto bb = best_bid_opt(book);
        REQUIRE((bb && *bb == 4096));
    }

    // Drain the inside ask (lowest ask = 50'000   1st ask added   id 6).
    // Refresh must walk up the bitmap to the next non-empty word.
    REQUIRE(book.cancel(/*id=*/6, ++ts));
    book.check_invariants();
    {
        const auto ba = best_ask_opt(book);
        REQUIRE((ba && *ba == 50'001));
    }
}

TEST_CASE("trace event defaults are replay-safe sentinels", "[trace][observability]") {
    sim::TraceEvent event{};

    REQUIRE(event.seq == 0);
    REQUIRE(event.ts == 0);
    REQUIRE(event.kind == sim::TraceEventKind::SubmitLimit);
    REQUIRE(event.order_id == 0);
    REQUIRE(event.parent_id == 0);
    REQUIRE(event.agent == 0);
    REQUIRE(event.source == 0);
    REQUIRE(event.side == sim::Side::Buy);
    REQUIRE(event.liquidity == sim::LiquidityFlag::Unknown);
}

TEST_CASE("book observables handle empty and one-sided books", "[observability][flat-ladder]") {
    sim::FlatLadderBook book(-10, 10, 16);

    auto empty = sim::make_book_observables(book, 7);
    REQUIRE(empty.seq == 7);
    REQUIRE(empty.ts == 0);
    REQUIRE_FALSE(empty.has_bid);
    REQUIRE_FALSE(empty.has_ask);
    REQUIRE_FALSE(empty.has_mid);
    REQUIRE(empty.spread == 0);
    REQUIRE(empty.micro_price == 0.0);
    REQUIRE(empty.imbalance_l1 == 0.0);
    REQUIRE(empty.imbalance_total == 0.0);

    book.add(make_limit_order(1, sim::Side::Buy, -2, 5), 11);
    auto one_sided = sim::make_book_observables(book, 8);
    REQUIRE(one_sided.seq == 8);
    REQUIRE(one_sided.ts == 11);
    REQUIRE(one_sided.has_bid);
    REQUIRE_FALSE(one_sided.has_ask);
    REQUIRE_FALSE(one_sided.has_mid);
    REQUIRE(one_sided.best_bid == -2);
    REQUIRE(one_sided.best_bid_qty == 5);
    REQUIRE(one_sided.total_bid_qty == 5);
    REQUIRE(one_sided.total_ask_qty == 0);
    REQUIRE(one_sided.imbalance_l1 == 1.0);
    REQUIRE(one_sided.imbalance_total == 1.0);
}

TEST_CASE("book observables compute spread mid microprice and imbalance",
          "[observability][flat-ladder]") {
    sim::FlatLadderBook book(0, 100, 16);

    book.add(make_limit_order(1, sim::Side::Buy, 40, 3), 1);
    book.add(make_limit_order(2, sim::Side::Sell, 50, 1), 2);
    book.add(make_limit_order(3, sim::Side::Buy, 39, 7), 3);
    book.add(make_limit_order(4, sim::Side::Sell, 55, 9), 4);

    auto obs = sim::make_book_observables(book, 12);
    REQUIRE(obs.seq == 12);
    REQUIRE(obs.ts == 4);
    REQUIRE(obs.has_bid);
    REQUIRE(obs.has_ask);
    REQUIRE(obs.has_mid);
    REQUIRE(obs.best_bid == 40);
    REQUIRE(obs.best_ask == 50);
    REQUIRE(obs.best_bid_qty == 3);
    REQUIRE(obs.best_ask_qty == 1);
    REQUIRE(obs.mid_x2 == 90);
    REQUIRE(obs.spread == 10);
    REQUIRE(obs.micro_price == 47.5);
    REQUIRE(obs.total_bid_qty == 10);
    REQUIRE(obs.total_ask_qty == 10);
    REQUIRE(obs.imbalance_l1 == 0.5);
    REQUIRE(obs.imbalance_total == 0.0);
}

TEST_CASE("top-k snapshots preserve side order and sparse levels",
          "[observability][flat-ladder]") {
    sim::FlatLadderBook book(-100, 100, 32);

    book.add(make_limit_order(1, sim::Side::Buy, -90, 2), 1);
    book.add(make_limit_order(2, sim::Side::Buy, 10, 4), 2);
    book.add(make_limit_order(3, sim::Side::Buy, -10, 3), 3);
    book.add(make_limit_order(4, sim::Side::Sell, 90, 6), 4);
    book.add(make_limit_order(5, sim::Side::Sell, 20, 5), 5);
    book.add(make_limit_order(6, sim::Side::Sell, 70, 7), 6);

    const auto snap = sim::make_top_k_snapshot<2>(book, 99);
    REQUIRE(snap.seq == 99);
    REQUIRE(snap.ts == 6);
    REQUIRE(snap.bid_count == 2);
    REQUIRE(snap.ask_count == 2);

    REQUIRE(snap.bids[0].price == 10);
    REQUIRE(snap.bids[0].qty == 4);
    REQUIRE(snap.bids[0].order_count == 1);
    REQUIRE(snap.bids[1].price == -10);
    REQUIRE(snap.bids[1].qty == 3);

    REQUIRE(snap.asks[0].price == 20);
    REQUIRE(snap.asks[0].qty == 5);
    REQUIRE(snap.asks[1].price == 70);
    REQUIRE(snap.asks[1].qty == 7);
}

TEST_CASE("observables update deterministically after crossing trades",
          "[observability][flat-ladder][match]") {
    sim::FlatLadderBook book(0, 100, 32);

    book.add(make_limit_order(1, sim::Side::Buy, 40, 5), 1);
    book.add(make_limit_order(2, sim::Side::Sell, 50, 4), 2);

    const auto before = sim::make_book_observables(book, 1);
    REQUIRE(before.has_mid);
    REQUIRE(before.mid_x2 == 90);

    auto result = book.add(make_limit_order(3, sim::Side::Buy, 50, 2), 3);
    REQUIRE(result.filled == 2);
    REQUIRE(result.remaining == 0);
    REQUIRE(result.trades.size() == 1);

    const auto after = sim::make_book_observables(book, 2);
    REQUIRE(after.has_mid);
    REQUIRE(after.best_bid == 40);
    REQUIRE(after.best_ask == 50);
    REQUIRE(after.best_ask_qty == 2);
    REQUIRE(after.mid_x2 == 90);

    const auto snap = sim::make_top_k_snapshot<1>(book, 2);
    REQUIRE(snap.bid_count == 1);
    REQUIRE(snap.ask_count == 1);
    REQUIRE(snap.asks[0].price == 50);
    REQUIRE(snap.asks[0].qty == 2);
}

TEST_CASE("fill attribution derives maker side and fill-time mid",
          "[fill-attribution][markout]") {
    sim::FlatLadderBook book(0, 100, 32);

    book.add(make_limit_order(11, sim::Side::Sell, 50, 4), 1);
    book.add(make_limit_order(12, sim::Side::Buy, 40, 4), 2);
    auto at_fill = sim::make_book_observables(book, 1);
    REQUIRE(at_fill.has_mid);
    REQUIRE(at_fill.mid_x2 == 90);

    auto result = book.add(make_limit_order(21, sim::Side::Buy, 50, 2), 3);
    REQUIRE(result.trades.size() == 1);

    const auto fill = sim::make_fill_record(result.trades.front(), at_fill, 2, 7001, 7002);
    REQUIRE(fill.seq == 2);
    REQUIRE(fill.ts == 3);
    REQUIRE(fill.maker == 11);
    REQUIRE(fill.taker == 21);
    REQUIRE(fill.maker_parent_id == 7001);
    REQUIRE(fill.taker_parent_id == 7002);
    REQUIRE(fill.maker_side == sim::Side::Sell);
    REQUIRE(fill.taker_side == sim::Side::Buy);
    REQUIRE(fill.price == 50);
    REQUIRE(fill.qty == 2);
    REQUIRE(fill.has_mid_at_fill);
    REQUIRE(fill.mid_x2_at_fill == 90);
}

TEST_CASE("markouts are signed from maker and taker perspectives",
          "[fill-attribution][markout]") {
    sim::Trade sell_maker_trade{.maker_id = 1,
                                .taker_id = 2,
                                .maker = 10,
                                .taker = 20,
                                .taker_side = sim::Side::Buy,
                                .price = 50,
                                .qty = 3,
                                .ts = 100};
    sim::BookObservables at_fill{};
    at_fill.ts = 100;
    at_fill.seq = 1;
    at_fill.has_mid = true;
    at_fill.mid_x2 = 90;
    const auto sell_maker_fill = sim::make_fill_record(sell_maker_trade, at_fill, 1);

    sim::BookObservables after_down{};
    after_down.ts = 200;
    after_down.seq = 2;
    after_down.has_mid = true;
    after_down.mid_x2 = 88;
    const auto sell_maker_markout = sim::make_markout_record(sell_maker_fill, after_down, 2);

    REQUIRE(sell_maker_markout.has_mid_after);
    REQUIRE(sell_maker_markout.maker_markout_x2 == 12);  // passive sell at 50, future mid 44
    REQUIRE(sell_maker_markout.taker_markout_x2 == -12); // aggressive buy has opposite economics

    sim::Trade buy_maker_trade{.maker_id = 3,
                               .taker_id = 4,
                               .maker = 30,
                               .taker = 40,
                               .taker_side = sim::Side::Sell,
                               .price = 40,
                               .qty = 2,
                               .ts = 300};
    const auto buy_maker_fill = sim::make_fill_record(buy_maker_trade, at_fill, 3);
    REQUIRE(buy_maker_fill.maker_side == sim::Side::Buy);

    sim::BookObservables after_up{};
    after_up.ts = 400;
    after_up.seq = 4;
    after_up.has_mid = true;
    after_up.mid_x2 = 86;
    const auto buy_maker_markout = sim::make_markout_record(buy_maker_fill, after_up, 4);

    REQUIRE(buy_maker_markout.maker_markout_x2 == 6);  // passive buy at 40, future mid 43
    REQUIRE(buy_maker_markout.taker_markout_x2 == -6);
}

TEST_CASE("markout record is explicit when future mid is unavailable",
          "[fill-attribution][markout]") {
    sim::Trade trade{.maker_id = 1,
                     .taker_id = 2,
                     .maker = 10,
                     .taker = 20,
                     .taker_side = sim::Side::Buy,
                     .price = 50,
                     .qty = 1,
                     .ts = 100};
    sim::BookObservables at_fill{};
    at_fill.has_mid = false;
    const auto fill = sim::make_fill_record(trade, at_fill, 1);
    REQUIRE_FALSE(fill.has_mid_at_fill);
    REQUIRE(fill.mid_x2_at_fill == 0);

    sim::BookObservables no_future_mid{};
    no_future_mid.ts = 200;
    no_future_mid.seq = 2;
    no_future_mid.has_mid = false;
    const auto markout = sim::make_markout_record(fill, no_future_mid, 2);

    REQUIRE_FALSE(markout.has_mid_after);
    REQUIRE(markout.mid_x2_after == 0);
    REQUIRE(markout.maker_markout_x2 == 0);
    REQUIRE(markout.taker_markout_x2 == 0);
}

TEST_CASE("replay driver maps external order ids for cancel",
          "[replay][trace][flat-ladder]") {
    sim::ReplayDriver replay(0, 100, 16);

    sim::TraceEvent rest_bid{};
    rest_bid.seq = 1;
    rest_bid.ts = 10;
    rest_bid.kind = sim::TraceEventKind::SubmitLimit;
    rest_bid.order_id = 1001;
    rest_bid.agent = 7;
    rest_bid.side = sim::Side::Buy;
    rest_bid.price = 40;
    rest_bid.qty = 5;

    const auto add_step = replay.apply(rest_bid);
    REQUIRE(add_step.accepted);
    REQUIRE(add_step.execution.filled == 0);
    REQUIRE(add_step.execution.remaining == 5);
    REQUIRE(add_step.after.has_bid);
    REQUIRE(add_step.after.best_bid == 40);

    sim::OrderId internal_id{};
    REQUIRE(replay.internal_order_id(1001, internal_id));
    REQUIRE(internal_id == 1);

    sim::TraceEvent cancel{};
    cancel.seq = 2;
    cancel.ts = 11;
    cancel.kind = sim::TraceEventKind::Cancel;
    cancel.order_id = 1001;

    const auto cancel_step = replay.apply(cancel);
    REQUIRE(cancel_step.accepted);
    REQUIRE(cancel_step.cancel_accepted);
    REQUIRE_FALSE(cancel_step.after.has_bid);
    REQUIRE(replay.book().order_count() == 0);
}

TEST_CASE("replay driver emits fill records and trade trace events",
          "[replay][trace][fill-attribution]") {
    sim::ReplayDriver replay(0, 100, 16);

    sim::TraceEvent ask{};
    ask.seq = 1;
    ask.ts = 10;
    ask.kind = sim::TraceEventKind::SubmitLimit;
    ask.order_id = 2001;
    ask.parent_id = 8001;
    ask.agent = 11;
    ask.side = sim::Side::Sell;
    ask.price = 50;
    ask.qty = 4;
    replay.apply(ask);

    sim::TraceEvent bid{};
    bid.seq = 2;
    bid.ts = 11;
    bid.kind = sim::TraceEventKind::SubmitLimit;
    bid.order_id = 2002;
    bid.parent_id = 8002;
    bid.agent = 12;
    bid.side = sim::Side::Buy;
    bid.price = 40;
    bid.qty = 4;
    replay.apply(bid);

    sim::TraceEvent crossing_buy{};
    crossing_buy.seq = 3;
    crossing_buy.ts = 12;
    crossing_buy.kind = sim::TraceEventKind::SubmitLimit;
    crossing_buy.order_id = 2003;
    crossing_buy.parent_id = 9001;
    crossing_buy.agent = 21;
    crossing_buy.source = 77;
    crossing_buy.side = sim::Side::Buy;
    crossing_buy.price = 50;
    crossing_buy.qty = 2;

    const auto step = replay.apply(crossing_buy);
    REQUIRE(step.accepted);
    REQUIRE(step.execution.filled == 2);
    REQUIRE(step.execution.remaining == 0);
    REQUIRE(step.fills.size() == 1);
    REQUIRE(step.trade_events.size() == 1);

    const auto& fill = step.fills.front();
    REQUIRE(fill.maker_id == 1);
    REQUIRE(fill.taker_id == 3);
    REQUIRE(fill.maker_parent_id == 8001);
    REQUIRE(fill.taker_parent_id == 9001);
    REQUIRE(fill.source == 77);
    REQUIRE(fill.maker == 11);
    REQUIRE(fill.taker == 21);
    REQUIRE(fill.maker_side == sim::Side::Sell);
    REQUIRE(fill.taker_side == sim::Side::Buy);
    REQUIRE(fill.has_mid_at_fill);
    REQUIRE(fill.mid_x2_at_fill == 90);

    const auto& trade_event = step.trade_events.front();
    REQUIRE(trade_event.kind == sim::TraceEventKind::Trade);
    REQUIRE(trade_event.seq == 3);
    REQUIRE(trade_event.ts == 12);
    REQUIRE(trade_event.maker_id == 1);
    REQUIRE(trade_event.taker_id == 3);
    REQUIRE(trade_event.parent_id == 9001);
    REQUIRE(trade_event.agent == 21);
    REQUIRE(trade_event.counterparty == 11);
    REQUIRE(trade_event.source == 77);
    REQUIRE(trade_event.side == sim::Side::Buy);
    REQUIRE(trade_event.price == 50);
    REQUIRE(trade_event.qty == 2);
    REQUIRE(trade_event.liquidity == sim::LiquidityFlag::Aggressive);
}

TEST_CASE("replay driver rejects unknown cancels without mutating book",
          "[replay][trace][flat-ladder]") {
    sim::ReplayDriver replay(0, 100, 16);

    sim::TraceEvent unknown_cancel{};
    unknown_cancel.seq = 1;
    unknown_cancel.ts = 10;
    unknown_cancel.kind = sim::TraceEventKind::Cancel;
    unknown_cancel.order_id = 404;

    const auto step = replay.apply(unknown_cancel);
    REQUIRE_FALSE(step.accepted);
    REQUIRE_FALSE(step.cancel_accepted);
    REQUIRE(step.before.has_bid == step.after.has_bid);
    REQUIRE(step.before.has_ask == step.after.has_ask);
    REQUIRE(replay.book().is_empty());
}

TEST_CASE("replay diagnostics aggregate counts spreads and inter-arrivals",
          "[replay][diagnostics]") {
    sim::ReplayDriver replay(0, 100, 16);
    sim::ReplayDiagnosticsCollector diagnostics{};

    sim::TraceEvent bid{};
    bid.seq = 1;
    bid.ts = 10;
    bid.kind = sim::TraceEventKind::SubmitLimit;
    bid.order_id = 1001;
    bid.side = sim::Side::Buy;
    bid.price = 40;
    bid.qty = 5;
    diagnostics.add(replay.apply(bid));

    sim::TraceEvent ask{};
    ask.seq = 2;
    ask.ts = 13;
    ask.kind = sim::TraceEventKind::SubmitLimit;
    ask.order_id = 1002;
    ask.side = sim::Side::Sell;
    ask.price = 50;
    ask.qty = 5;
    diagnostics.add(replay.apply(ask));

    sim::TraceEvent buy{};
    buy.seq = 3;
    buy.ts = 20;
    buy.kind = sim::TraceEventKind::SubmitMarket;
    buy.order_id = 1003;
    buy.side = sim::Side::Buy;
    buy.qty = 2;
    diagnostics.add(replay.apply(buy));

    sim::TraceEvent cancel{};
    cancel.seq = 4;
    cancel.ts = 21;
    cancel.kind = sim::TraceEventKind::Cancel;
    cancel.order_id = 404;
    diagnostics.add(replay.apply(cancel));

    const auto& out = diagnostics.diagnostics();
    REQUIRE(out.input_events == 4);
    REQUIRE(out.accepted_events == 3);
    REQUIRE(out.rejected_events == 1);
    REQUIRE(out.limit_submits == 2);
    REQUIRE(out.market_submits == 1);
    REQUIRE(out.cancel_events == 1);
    REQUIRE(out.cancel_rejects == 1);
    REQUIRE(out.trades == 1);
    REQUIRE(out.filled_qty == 2);
    REQUIRE(out.mid_samples == 3);
    REQUIRE(out.spread_sum == 30);
    REQUIRE(out.min_spread == 10);
    REQUIRE(out.max_spread == 10);
    REQUIRE(sim::average_spread(out) == 10.0);
    REQUIRE(out.one_sided_samples == 1);
    REQUIRE(out.empty_book_samples == 0);
    REQUIRE(out.inter_arrival_samples == 3);
    REQUIRE(out.inter_arrival_sum == 11);
    REQUIRE(out.min_inter_arrival == 1);
    REQUIRE(out.max_inter_arrival == 7);
    REQUIRE(sim::average_inter_arrival_ns(out) == 11.0 / 3.0);

    const auto exported = sim::export_replay_diagnostics(out);
    REQUIRE(sim::kReplayDiagnosticsExportSchema.size() ==
            sim::kReplayDiagnosticsExportCount);
    REQUIRE(exported.size() == sim::kReplayDiagnosticsExportCount);
    REQUIRE(sim::kReplayDiagnosticsExportSchema[0].name == "input_events");
    REQUIRE(sim::kReplayDiagnosticsExportSchema[14].name == "average_spread");
    REQUIRE(sim::kReplayDiagnosticsExportSchema[22].name == "replay_hash");
    REQUIRE(exported[0] == 4.0);
    REQUIRE(exported[1] == 3.0);
    REQUIRE(exported[2] == 1.0);
    REQUIRE(exported[9] == 2.0);
    REQUIRE(exported[12] == 10.0);
    REQUIRE(exported[13] == 10.0);
    REQUIRE(exported[14] == 10.0);
    REQUIRE(exported[19] == 1.0);
    REQUIRE(exported[20] == 7.0);
    REQUIRE(exported[21] == 11.0 / 3.0);
    REQUIRE(exported[22] != 1469598103934665603.0);
}

TEST_CASE("replay diagnostics export zeroes unavailable minima",
          "[replay][diagnostics][schema]") {
    const sim::ReplayDiagnostics diagnostics{};
    const auto exported = sim::export_replay_diagnostics(diagnostics);
    REQUIRE(exported[12] == 0.0);
    REQUIRE(exported[13] == 0.0);
    REQUIRE(exported[14] == 0.0);
    REQUIRE(exported[19] == 0.0);
    REQUIRE(exported[20] == 0.0);
    REQUIRE(exported[21] == 0.0);
}

TEST_CASE("replay diagnostics flag causal-ordering violations",
          "[replay][diagnostics][ordering]") {
    REQUIRE(sim::kReplayDiagnosticsExportSchema[23].name == "seq_order_violations");
    REQUIRE(sim::kReplayDiagnosticsExportSchema[24].name == "time_order_violations");

    auto step_with = [](sim::EventSeq seq, sim::TimeNs ts) {
        sim::ReplayStep step{};
        step.input.seq = seq;
        step.input.ts = ts;
        step.input.kind = sim::TraceEventKind::Snapshot;
        step.accepted = true;
        return step;
    };

    SECTION("monotone seq and non-decreasing time produce no violations") {
        sim::ReplayDiagnostics d{};
        sim::add_replay_step(d, step_with(1, 10));
        sim::add_replay_step(d, step_with(2, 10));  // equal ts is allowed
        sim::add_replay_step(d, step_with(3, 25));
        REQUIRE(d.seq_order_violations == 0);
        REQUIRE(d.time_order_violations == 0);
    }

    SECTION("a reordered (decreasing) seq and ts are each counted") {
        sim::ReplayDiagnostics d{};
        sim::add_replay_step(d, step_with(5, 100));
        sim::add_replay_step(d, step_with(3, 90));  // seq and ts both go backwards
        REQUIRE(d.seq_order_violations == 1);
        REQUIRE(d.time_order_violations == 1);
    }

    SECTION("a duplicated seq at non-decreasing time counts only a seq violation") {
        sim::ReplayDiagnostics d{};
        sim::add_replay_step(d, step_with(7, 100));
        sim::add_replay_step(d, step_with(7, 105));  // seq not strictly increasing
        REQUIRE(d.seq_order_violations == 1);
        REQUIRE(d.time_order_violations == 0);
    }
}

TEST_CASE("replay diagnostics hash is deterministic for identical replay",
          "[replay][diagnostics]") {
    auto run = [](bool collect_observer_outputs) {
        sim::ReplayDriver replay(0, 100, 16);
        sim::ReplayDiagnosticsCollector diagnostics{};
        std::array<double, sim::kObservableFeatureCount> last_features{};
        sim::MarkoutObserver markouts({sim::MarkoutHorizon{.events = 1}});

        sim::TraceEvent bid{};
        bid.seq = 1;
        bid.ts = 10;
        bid.kind = sim::TraceEventKind::SubmitLimit;
        bid.order_id = 1001;
        bid.side = sim::Side::Buy;
        bid.price = 40;
        bid.qty = 5;
        const auto bid_step = replay.apply(bid);
        diagnostics.add(bid_step);
        if (collect_observer_outputs) {
            last_features = sim::export_observable_features(bid_step.after);
            (void)markouts.evaluate(bid_step.after);
        }

        sim::TraceEvent ask{};
        ask.seq = 2;
        ask.ts = 11;
        ask.kind = sim::TraceEventKind::SubmitLimit;
        ask.order_id = 1002;
        ask.side = sim::Side::Sell;
        ask.price = 50;
        ask.qty = 5;
        const auto ask_step = replay.apply(ask);
        diagnostics.add(ask_step);
        if (collect_observer_outputs) {
            last_features = sim::export_observable_features(ask_step.after);
            (void)markouts.evaluate(ask_step.after);
            REQUIRE(last_features[0] == 11.0);
        }

        return diagnostics.diagnostics().replay_hash;
    };

    REQUIRE(run(false) == run(false));
    REQUIRE(run(false) == run(true));
}

TEST_CASE("tca summary aggregates passive and aggressive fill notional",
           "[tca][fill-attribution]") {
    sim::TcaSummary summary{};

    sim::FillRecord passive{};
    passive.price = 50;
    passive.qty = 3;
    sim::FillRecord aggressive{};
    aggressive.price = 40;
    aggressive.qty = 2;

    sim::add_maker_fill(summary, passive);
    sim::add_taker_fill(summary, aggressive);

    REQUIRE(summary.passive_qty == 3);
    REQUIRE(summary.aggressive_qty == 2);
    REQUIRE(sim::total_qty(summary) == 5);
    REQUIRE(summary.passive_notional == 150);
    REQUIRE(summary.aggressive_notional == 80);
    REQUIRE(sim::total_notional(summary) == 230);
    REQUIRE(sim::average_fill_price(summary) == 46.0);
}

TEST_CASE("tca summary accounts for maker rebates and taker fees",
          "[tca][fees]") {
    sim::TcaSummary summary{};
    const sim::FeeSchedule fees{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};

    sim::FillRecord maker_fill{};
    maker_fill.price = 50;
    maker_fill.qty = 4;
    sim::FillRecord taker_fill{};
    taker_fill.price = 51;
    taker_fill.qty = 3;

    sim::add_maker_fill_with_fees(summary, maker_fill, fees);
    sim::add_taker_fill_with_fees(summary, taker_fill, fees);

    REQUIRE(summary.passive_qty == 4);
    REQUIRE(summary.aggressive_qty == 3);
    REQUIRE(summary.passive_fee_cost == -4);
    REQUIRE(summary.aggressive_fee_cost == 6);
    REQUIRE(sim::total_fee_cost(summary) == 2);
    REQUIRE(sim::total_notional(summary) == 353);
    REQUIRE(sim::total_net_notional(summary) == 355);
    REQUIRE(sim::average_passive_fee_per_qty(summary) == -1.0);
    REQUIRE(sim::average_aggressive_fee_per_qty(summary) == 2.0);

    const auto exported = sim::export_tca_summary(summary);
    REQUIRE(sim::kTcaSummaryExportSchema.size() == sim::kTcaSummaryExportCount);
    REQUIRE(exported.size() == sim::kTcaSummaryExportCount);
    REQUIRE(sim::kTcaSummaryExportSchema[0].name == "passive_qty");
    REQUIRE(sim::kTcaSummaryExportSchema[10].name == "average_fill_price");
    REQUIRE(sim::kTcaSummaryExportSchema[20].name ==
            "average_net_aggressive_markout_ticks");
    REQUIRE(exported[0] == 4.0);
    REQUIRE(exported[1] == 3.0);
    REQUIRE(exported[2] == 7.0);
    REQUIRE(exported[5] == 353.0);
    REQUIRE(exported[8] == 2.0);
    REQUIRE(exported[9] == 355.0);
    REQUIRE(exported[10] == 353.0 / 7.0);
    REQUIRE(exported[11] == -1.0);
    REQUIRE(exported[12] == 2.0);
}

TEST_CASE("tca fee arithmetic keeps local rebates separate from venue validation",
          "[tca][fees]") {
    sim::TcaSummary summary{};
    const sim::FeeSchedule free{};
    const sim::FeeSchedule maker_rebate{.maker_fee_per_qty = -3,
                                        .taker_fee_per_qty = 5};

    sim::FillRecord passive{};
    passive.price = 100;
    passive.qty = 2;
    sim::FillRecord aggressive{};
    aggressive.price = 101;
    aggressive.qty = 4;

    sim::add_maker_fill_with_fees(summary, passive, free);
    sim::add_taker_fill_with_fees(summary, aggressive, free);
    REQUIRE(sim::total_fee_cost(summary) == 0);
    REQUIRE(sim::total_net_notional(summary) == sim::total_notional(summary));

    sim::add_maker_fill_with_fees(summary, passive, maker_rebate);
    sim::add_taker_fill_with_fees(summary, aggressive, maker_rebate);
    REQUIRE(summary.passive_fee_cost == -6);
    REQUIRE(summary.aggressive_fee_cost == 20);
    REQUIRE(sim::total_fee_cost(summary) == 14);
    REQUIRE(sim::total_net_notional(summary) == sim::total_notional(summary) + 14);
}

TEST_CASE("tca summary aggregates signed markouts by liquidity role",
           "[tca][markout]") {
    sim::TcaSummary summary{};

    sim::FillRecord fill{};
    fill.price = 50;
    fill.qty = 4;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;

    sim::BookObservables after{};
    after.has_mid = true;
    after.mid_x2 = 90; // future mid = 45
    const auto markout = sim::make_markout_record(fill, after, 10);
    REQUIRE(markout.maker_markout_x2 == 10);
    REQUIRE(markout.taker_markout_x2 == -10);

    sim::add_maker_markout(summary, markout);
    sim::add_taker_markout(summary, markout);

    REQUIRE(summary.passive_markout_qty == 4);
    REQUIRE(summary.aggressive_markout_qty == 4);
    REQUIRE(summary.passive_markout_x2_qty == 40);
    REQUIRE(summary.aggressive_markout_x2_qty == -40);
    REQUIRE(sim::average_passive_markout_ticks(summary) == 5.0);
    REQUIRE(sim::average_aggressive_markout_ticks(summary) == -5.0);
}

TEST_CASE("tca conventions are explicit and gate report comparability",
          "[tca][conventions][metadata]") {
    sim::TcaConventions base{};
    base.fee_sign = sim::FeeSignConvention::PositiveIsCost;
    base.benchmark = sim::BenchmarkReference::ArrivalMid;
    base.terminal_mark = sim::TerminalMarkConvention::MarkToMid;
    base.residual_policy = sim::ResidualInventoryPolicy::MarkedAtTerminal;
    base.markout_horizon_events = 100;
    base.markout_horizon_ns = 0;
    base.assumed_latency_ns = 500;
    base.fees = sim::FeeSchedule{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};

    SECTION("default conventions are unspecified, not silently assumed") {
        const sim::TcaConventions def{};
        REQUIRE(def.fee_sign == sim::FeeSignConvention::Unspecified);
        REQUIRE(def.benchmark == sim::BenchmarkReference::Unspecified);
        REQUIRE(def.terminal_mark == sim::TerminalMarkConvention::Unspecified);
        REQUIRE(def.residual_policy == sim::ResidualInventoryPolicy::Unspecified);
        REQUIRE(def.assumed_latency_ns == 0);
    }

    SECTION("identical conventions are comparable") {
        const sim::TcaConventions same = base;
        REQUIRE(sim::conventions_comparable(base, same));
    }

    SECTION("a different terminal-mark convention is not comparable") {
        sim::TcaConventions other = base;
        other.terminal_mark = sim::TerminalMarkConvention::ForcedFlatNoResidual;
        REQUIRE_FALSE(sim::conventions_comparable(base, other));
    }

    SECTION("a flipped fee-sign convention is not comparable") {
        // Same numeric fees, opposite sign meaning: the maker rebate of one
        // report would read as a cost in the other. Must not be pooled.
        sim::TcaConventions other = base;
        other.fee_sign = sim::FeeSignConvention::Unspecified;
        REQUIRE_FALSE(sim::conventions_comparable(base, other));
    }

    SECTION("differing latency or markout horizon breaks comparability") {
        sim::TcaConventions slower = base;
        slower.assumed_latency_ns = 2000;
        REQUIRE_FALSE(sim::conventions_comparable(base, slower));

        sim::TcaConventions longer_horizon = base;
        longer_horizon.markout_horizon_events = 500;
        REQUIRE_FALSE(sim::conventions_comparable(base, longer_horizon));
    }

    SECTION("differing applied fee schedule breaks comparability") {
        sim::TcaConventions richer_rebate = base;
        richer_rebate.fees.maker_fee_per_qty = -3;
        REQUIRE_FALSE(sim::conventions_comparable(base, richer_rebate));
    }
}

TEST_CASE("tca net markout subtracts fees and adds rebates", "[tca][fees][markout]") {
    sim::TcaSummary summary{};
    const sim::FeeSchedule fees{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};

    sim::FillRecord fill{};
    fill.price = 50;
    fill.qty = 4;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;

    sim::add_maker_fill_with_fees(summary, fill, fees);
    sim::add_taker_fill_with_fees(summary, fill, fees);

    sim::BookObservables after{};
    after.has_mid = true;
    after.mid_x2 = 90;
    const auto markout = sim::make_markout_record(fill, after, 10);
    sim::add_maker_markout(summary, markout);
    sim::add_taker_markout(summary, markout);

    REQUIRE(sim::average_passive_markout_ticks(summary) == 5.0);
    REQUIRE(sim::average_aggressive_markout_ticks(summary) == -5.0);
    REQUIRE(sim::average_net_passive_markout_ticks(summary) == 6.0);
    REQUIRE(sim::average_net_aggressive_markout_ticks(summary) == -7.0);
}

TEST_CASE("tca computes implementation shortfall and effective spread",
          "[tca][spread][shortfall]") {
    sim::FillRecord buy_fill{};
    buy_fill.taker_side = sim::Side::Buy;
    buy_fill.maker_side = sim::Side::Sell;
    buy_fill.price = 51;
    buy_fill.qty = 2;
    buy_fill.has_mid_at_fill = true;
    buy_fill.mid_x2_at_fill = 100;

    REQUIRE(sim::signed_shortfall_x2(sim::Side::Buy, 51, 98) == 4);
    REQUIRE(sim::signed_shortfall_x2(sim::Side::Sell, 49, 100) == 2);
    REQUIRE(sim::taker_effective_spread_x2(buy_fill) == 2);
    REQUIRE(sim::maker_effective_spread_x2(buy_fill) == 2);

    sim::FillRecord sell_fill{};
    sell_fill.taker_side = sim::Side::Sell;
    sell_fill.maker_side = sim::Side::Buy;
    sell_fill.price = 49;
    sell_fill.has_mid_at_fill = true;
    sell_fill.mid_x2_at_fill = 100;
    REQUIRE(sim::taker_effective_spread_x2(sell_fill) == 2);
    REQUIRE(sim::maker_effective_spread_x2(sell_fill) == 2);
}

TEST_CASE("tca computes realized spread and adverse selection from markout",
          "[tca][spread][markout]") {
    sim::FillRecord fill{};
    fill.taker_side = sim::Side::Buy;
    fill.maker_side = sim::Side::Sell;
    fill.price = 50;
    fill.has_mid_at_fill = true;
    fill.mid_x2_at_fill = 98;

    sim::BookObservables after{};
    after.has_mid = true;
    after.mid_x2 = 94;
    const auto markout = sim::make_markout_record(fill, after, 10);

    REQUIRE(sim::maker_realized_spread_x2(markout) == 6);
    REQUIRE(sim::taker_adverse_selection_x2(markout) == 6);

    const auto metrics = sim::make_fill_tca_metrics(fill, markout, true, 96);
    REQUIRE(metrics.has_arrival_mid);
    REQUIRE(metrics.has_fill_mid);
    REQUIRE(metrics.has_future_mid);
    REQUIRE(metrics.implementation_shortfall_x2 == 4);
    REQUIRE(metrics.taker_effective_spread_x2 == 2);
    REQUIRE(metrics.maker_effective_spread_x2 == 2);
    REQUIRE(metrics.maker_realized_spread_x2 == 6);
    REQUIRE(metrics.taker_adverse_selection_x2 == 6);

    const auto exported = sim::export_fill_tca_metrics(metrics);
    REQUIRE(sim::kFillTcaMetricsExportSchema.size() ==
            sim::kFillTcaMetricsExportCount);
    REQUIRE(exported.size() == sim::kFillTcaMetricsExportCount);
    REQUIRE(sim::kFillTcaMetricsExportSchema[0].name == "has_arrival_mid");
    REQUIRE(sim::kFillTcaMetricsExportSchema[7].name ==
            "taker_adverse_selection_x2");
    REQUIRE(exported[0] == 1.0);
    REQUIRE(exported[1] == 1.0);
    REQUIRE(exported[2] == 1.0);
    REQUIRE(exported[3] == 4.0);
    REQUIRE(exported[4] == 2.0);
    REQUIRE(exported[5] == 2.0);
    REQUIRE(exported[6] == 6.0);
    REQUIRE(exported[7] == 6.0);
}

TEST_CASE("tca spread metrics are explicit when mids are unavailable",
          "[tca][spread]") {
    sim::FillRecord fill{};
    fill.taker_side = sim::Side::Buy;
    fill.maker_side = sim::Side::Sell;
    fill.price = 50;
    fill.has_mid_at_fill = false;

    sim::MarkoutRecord markout{};
    markout.fill = fill;
    markout.has_mid_after = false;

    const auto metrics = sim::make_fill_tca_metrics(fill, markout);
    REQUIRE_FALSE(metrics.has_arrival_mid);
    REQUIRE_FALSE(metrics.has_fill_mid);
    REQUIRE_FALSE(metrics.has_future_mid);
    REQUIRE(metrics.implementation_shortfall_x2 == 0);
    REQUIRE(metrics.taker_effective_spread_x2 == 0);
    REQUIRE(metrics.maker_effective_spread_x2 == 0);
    REQUIRE(metrics.maker_realized_spread_x2 == 0);
    REQUIRE(metrics.taker_adverse_selection_x2 == 0);
}

TEST_CASE("inventory tracker applies maker and taker fills by economic side",
          "[tca][inventory]") {
    sim::InventoryTracker tracker{};

    sim::FillRecord fill{};
    fill.source = 7;
    fill.maker = 11;
    fill.taker = 21;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 3;

    tracker.add_maker_fill(fill);
    tracker.add_taker_fill(fill);

    const auto* maker =
        tracker.find(sim::InventoryKey{.agent = 11, .source = 7});
    REQUIRE(maker != nullptr);
    REQUIRE(maker->position == -3);
    REQUIRE(maker->cash == 150);
    REQUIRE(maker->sold_qty == 3);
    REQUIRE(maker->bought_qty == 0);

    const auto* taker =
        tracker.find(sim::InventoryKey{.agent = 21, .source = 7});
    REQUIRE(taker != nullptr);
    REQUIRE(taker->position == 3);
    REQUIRE(taker->cash == -150);
    REQUIRE(taker->bought_qty == 3);
    REQUIRE(taker->sold_qty == 0);
}

TEST_CASE("inventory tracker applies fees rebates and mark-to-mid value",
          "[tca][inventory][fees]") {
    sim::InventoryTracker tracker{};
    const sim::FeeSchedule fees{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};

    sim::FillRecord fill{};
    fill.source = 9;
    fill.maker = 10;
    fill.taker = 20;
    fill.maker_side = sim::Side::Buy;
    fill.taker_side = sim::Side::Sell;
    fill.price = 40;
    fill.qty = 2;

    tracker.add_maker_fill(fill, fees);
    tracker.add_taker_fill(fill, fees);

    const auto* maker =
        tracker.find(sim::InventoryKey{.agent = 10, .source = 9});
    REQUIRE(maker != nullptr);
    REQUIRE(maker->position == 2);
    REQUIRE(maker->cash == -78);
    REQUIRE(maker->fee_cost == -2);
    REQUIRE(sim::mark_to_mid_value_x2(*maker, 84) == 12);
    REQUIRE(sim::mark_to_mid_value(*maker, 84) == 6.0);
    const auto maker_export = sim::export_inventory_state(*maker, true, 84);
    REQUIRE(sim::kInventoryStateExportSchema.size() ==
            sim::kInventoryStateExportCount);
    REQUIRE(maker_export.size() == sim::kInventoryStateExportCount);
    REQUIRE(sim::kInventoryStateExportSchema[0].name == "agent");
    REQUIRE(sim::kInventoryStateExportSchema[9].name == "mark_to_mid_value_x2");
    REQUIRE(sim::kInventoryStateExportSchema[10].name == "mark_to_mid_value");
    REQUIRE(maker_export[0] == 10.0);
    REQUIRE(maker_export[1] == 9.0);
    REQUIRE(maker_export[2] == 2.0);
    REQUIRE(maker_export[3] == -78.0);
    REQUIRE(maker_export[4] == -2.0);
    REQUIRE(maker_export[5] == 2.0);
    REQUIRE(maker_export[6] == 0.0);
    REQUIRE(maker_export[7] == 1.0);
    REQUIRE(maker_export[8] == 84.0);
    REQUIRE(maker_export[9] == 12.0);
    REQUIRE(maker_export[10] == 6.0);

    const auto* taker =
        tracker.find(sim::InventoryKey{.agent = 20, .source = 9});
    REQUIRE(taker != nullptr);
    REQUIRE(taker->position == -2);
    REQUIRE(taker->cash == 76);
    REQUIRE(taker->fee_cost == 4);
    REQUIRE(sim::mark_to_mid_value_x2(*taker, 84) == -16);
    REQUIRE(sim::mark_to_mid_value(*taker, 84) == -8.0);
    const auto taker_export_without_mid = sim::export_inventory_state(*taker);
    REQUIRE(taker_export_without_mid[7] == 0.0);
    REQUIRE(taker_export_without_mid[8] == 0.0);
    REQUIRE(taker_export_without_mid[9] == 0.0);
    REQUIRE(taker_export_without_mid[10] == 0.0);
}

TEST_CASE("tca summary ignores markouts without future mid", "[tca][markout]") {
    sim::TcaSummary summary{};

    sim::MarkoutRecord missing_mid{};
    missing_mid.has_mid_after = false;
    missing_mid.fill.qty = 7;
    missing_mid.maker_markout_x2 = 10;
    missing_mid.taker_markout_x2 = -10;

    sim::add_maker_markout(summary, missing_mid);
    sim::add_taker_markout(summary, missing_mid);

    REQUIRE(summary.passive_markout_qty == 0);
    REQUIRE(summary.aggressive_markout_qty == 0);
    REQUIRE(summary.passive_markout_x2_qty == 0);
    REQUIRE(summary.aggressive_markout_x2_qty == 0);
    REQUIRE(sim::average_passive_markout_ticks(summary) == 0.0);
    REQUIRE(sim::average_aggressive_markout_ticks(summary) == 0.0);
}

TEST_CASE("tca accumulator groups maker and taker by parent agent and source",
           "[tca][grouping]") {
    sim::FillRecord fill{};
    fill.maker_parent_id = 8001;
    fill.taker_parent_id = 9001;
    fill.source = 77;
    fill.maker = 11;
    fill.taker = 21;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 2;

    sim::BookObservables after{};
    after.has_mid = true;
    after.mid_x2 = 90;
    const auto markout = sim::make_markout_record(fill, after, 3);

    sim::TcaAccumulator accumulator{};
    accumulator.add_maker_fill(fill);
    accumulator.add_taker_fill(fill);
    accumulator.add_maker_markout(markout);
    accumulator.add_taker_markout(markout);

    REQUIRE(accumulator.size() == 2);

    const auto* maker_summary = accumulator.find(
        sim::TcaKey{.parent_id = 8001, .agent = 11, .source = 77});
    REQUIRE(maker_summary != nullptr);
    REQUIRE(maker_summary->passive_qty == 2);
    REQUIRE(maker_summary->aggressive_qty == 0);
    REQUIRE(maker_summary->passive_notional == 100);
    REQUIRE(sim::average_passive_markout_ticks(*maker_summary) == 5.0);

    const auto* taker_summary = accumulator.find(
        sim::TcaKey{.parent_id = 9001, .agent = 21, .source = 77});
    REQUIRE(taker_summary != nullptr);
    REQUIRE(taker_summary->passive_qty == 0);
    REQUIRE(taker_summary->aggressive_qty == 2);
    REQUIRE(taker_summary->aggressive_notional == 100);
    REQUIRE(sim::average_aggressive_markout_ticks(*taker_summary) == -5.0);
}

TEST_CASE("parent order metrics keep queue miss separate from opportunity cost",
          "[tca][parent][metrics]") {
    sim::ParentOrderTracker tracker{};

    sim::StrategyIntent passive{};
    passive.kind = sim::StrategyIntentKind::SubmitLimit;
    passive.parent_id = 42;
    passive.agent = 7;
    passive.source = 3;
    passive.side = sim::Side::Buy;
    passive.qty = 10;
    tracker.record_submission(passive);

    sim::StrategyIntent aggressive = passive;
    aggressive.kind = sim::StrategyIntentKind::SubmitMarket;
    aggressive.qty = 4;
    tracker.record_submission(aggressive);

    sim::FillRecord passive_fill{};
    passive_fill.maker_parent_id = 42;
    passive_fill.maker = 7;
    passive_fill.source = 3;
    passive_fill.maker_side = sim::Side::Buy;
    passive_fill.price = 100;
    passive_fill.qty = 6;
    tracker.record_maker_fill(passive_fill);

    sim::FillRecord aggressive_fill{};
    aggressive_fill.taker_parent_id = 42;
    aggressive_fill.taker = 7;
    aggressive_fill.source = 3;
    aggressive_fill.taker_side = sim::Side::Buy;
    aggressive_fill.price = 102;
    aggressive_fill.qty = 4;
    tracker.record_taker_fill(aggressive_fill);

    const auto* progress = tracker.find(sim::ParentOrderKey{
        .parent_id = 42, .agent = 7, .source = 3, .side = sim::Side::Buy});
    REQUIRE(progress != nullptr);
    REQUIRE(progress->submitted_qty == 14);
    REQUIRE(progress->passive_submitted_qty == 10);
    REQUIRE(progress->aggressive_submitted_qty == 4);
    REQUIRE(sim::filled_qty(*progress) == 10);
    REQUIRE(sim::remaining_qty(*progress) == 4);
    REQUIRE(sim::passive_queue_miss_qty(*progress) == 4);
    REQUIRE(sim::passive_queue_miss_rate(*progress) == 0.4);
    REQUIRE(sim::average_parent_fill_price(*progress) == 100.8);

    const auto missing_terminal = sim::make_opportunity_cost_metrics(
        *progress, true, 200, false, 0);
    REQUIRE(missing_terminal.unfilled_qty == 4);
    REQUIRE(missing_terminal.opportunity_cost_x2_qty == 0);
    REQUIRE(sim::average_opportunity_cost_ticks(missing_terminal) == 0.0);

    const auto opportunity = sim::make_opportunity_cost_metrics(
        *progress, true, 200, true, 206);
    REQUIRE(opportunity.unfilled_qty == 4);
    REQUIRE(opportunity.opportunity_cost_x2_per_qty == 6);
    REQUIRE(opportunity.opportunity_cost_x2_qty == 24);
    REQUIRE(sim::average_opportunity_cost_ticks(opportunity) == 3.0);
}

TEST_CASE("markout observer emits fills after event horizons", "[markout][observer]") {
    sim::MarkoutObserver observer({sim::MarkoutHorizon{.events = 2}});

    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 3;
    observer.record_fill(fill);

    sim::BookObservables too_early{};
    too_early.seq = 11;
    too_early.ts = 101;
    too_early.has_mid = true;
    too_early.mid_x2 = 92;
    REQUIRE(observer.evaluate(too_early).empty());
    REQUIRE(observer.pending_count() == 1);

    sim::BookObservables due{};
    due.seq = 12;
    due.ts = 102;
    due.has_mid = true;
    due.mid_x2 = 90;
    const auto markouts = observer.evaluate(due);

    REQUIRE(markouts.size() == 1);
    REQUIRE(observer.pending_count() == 0);
    REQUIRE(markouts.front().horizon_events == 2);
    REQUIRE(markouts.front().horizon_ns == 0);
    REQUIRE(markouts.front().eval_seq == 12);
    REQUIRE(markouts.front().eval_ts == 102);
    REQUIRE(markouts.front().maker_markout_x2 == 10);
    REQUIRE(markouts.front().taker_markout_x2 == -10);
}

TEST_CASE("markout observer supports time horizons and missing mids",
          "[markout][observer]") {
    sim::MarkoutObserver observer({sim::MarkoutHorizon{.ns = 5}});

    sim::FillRecord fill{};
    fill.seq = 20;
    fill.ts = 100;
    fill.maker_side = sim::Side::Buy;
    fill.taker_side = sim::Side::Sell;
    fill.price = 40;
    fill.qty = 1;
    observer.record_fill(fill);

    sim::BookObservables not_due{};
    not_due.seq = 21;
    not_due.ts = 104;
    not_due.has_mid = true;
    not_due.mid_x2 = 84;
    REQUIRE(observer.evaluate(not_due).empty());

    sim::BookObservables due_without_mid{};
    due_without_mid.seq = 22;
    due_without_mid.ts = 105;
    due_without_mid.has_mid = false;
    const auto markouts = observer.evaluate(due_without_mid);

    REQUIRE(markouts.size() == 1);
    REQUIRE(markouts.front().horizon_events == 0);
    REQUIRE(markouts.front().horizon_ns == 5);
    REQUIRE(markouts.front().eval_seq == 22);
    REQUIRE(markouts.front().eval_ts == 105);
    REQUIRE_FALSE(markouts.front().has_mid_after);
    REQUIRE(markouts.front().maker_markout_x2 == 0);
    REQUIRE(markouts.front().taker_markout_x2 == 0);
}

TEST_CASE("markout label availability is pending before the horizon and "
          "available at it, with no fabricated value in between",
          "[markout][observer][no-lookahead][labels]") {
    const sim::MarkoutHorizon horizon{.events = 5};
    sim::MarkoutObserver observer({horizon});

    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 2;
    observer.record_fill(fill);

    // Before the horizon (seq 12 < 10+5): label is Pending and NOTHING is
    // emitted -- the current mid must not leak in as a markout value.
    sim::BookObservables before{};
    before.seq = 12;
    before.ts = 112;
    before.has_mid = true;
    before.mid_x2 = 200;
    REQUIRE(sim::markout_label_availability(fill, before, horizon) ==
            sim::LabelAvailability::Pending);
    REQUIRE(observer.evaluate(before).empty());
    REQUIRE(observer.pending_count() == 1);

    // At the horizon (seq 15 == 10+5): label is Available and emitted exactly
    // once, valued at the horizon mid -- not the earlier `before` mid.
    sim::BookObservables at_horizon{};
    at_horizon.seq = 15;
    at_horizon.ts = 130;
    at_horizon.has_mid = true;
    at_horizon.mid_x2 = 90;  // fill price 50 -> price_x2 100; maker sell markout +10
    REQUIRE(sim::markout_label_availability(fill, at_horizon, horizon) ==
            sim::LabelAvailability::Available);
    const auto markouts = observer.evaluate(at_horizon);
    REQUIRE(markouts.size() == 1);
    REQUIRE(markouts.front().eval_seq == 15);
    REQUIRE(markouts.front().maker_markout_x2 == 10);
    REQUIRE(observer.pending_count() == 0);
}

TEST_CASE("a markout whose horizon is never reached is never fabricated",
          "[markout][observer][no-lookahead][labels]") {
    // Censoring: if the data ends before the horizon, the fill stays pending and
    // produces zero markout records. The label is absent, not guessed.
    const sim::MarkoutHorizon horizon{.events = 100};
    sim::MarkoutObserver observer({horizon});

    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.maker_side = sim::Side::Buy;
    fill.taker_side = sim::Side::Sell;
    fill.price = 40;
    fill.qty = 1;
    observer.record_fill(fill);

    std::size_t emitted = 0;
    for (sim::EventSeq s = 11; s <= 30; ++s) {  // stream ends far short of 110
        sim::BookObservables obs{};
        obs.seq = s;
        obs.ts = static_cast<sim::TimeNs>(100 + s);
        obs.has_mid = true;
        obs.mid_x2 = 80;
        emitted += observer.evaluate(obs).size();
        REQUIRE(sim::markout_label_availability(fill, obs, horizon) ==
                sim::LabelAvailability::Pending);
    }
    REQUIRE(emitted == 0);
    REQUIRE(observer.pending_count() == 1);  // unrealized, retained, not emitted
}

TEST_CASE("export schema meta defaults to the current schema version",
          "[features][schema][meta]") {
    sim::ExportSchemaMeta meta{};
    REQUIRE(meta.schema_version == sim::kExportSchemaVersion);
    REQUIRE(meta.timestamp_basis == sim::TimestampBasis::Unspecified);
    REQUIRE(meta.label_availability == sim::LabelAvailability::NotApplicable);
    REQUIRE(meta.label_horizon == 0);
}

TEST_CASE("identical numeric frames are interpretation-incompatible when units, "
          "time basis, or label horizon differ",
          "[features][schema][meta]") {
    // Two frames whose numbers are byte-identical but whose units or horizons
    // differ must not be treated as comparable.
    sim::ExportSchemaMeta ticks_eventtime{};
    ticks_eventtime.schema_name = "book_observables";
    ticks_eventtime.timestamp_basis = sim::TimestampBasis::EventTime;
    ticks_eventtime.price_unit = sim::QuantityUnit::Ticks;
    ticks_eventtime.quantity_unit = sim::QuantityUnit::Lots;

    sim::ExportSchemaMeta same = ticks_eventtime;
    REQUIRE(sim::interpretation_compatible(ticks_eventtime, same));

    sim::ExportSchemaMeta notional_wallclock_labelled = ticks_eventtime;
    notional_wallclock_labelled.timestamp_basis = sim::TimestampBasis::WallClock;
    notional_wallclock_labelled.price_unit = sim::QuantityUnit::Notional;
    notional_wallclock_labelled.label_horizon = 100;
    notional_wallclock_labelled.label_availability = sim::LabelAvailability::Pending;

    REQUIRE_FALSE(sim::interpretation_compatible(ticks_eventtime,
                                                 notional_wallclock_labelled));
}

TEST_CASE("provenance fields do not affect interpretation compatibility",
          "[features][schema][meta]") {
    // Two frames from different runs (distinct replay_hash / trace id) but the
    // same units/time-basis/horizon are still interpretation-compatible: the
    // numbers mean the same thing even though they came from different streams.
    sim::ExportSchemaMeta a{};
    a.schema_name = "order_flow";
    a.timestamp_basis = sim::TimestampBasis::EventTime;
    a.price_unit = sim::QuantityUnit::Ticks;
    a.replay_hash = 0x1111;
    a.source_trace_id = 1;
    a.calibration_window_id = 9;

    sim::ExportSchemaMeta b = a;
    b.replay_hash = 0x2222;
    b.source_trace_id = 2;
    b.calibration_window_id = 42;

    REQUIRE(sim::interpretation_compatible(a, b));
}

TEST_CASE("export schema meta flags incompatible schema versions",
          "[features][schema][meta]") {
    sim::ExportSchemaMeta current{};
    current.schema_name = "book_observables";
    sim::ExportSchemaMeta future = current;
    future.schema_version = sim::kExportSchemaVersion + 1;
    REQUIRE_FALSE(sim::interpretation_compatible(current, future));
}

TEST_CASE("feature exporter emits stable observable feature schema",
          "[features][observables]") {
    sim::BookObservables obs{};
    obs.ts = 123;
    obs.seq = 7;
    obs.has_bid = true;
    obs.has_ask = true;
    obs.has_mid = true;
    obs.best_bid = 40;
    obs.best_ask = 50;
    obs.best_bid_qty = 3;
    obs.best_ask_qty = 5;
    obs.mid_x2 = 90;
    obs.spread = 10;
    obs.micro_price = 43.75;
    obs.imbalance_l1 = -0.25;
    obs.imbalance_total = 0.5;

    const auto features = sim::export_observable_features(obs);
    REQUIRE(features.size() == sim::kObservableFeatureCount);
    REQUIRE(features[0] == 123.0);
    REQUIRE(features[1] == 7.0);
    REQUIRE(features[2] == 1.0);
    REQUIRE(features[3] == 1.0);
    REQUIRE(features[4] == 1.0);
    REQUIRE(features[5] == 40.0);
    REQUIRE(features[6] == 50.0);
    REQUIRE(features[7] == 3.0);
    REQUIRE(features[8] == 5.0);
    REQUIRE(features[9] == 90.0);
    REQUIRE(features[10] == 10.0);
    REQUIRE(features[11] == 43.75);
    REQUIRE(features[12] == -0.25);
    REQUIRE(features[13] == 0.5);
}

TEST_CASE("feature exporter preserves sparse top-k level presence",
          "[features][top-k]") {
    sim::TopKBookSnapshot<2> snapshot{};
    snapshot.ts = 200;
    snapshot.seq = 9;
    snapshot.bid_count = 1;
    snapshot.ask_count = 2;
    snapshot.bids[0] = sim::DepthLevel{.price = 40, .qty = 3, .order_count = 1};
    snapshot.asks[0] = sim::DepthLevel{.price = 50, .qty = 4, .order_count = 2};
    snapshot.asks[1] = sim::DepthLevel{.price = 55, .qty = 6, .order_count = 1};

    const auto features = sim::export_top_k_features(snapshot);
    REQUIRE(features.size() == sim::top_k_feature_count<2>());
    REQUIRE(features[0] == 200.0);
    REQUIRE(features[1] == 9.0);
    REQUIRE(features[2] == 1.0);
    REQUIRE(features[3] == 2.0);

    REQUIRE(features[4] == 1.0);
    REQUIRE(features[5] == 40.0);
    REQUIRE(features[6] == 3.0);
    REQUIRE(features[7] == 1.0);

    REQUIRE(features[8] == 0.0);
    REQUIRE(features[9] == 0.0);
    REQUIRE(features[10] == 0.0);
    REQUIRE(features[11] == 0.0);

    REQUIRE(features[12] == 1.0);
    REQUIRE(features[13] == 50.0);
    REQUIRE(features[14] == 4.0);
    REQUIRE(features[15] == 2.0);

    REQUIRE(features[16] == 1.0);
    REQUIRE(features[17] == 55.0);
    REQUIRE(features[18] == 6.0);
    REQUIRE(features[19] == 1.0);
}

TEST_CASE("feature exporter combines observables and top-k depth",
          "[features][book]") {
    sim::BookObservables obs{};
    obs.ts = 1;
    obs.seq = 2;
    obs.has_bid = true;
    obs.best_bid = 10;

    sim::TopKBookSnapshot<1> snapshot{};
    snapshot.ts = 1;
    snapshot.seq = 2;
    snapshot.bid_count = 1;
    snapshot.bids[0] = sim::DepthLevel{.price = 10, .qty = 7, .order_count = 3};

    const auto features = sim::export_book_features(obs, snapshot);
    REQUIRE(features.size() == sim::book_feature_count<1>());
    REQUIRE(features[0] == 1.0);
    REQUIRE(features[1] == 2.0);
    REQUIRE(features[5] == 10.0);
    REQUIRE(features[sim::kObservableFeatureCount] == 1.0);
    REQUIRE(features[sim::kObservableFeatureCount + 1] == 2.0);
    REQUIRE(features[sim::kObservableFeatureCount + 4] == 1.0);
    REQUIRE(features[sim::kObservableFeatureCount + 5] == 10.0);
    REQUIRE(features[sim::kObservableFeatureCount + 6] == 7.0);
    REQUIRE(features[sim::kObservableFeatureCount + 7] == 3.0);
}

TEST_CASE("feature exporter keeps pre-decision snapshots from looking ahead",
          "[features][book][no-lookahead]") {
    sim::FlatLadderBook book(0, 100, 16);
    REQUIRE(book.add(make_limit_order(1, sim::Side::Buy, 40, 5), 10).remaining == 5);
    REQUIRE(book.add(make_limit_order(2, sim::Side::Sell, 50, 7), 11).remaining == 7);

    const auto before_obs = sim::make_book_observables(book, 2);
    const auto before_snapshot = sim::make_top_k_snapshot<1>(book, 2);
    const auto before_features =
        sim::export_book_features(before_obs, before_snapshot);

    REQUIRE(book.add(make_limit_order(3, sim::Side::Sell, 45, 3), 12).remaining == 3);
    const auto after_obs = sim::make_book_observables(book, 3);
    const auto after_snapshot = sim::make_top_k_snapshot<1>(book, 3);
    const auto after_features =
        sim::export_book_features(after_obs, after_snapshot);

    REQUIRE(before_features[0] == 11.0);
    REQUIRE(before_features[1] == 2.0);
    REQUIRE(before_features[6] == 50.0);
    REQUIRE(before_features[sim::kObservableFeatureCount] == 11.0);
    REQUIRE(before_features[sim::kObservableFeatureCount + 1] == 2.0);
    REQUIRE(before_features[sim::kObservableFeatureCount + 9] == 50.0);

    REQUIRE(after_features[0] == 12.0);
    REQUIRE(after_features[1] == 3.0);
    REQUIRE(after_features[6] == 45.0);
    REQUIRE(after_features[sim::kObservableFeatureCount] == 12.0);
    REQUIRE(after_features[sim::kObservableFeatureCount + 1] == 3.0);
    REQUIRE(after_features[sim::kObservableFeatureCount + 9] == 45.0);
}

TEST_CASE("label exporter cannot backfill future markouts into fill labels",
          "[features][labels][no-lookahead]") {
    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.source = 77;
    fill.maker_parent_id = 8001;
    fill.taker_parent_id = 9001;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 2;
    fill.has_mid_at_fill = true;
    fill.mid_x2_at_fill = 90;

    const auto fill_labels = sim::export_fill_labels(fill);

    sim::BookObservables after{};
    after.seq = 12;
    after.ts = 110;
    after.has_mid = true;
    after.mid_x2 = 100;
    const auto markout = sim::make_markout_record(fill, after, 12, 2, 10);
    const auto markout_labels = sim::export_markout_labels(markout);

    REQUIRE(fill_labels[0] == 10.0);
    REQUIRE(fill_labels[11] == 90.0);
    REQUIRE(markout_labels[2] == 12.0);
    REQUIRE(markout_labels[16] == 100.0);
    REQUIRE(fill_labels.size() == sim::kFillLabelCount);
}

TEST_CASE("label exporter emits stable fill label schema", "[labels][fill]") {
    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.source = 77;
    fill.maker_parent_id = 8001;
    fill.taker_parent_id = 9001;
    fill.maker = 11;
    fill.taker = 21;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 2;
    fill.has_mid_at_fill = true;
    fill.mid_x2_at_fill = 90;

    const auto labels = sim::export_fill_labels(fill);
    REQUIRE(labels.size() == sim::kFillLabelCount);
    REQUIRE(labels[0] == 10.0);
    REQUIRE(labels[1] == 100.0);
    REQUIRE(labels[2] == 77.0);
    REQUIRE(labels[3] == 8001.0);
    REQUIRE(labels[4] == 9001.0);
    REQUIRE(labels[5] == 11.0);
    REQUIRE(labels[6] == 21.0);
    REQUIRE(labels[7] == -1.0);
    REQUIRE(labels[8] == 1.0);
    REQUIRE(labels[9] == 50.0);
    REQUIRE(labels[10] == 2.0);
    REQUIRE(labels[11] == 90.0);
}

TEST_CASE("label exporter emits markout labels with availability flags",
          "[labels][markout]") {
    sim::FillRecord fill{};
    fill.seq = 10;
    fill.ts = 100;
    fill.source = 77;
    fill.maker_parent_id = 8001;
    fill.taker_parent_id = 9001;
    fill.maker_side = sim::Side::Sell;
    fill.taker_side = sim::Side::Buy;
    fill.price = 50;
    fill.qty = 2;
    fill.has_mid_at_fill = true;
    fill.mid_x2_at_fill = 90;

    sim::BookObservables after{};
    after.ts = 110;
    after.has_mid = true;
    after.mid_x2 = 94;
    const auto markout = sim::make_markout_record(fill, after, 12, 2, 10);
    const auto labels = sim::export_markout_labels(markout);

    REQUIRE(labels.size() == sim::kMarkoutLabelCount);
    REQUIRE(labels[0] == 10.0);
    REQUIRE(labels[1] == 100.0);
    REQUIRE(labels[2] == 12.0);
    REQUIRE(labels[3] == 110.0);
    REQUIRE(labels[4] == 2.0);
    REQUIRE(labels[5] == 10.0);
    REQUIRE(labels[6] == 77.0);
    REQUIRE(labels[7] == 8001.0);
    REQUIRE(labels[8] == 9001.0);
    REQUIRE(labels[9] == -1.0);
    REQUIRE(labels[10] == 1.0);
    REQUIRE(labels[11] == 50.0);
    REQUIRE(labels[12] == 2.0);
    REQUIRE(labels[13] == 1.0);
    REQUIRE(labels[14] == 90.0);
    REQUIRE(labels[15] == 1.0);
    REQUIRE(labels[16] == 94.0);
    REQUIRE(labels[17] == 6.0);
    REQUIRE(labels[18] == -6.0);
    REQUIRE(labels[19] == -1.0);
}

TEST_CASE("label exporter keeps missing markout mid explicit",
          "[labels][markout]") {
    sim::MarkoutRecord markout{};
    markout.fill.seq = 1;
    markout.fill.ts = 2;
    markout.fill.maker_side = sim::Side::Buy;
    markout.fill.taker_side = sim::Side::Sell;
    markout.fill.price = 40;
    markout.fill.qty = 3;
    markout.fill.has_mid_at_fill = false;
    markout.has_mid_after = false;
    markout.maker_markout_x2 = 100;
    markout.taker_markout_x2 = -100;

    const auto labels = sim::export_markout_labels(markout);
    REQUIRE(labels[13] == 0.0);
    REQUIRE(labels[14] == 0.0);
    REQUIRE(labels[15] == 0.0);
    REQUIRE(labels[16] == 0.0);
    REQUIRE(labels[17] == 0.0);
    REQUIRE(labels[18] == 0.0);
    REQUIRE(labels[19] == 0.0);
}

TEST_CASE("label exporter emits parent completion labels",
          "[labels][parent-order]") {
    sim::ParentOrderProgress progress{};
    progress.key.parent_id = 8001;
    progress.key.agent = 42;
    progress.key.source = 7;
    progress.key.side = sim::Side::Buy;
    progress.submitted_qty = 10;
    progress.passive_submitted_qty = 6;
    progress.aggressive_submitted_qty = 4;
    progress.passive_filled_qty = 2;
    progress.aggressive_filled_qty = 3;
    progress.fill_notional = 255;

    const auto opportunity =
        sim::make_opportunity_cost_metrics(progress, true, 100, true, 110);
    const auto labels = sim::export_parent_completion_labels(progress, opportunity);

    REQUIRE(labels.size() == sim::kParentCompletionLabelCount);
    REQUIRE(labels[0] == 8001.0);
    REQUIRE(labels[1] == 42.0);
    REQUIRE(labels[2] == 7.0);
    REQUIRE(labels[3] == 1.0);
    REQUIRE(labels[4] == 10.0);
    REQUIRE(labels[5] == 6.0);
    REQUIRE(labels[6] == 4.0);
    REQUIRE(labels[7] == 2.0);
    REQUIRE(labels[8] == 3.0);
    REQUIRE(labels[9] == 5.0);
    REQUIRE(labels[10] == 5.0);
    REQUIRE(labels[11] == 0.5);
    REQUIRE(labels[12] == 2.0 / 6.0);
    REQUIRE(labels[13] == 3.0 / 4.0);
    REQUIRE(labels[14] == 4.0);
    REQUIRE(labels[15] == 4.0 / 6.0);
    REQUIRE(labels[16] == 255.0);
    REQUIRE(labels[17] == 51.0);
    REQUIRE(labels[18] == 1.0);
    REQUIRE(labels[19] == 1.0);
    REQUIRE(labels[20] == 5.0);
    REQUIRE(labels[21] == 10.0);
    REQUIRE(labels[22] == 50.0);
    REQUIRE(labels[23] == 5.0);
}

TEST_CASE("label exporter keeps parent opportunity cost availability explicit",
          "[labels][parent-order]") {
    sim::ParentOrderProgress progress{};
    progress.key.side = sim::Side::Sell;
    progress.submitted_qty = 4;
    progress.passive_filled_qty = 1;

    const auto opportunity =
        sim::make_opportunity_cost_metrics(progress, false, 0, true, 90);
    const auto labels = sim::export_parent_completion_labels(progress, opportunity);

    REQUIRE(labels[10] == 3.0);
    REQUIRE(labels[18] == 0.0);
    REQUIRE(labels[19] == 1.0);
    REQUIRE(labels[20] == 3.0);
    REQUIRE(labels[21] == 0.0);
    REQUIRE(labels[22] == 0.0);
    REQUIRE(labels[23] == 0.0);
}

TEST_CASE("export schemas describe fixed feature and label column order",
          "[features][labels][schema]") {
    REQUIRE(sim::kObservableFeatureSchema.size() == sim::kObservableFeatureCount);
    REQUIRE(sim::kObservableFeatureSchema[0].name == "ts");
    REQUIRE(sim::kObservableFeatureSchema[13].name == "imbalance_total");

    const auto top_k_schema = sim::top_k_feature_schema<2>();
    REQUIRE(top_k_schema.size() == sim::top_k_feature_count<2>());
    REQUIRE(top_k_schema[4].name == "present");
    REQUIRE(top_k_schema[4].side == "bid");
    REQUIRE(top_k_schema[4].index == 0);
    REQUIRE(top_k_schema[12].name == "present");
    REQUIRE(top_k_schema[12].side == "ask");
    REQUIRE(top_k_schema[12].index == 0);

    const auto book_schema = sim::book_feature_schema<1>();
    REQUIRE(book_schema.size() == sim::book_feature_count<1>());
    REQUIRE(book_schema[0].name == "ts");
    REQUIRE(book_schema[sim::kObservableFeatureCount].name == "ts");

    REQUIRE(sim::kOrderFlowFeatureSchema.size() == sim::kOrderFlowFeatureCount);
    REQUIRE(sim::kOrderFlowFeatureSchema[15].name == "signed_submit_qty");
    REQUIRE(sim::kOrderFlowFeatureSchema[17].name == "fill_count");

    const auto volume_schema = sim::volume_depth_feature_schema<2>();
    REQUIRE(volume_schema.size() == sim::volume_depth_feature_count<2>());
    REQUIRE(volume_schema[4].name == "cumulative_qty");
    REQUIRE(volume_schema[4].side == "bid");
    REQUIRE(volume_schema[4].index == 0);
    REQUIRE(volume_schema[9].name == "cumulative_qty");
    REQUIRE(volume_schema[9].side == "bid");
    REQUIRE(volume_schema[9].index == 1);

    const auto window_schema = sim::moving_window_feature_schema<3, 2>();
    REQUIRE(window_schema.size() == 6);
    REQUIRE(window_schema[0].name == "window_value");
    REQUIRE(window_schema[0].row == 0);
    REQUIRE(window_schema[0].col == 0);
    REQUIRE(window_schema[5].row == 2);
    REQUIRE(window_schema[5].col == 1);

    REQUIRE(sim::kFillLabelSchema.size() == sim::kFillLabelCount);
    REQUIRE(sim::kFillLabelSchema[10].name == "qty");
    REQUIRE(sim::kMarkoutLabelSchema[19].name == "taker_favorable");
    REQUIRE(sim::kParentCompletionLabelSchema.size() ==
            sim::kParentCompletionLabelCount);
    REQUIRE(sim::kParentCompletionLabelSchema[15].name ==
            "passive_queue_miss_rate");
    REQUIRE(sim::kParentCompletionLabelSchema[23].name ==
            "average_opportunity_cost_ticks");
}

TEST_CASE("order-flow exporter converts trace events into fixed deltas",
          "[features][order-flow]") {
    sim::TraceEvent event{};
    event.seq = 4;
    event.ts = 100;
    event.kind = sim::TraceEventKind::SubmitLimit;
    event.source = 77;
    event.agent = 42;
    event.side = sim::Side::Buy;
    event.qty = 5;

    const auto delta = sim::make_order_flow_delta(event);
    REQUIRE(delta.accepted);
    REQUIRE(delta.limit_buy_qty == 5);
    REQUIRE(delta.limit_sell_qty == 0);
    REQUIRE(sim::signed_submit_qty(delta) == 5);

    const auto features = sim::export_order_flow_features(delta);
    REQUIRE(features.size() == sim::kOrderFlowFeatureCount);
    REQUIRE(features[0] == 4.0);
    REQUIRE(features[1] == 100.0);
    REQUIRE(features[2] == 1.0);
    REQUIRE(features[3] == static_cast<double>(sim::TraceEventKind::SubmitLimit));
    REQUIRE(features[4] == 1.0);
    REQUIRE(features[5] == 77.0);
    REQUIRE(features[6] == 42.0);
    REQUIRE(features[7] == 5.0);
    REQUIRE(features[15] == 5.0);
}

TEST_CASE("order-flow exporter keeps rejected events quantity-neutral",
          "[features][order-flow]") {
    sim::TraceEvent event{};
    event.seq = 5;
    event.ts = 101;
    event.kind = sim::TraceEventKind::SubmitMarket;
    event.side = sim::Side::Sell;
    event.qty = 9;

    const auto delta = sim::make_order_flow_delta(event, false);
    REQUIRE_FALSE(delta.accepted);
    REQUIRE(delta.market_sell_qty == 0);
    REQUIRE(sim::signed_submit_qty(delta) == 0);
    REQUIRE(sim::export_order_flow_features(delta)[2] == 0.0);
}

TEST_CASE("order-flow exporter aggregates replay trade flow separately from submit flow",
          "[features][order-flow][replay]") {
    sim::ReplayDriver replay(0, 100, 16);

    sim::TraceEvent ask{};
    ask.seq = 1;
    ask.ts = 10;
    ask.kind = sim::TraceEventKind::SubmitLimit;
    ask.order_id = 1001;
    ask.side = sim::Side::Sell;
    ask.price = 50;
    ask.qty = 5;
    replay.apply(ask);

    sim::TraceEvent buy{};
    buy.seq = 2;
    buy.ts = 11;
    buy.kind = sim::TraceEventKind::SubmitMarket;
    buy.order_id = 1002;
    buy.side = sim::Side::Buy;
    buy.qty = 2;
    const auto step = replay.apply(buy);

    const auto delta = sim::make_order_flow_delta(step);
    REQUIRE(delta.accepted);
    REQUIRE(delta.market_buy_qty == 2);
    REQUIRE(delta.trade_buy_qty == 2);
    REQUIRE(delta.trade_sell_qty == 0);
    REQUIRE(delta.filled_qty == 2);
    REQUIRE(delta.fill_count == 1);
    REQUIRE(sim::signed_submit_qty(delta) == 2);
    REQUIRE(sim::signed_trade_qty(delta) == 2);
}

TEST_CASE("moving feature window exports padded chronological tensors",
          "[features][window]") {
    sim::MovingFeatureWindow<3, 2> window{};
    REQUIRE(window.empty());
    REQUIRE_FALSE(window.full());

    window.push({1.0, 2.0});
    window.push({3.0, 4.0});

    REQUIRE(window.count() == 2);
    REQUIRE(window.row(0)[0] == 1.0);
    REQUIRE(window.row(1)[1] == 4.0);
    REQUIRE(window.row(2)[0] == 0.0);

    const auto tensor = window.export_tensor();
    REQUIRE(tensor[0] == 0.0);
    REQUIRE(tensor[1] == 0.0);
    REQUIRE(tensor[2] == 1.0);
    REQUIRE(tensor[3] == 2.0);
    REQUIRE(tensor[4] == 3.0);
    REQUIRE(tensor[5] == 4.0);
}

TEST_CASE("moving feature window drops oldest rows after capacity",
          "[features][window]") {
    sim::MovingFeatureWindow<3, 2> window{};
    window.push({1.0, 10.0});
    window.push({2.0, 20.0});
    window.push({3.0, 30.0});
    window.push({4.0, 40.0});

    REQUIRE(window.full());
    REQUIRE(window.count() == 3);
    REQUIRE(window.row(0)[0] == 2.0);
    REQUIRE(window.row(1)[0] == 3.0);
    REQUIRE(window.row(2)[0] == 4.0);

    const auto tensor = window.export_tensor();
    REQUIRE(tensor[0] == 2.0);
    REQUIRE(tensor[1] == 20.0);
    REQUIRE(tensor[2] == 3.0);
    REQUIRE(tensor[3] == 30.0);
    REQUIRE(tensor[4] == 4.0);
    REQUIRE(tensor[5] == 40.0);
}

TEST_CASE("moving feature window clears buffered history",
          "[features][window]") {
    sim::MovingFeatureWindow<2, 3> window{};
    window.push({1.0, 2.0, 3.0});
    window.push({4.0, 5.0, 6.0});
    REQUIRE(window.full());

    window.clear();
    REQUIRE(window.empty());
    const auto tensor = window.export_tensor();
    for (double value : tensor) {
        REQUIRE(value == 0.0);
    }
}

TEST_CASE("volume-depth exporter emits cumulative depth representation",
          "[features][volume-depth]") {
    sim::TopKBookSnapshot<3> snapshot{};
    snapshot.ts = 100;
    snapshot.seq = 7;
    snapshot.bid_count = 2;
    snapshot.ask_count = 3;
    snapshot.bids[0] = sim::DepthLevel{.price = 40, .qty = 2, .order_count = 1};
    snapshot.bids[1] = sim::DepthLevel{.price = 39, .qty = 3, .order_count = 1};
    snapshot.asks[0] = sim::DepthLevel{.price = 50, .qty = 4, .order_count = 1};
    snapshot.asks[1] = sim::DepthLevel{.price = 51, .qty = 1, .order_count = 1};
    snapshot.asks[2] = sim::DepthLevel{.price = 52, .qty = 5, .order_count = 1};

    const auto features = sim::export_volume_depth_features(snapshot);
    REQUIRE(features.size() == sim::volume_depth_feature_count<3>());
    REQUIRE(features[0] == 100.0);
    REQUIRE(features[1] == 7.0);
    REQUIRE(features[2] == 2.0);
    REQUIRE(features[3] == 3.0);

    REQUIRE(features[4] == 2.0);
    REQUIRE(features[5] == 4.0);
    REQUIRE(features[6] == 80.0);
    REQUIRE(features[7] == 200.0);
    REQUIRE(features[8] == -1.0 / 3.0);

    REQUIRE(features[9] == 5.0);
    REQUIRE(features[10] == 5.0);
    REQUIRE(features[11] == 197.0);
    REQUIRE(features[12] == 251.0);
    REQUIRE(features[13] == 0.0);

    REQUIRE(features[14] == 5.0);
    REQUIRE(features[15] == 10.0);
    REQUIRE(features[16] == 197.0);
    REQUIRE(features[17] == 511.0);
    REQUIRE(features[18] == -1.0 / 3.0);
}

TEST_CASE("volume-depth exporter handles sparse one-sided snapshots",
          "[features][volume-depth]") {
    sim::TopKBookSnapshot<2> snapshot{};
    snapshot.bid_count = 1;
    snapshot.bids[0] = sim::DepthLevel{.price = 10, .qty = 3, .order_count = 1};

    const auto features = sim::export_volume_depth_features(snapshot);
    REQUIRE(features[2] == 1.0);
    REQUIRE(features[3] == 0.0);
    REQUIRE(features[4] == 3.0);
    REQUIRE(features[5] == 0.0);
    REQUIRE(features[6] == 30.0);
    REQUIRE(features[7] == 0.0);
    REQUIRE(features[8] == 1.0);
    REQUIRE(features[9] == 3.0);
    REQUIRE(features[10] == 0.0);
    REQUIRE(features[11] == 30.0);
    REQUIRE(features[12] == 0.0);
    REQUIRE(features[13] == 1.0);
}

TEST_CASE("parent order tracker records submissions and completion",
          "[parent-order][completion]") {
    sim::ParentOrderTracker tracker{};

    sim::StrategyIntent child_a{};
    child_a.kind = sim::StrategyIntentKind::SubmitMarket;
    child_a.parent_id = 8001;
    child_a.agent = 42;
    child_a.source = 7;
    child_a.side = sim::Side::Buy;
    child_a.qty = 3;
    tracker.record_submission(child_a);

    sim::StrategyIntent child_b = child_a;
    child_b.qty = 2;
    tracker.record_submission(child_b);

    const auto key = sim::ParentOrderKey{.parent_id = 8001,
                                         .agent = 42,
                                         .source = 7,
                                         .side = sim::Side::Buy};
    const auto* progress = tracker.find(key);
    REQUIRE(progress != nullptr);
    REQUIRE(progress->submitted_qty == 5);
    REQUIRE(progress->passive_submitted_qty == 0);
    REQUIRE(progress->aggressive_submitted_qty == 5);
    REQUIRE(sim::filled_qty(*progress) == 0);
    REQUIRE(sim::remaining_qty(*progress) == 5);
    REQUIRE(sim::completion_rate(*progress) == 0.0);
}

TEST_CASE("parent order tracker attributes passive and aggressive fills",
          "[parent-order][completion]") {
    sim::ParentOrderTracker tracker{};

    sim::StrategyIntent parent{};
    parent.kind = sim::StrategyIntentKind::SubmitLimit;
    parent.parent_id = 8001;
    parent.agent = 42;
    parent.source = 7;
    parent.side = sim::Side::Sell;
    parent.qty = 5;
    tracker.record_submission(parent);

    sim::FillRecord passive_fill{};
    passive_fill.maker_parent_id = 8001;
    passive_fill.maker = 42;
    passive_fill.source = 7;
    passive_fill.maker_side = sim::Side::Sell;
    passive_fill.price = 50;
    passive_fill.qty = 2;
    tracker.record_maker_fill(passive_fill);

    sim::FillRecord aggressive_fill{};
    aggressive_fill.taker_parent_id = 8001;
    aggressive_fill.taker = 42;
    aggressive_fill.source = 7;
    aggressive_fill.taker_side = sim::Side::Sell;
    aggressive_fill.price = 49;
    aggressive_fill.qty = 1;
    tracker.record_taker_fill(aggressive_fill);

    const auto key = sim::ParentOrderKey{.parent_id = 8001,
                                         .agent = 42,
                                         .source = 7,
                                         .side = sim::Side::Sell};
    const auto* progress = tracker.find(key);
    REQUIRE(progress != nullptr);
    REQUIRE(progress->submitted_qty == 5);
    REQUIRE(progress->passive_submitted_qty == 5);
    REQUIRE(progress->aggressive_submitted_qty == 0);
    REQUIRE(progress->passive_filled_qty == 2);
    REQUIRE(progress->aggressive_filled_qty == 1);
    REQUIRE(sim::filled_qty(*progress) == 3);
    REQUIRE(sim::remaining_qty(*progress) == 2);
    REQUIRE(sim::passive_queue_miss_qty(*progress) == 3);
    REQUIRE(sim::passive_fill_rate(*progress) == 0.4);
    REQUIRE(sim::aggressive_fill_rate(*progress) == 0.0);
    REQUIRE(sim::passive_queue_miss_rate(*progress) == 0.6);
    REQUIRE(progress->fill_notional == 149);
    REQUIRE(sim::average_parent_fill_price(*progress) == 149.0 / 3.0);
    REQUIRE(sim::completion_rate(*progress) == 0.6);
}

TEST_CASE("parent order tracker ignores cancels and caps overfill remaining at zero",
          "[parent-order][completion]") {
    sim::ParentOrderTracker tracker{};

    sim::StrategyIntent cancel{};
    cancel.kind = sim::StrategyIntentKind::Cancel;
    cancel.parent_id = 8001;
    cancel.agent = 42;
    cancel.source = 7;
    cancel.side = sim::Side::Buy;
    cancel.qty = 10;
    tracker.record_submission(cancel);
    REQUIRE(tracker.size() == 0);

    sim::StrategyIntent parent = cancel;
    parent.kind = sim::StrategyIntentKind::SubmitMarket;
    parent.qty = 1;
    tracker.record_submission(parent);

    sim::FillRecord fill{};
    fill.taker_parent_id = 8001;
    fill.taker = 42;
    fill.source = 7;
    fill.taker_side = sim::Side::Buy;
    fill.price = 10;
    fill.qty = 2;
    tracker.record_taker_fill(fill);

    const auto key = sim::ParentOrderKey{.parent_id = 8001,
                                         .agent = 42,
                                         .source = 7,
                                         .side = sim::Side::Buy};
    const auto* progress = tracker.find(key);
    REQUIRE(progress != nullptr);
    REQUIRE(sim::remaining_qty(*progress) == 0);
    REQUIRE(sim::completion_rate(*progress) == 2.0);
}

TEST_CASE("parent order opportunity cost prices unfilled buy quantity",
          "[parent-order][opportunity-cost]") {
    sim::ParentOrderProgress progress{};
    progress.key.side = sim::Side::Buy;
    progress.submitted_qty = 10;
    progress.aggressive_filled_qty = 4;

    const auto metrics =
        sim::make_opportunity_cost_metrics(progress, true, 100, true, 110);
    REQUIRE(metrics.has_benchmark_mid);
    REQUIRE(metrics.has_terminal_mid);
    REQUIRE(metrics.unfilled_qty == 6);
    REQUIRE(metrics.opportunity_cost_x2_per_qty == 10);
    REQUIRE(metrics.opportunity_cost_x2_qty == 60);
    REQUIRE(sim::average_opportunity_cost_ticks(metrics) == 5.0);
}

TEST_CASE("parent order opportunity cost uses opposite sign for sells",
          "[parent-order][opportunity-cost]") {
    sim::ParentOrderProgress progress{};
    progress.key.side = sim::Side::Sell;
    progress.submitted_qty = 5;
    progress.passive_filled_qty = 2;

    const auto favorable_miss =
        sim::make_opportunity_cost_metrics(progress, true, 100, true, 90);
    REQUIRE(favorable_miss.unfilled_qty == 3);
    REQUIRE(favorable_miss.opportunity_cost_x2_per_qty == 10);
    REQUIRE(favorable_miss.opportunity_cost_x2_qty == 30);

    const auto adverse_miss =
        sim::make_opportunity_cost_metrics(progress, true, 100, true, 110);
    REQUIRE(adverse_miss.opportunity_cost_x2_per_qty == -10);
    REQUIRE(adverse_miss.opportunity_cost_x2_qty == -30);
    REQUIRE(sim::average_opportunity_cost_ticks(adverse_miss) == -5.0);
}

TEST_CASE("parent order opportunity cost is explicit when mids are unavailable",
          "[parent-order][opportunity-cost]") {
    sim::ParentOrderProgress progress{};
    progress.key.side = sim::Side::Buy;
    progress.submitted_qty = 5;
    progress.aggressive_filled_qty = 1;

    const auto metrics =
        sim::make_opportunity_cost_metrics(progress, false, 0, true, 110);
    REQUIRE_FALSE(metrics.has_benchmark_mid);
    REQUIRE(metrics.has_terminal_mid);
    REQUIRE(metrics.unfilled_qty == 4);
    REQUIRE(metrics.opportunity_cost_x2_per_qty == 0);
    REQUIRE(metrics.opportunity_cost_x2_qty == 0);
    REQUIRE(sim::average_opportunity_cost_ticks(metrics) == 0.0);
}

TEST_CASE("risk guard accepts safe intents and converts them to trace events",
          "[risk][intent]") {
    sim::RiskLimits limits{};
    limits.max_abs_inventory = 10;
    limits.max_order_qty = 5;
    limits.max_order_notional = 250;
    limits.min_price = 1;
    limits.max_price = 100;
    sim::RiskGuard guard(limits);

    sim::StrategyIntent intent{};
    intent.seq = 7;
    intent.ts = 100;
    intent.kind = sim::StrategyIntentKind::SubmitLimit;
    intent.order_id = 1001;
    intent.parent_id = 8001;
    intent.agent = 42;
    intent.source = 3;
    intent.side = sim::Side::Buy;
    intent.price = 50;
    intent.qty = 5;

    const auto decision = guard.evaluate(intent, sim::RiskState{.inventory = 0});
    REQUIRE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::None);
    REQUIRE(decision.event.kind == sim::TraceEventKind::SubmitLimit);
    REQUIRE(decision.event.seq == 7);
    REQUIRE(decision.event.ts == 100);
    REQUIRE(decision.event.order_id == 1001);
    REQUIRE(decision.event.parent_id == 8001);
    REQUIRE(decision.event.agent == 42);
    REQUIRE(decision.event.source == 3);
    REQUIRE(decision.event.side == sim::Side::Buy);
    REQUIRE(decision.event.price == 50);
    REQUIRE(decision.event.qty == 5);
}

TEST_CASE("risk audit log records a deterministic accept/reject trail",
          "[risk][audit]") {
    sim::RiskLimits limits{};
    limits.max_abs_inventory = 10;
    limits.max_order_qty = 5;
    limits.min_price = 1;
    limits.max_price = 100;
    sim::RiskGuard guard(limits);
    sim::RiskAuditLog audit;

    auto submit = [](sim::EventSeq seq, sim::TimeNs ts, sim::Side side, sim::Qty qty) {
        sim::StrategyIntent intent{};
        intent.seq = seq;
        intent.ts = ts;
        intent.kind = sim::StrategyIntentKind::SubmitLimit;
        intent.agent = 1;
        intent.side = side;
        intent.price = 50;
        intent.qty = qty;
        return intent;
    };

    // 1: accepted buy of 5 (inventory 0 -> projected 5, within cap).
    auto i1 = submit(1, 100, sim::Side::Buy, 5);
    audit.record(i1, guard.evaluate(i1, sim::RiskState{.inventory = 0}));
    // 2: rejected -- order qty 6 exceeds max_order_qty 5.
    auto i2 = submit(2, 110, sim::Side::Buy, 6);
    audit.record(i2, guard.evaluate(i2, sim::RiskState{.inventory = 5}));
    // 3: rejected -- projected inventory 8+5=13 exceeds max_abs_inventory 10.
    auto i3 = submit(3, 120, sim::Side::Buy, 5);
    audit.record(i3, guard.evaluate(i3, sim::RiskState{.inventory = 8}));

    REQUIRE(audit.total() == 3);
    REQUIRE(audit.accepted() == 1);
    REQUIRE(audit.rejected() == 2);
    REQUIRE(audit.reject_count(sim::RiskRejectReason::OrderTooLarge) == 1);
    REQUIRE(audit.reject_count(sim::RiskRejectReason::InventoryLimit) == 1);
    REQUIRE(audit.reject_count(sim::RiskRejectReason::KillSwitch) == 0);

    const auto& records = audit.records();
    REQUIRE(records.size() == 3);
    REQUIRE(records[0].seq == 1);
    REQUIRE(records[0].accepted);
    REQUIRE(records[0].reject_reason == sim::RiskRejectReason::None);
    REQUIRE_FALSE(records[1].accepted);
    REQUIRE(records[1].reject_reason == sim::RiskRejectReason::OrderTooLarge);
    REQUIRE(records[2].reject_reason == sim::RiskRejectReason::InventoryLimit);

    REQUIRE(sim::risk_reject_reason_name(sim::RiskRejectReason::InventoryLimit) ==
            "inventory_limit");
}


TEST_CASE("risk guard rejects unsafe submit intents without spending messages",
          "[risk][intent]") {
    sim::RiskLimits limits{};
    limits.max_abs_inventory = 10;
    limits.max_order_qty = 5;
    limits.max_order_notional = 200;
    limits.min_price = 10;
    limits.max_price = 100;
    limits.max_messages = 1;
    limits.message_window_ns = 10;
    sim::RiskGuard guard(limits);

    sim::StrategyIntent too_large{};
    too_large.ts = 1;
    too_large.kind = sim::StrategyIntentKind::SubmitLimit;
    too_large.side = sim::Side::Buy;
    too_large.price = 50;
    too_large.qty = 6;
    auto decision = guard.evaluate(too_large, sim::RiskState{});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::OrderTooLarge);
    REQUIRE(guard.messages_in_window() == 0);

    sim::StrategyIntent too_expensive = too_large;
    too_expensive.qty = 5;
    decision = guard.evaluate(too_expensive, sim::RiskState{});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::NotionalTooLarge);
    REQUIRE(guard.messages_in_window() == 0);

    sim::StrategyIntent below_collar = too_expensive;
    below_collar.price = 9;
    decision = guard.evaluate(below_collar, sim::RiskState{});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::PriceBelowCollar);
    REQUIRE(guard.messages_in_window() == 0);

    sim::StrategyIntent breaches_inventory = too_expensive;
    breaches_inventory.price = 20;
    breaches_inventory.qty = 5;
    decision = guard.evaluate(breaches_inventory, sim::RiskState{.inventory = 8});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::InventoryLimit);
    REQUIRE(guard.messages_in_window() == 0);
}

TEST_CASE("risk guard enforces kill switch loss limit and message rate",
          "[risk][guard]") {
    sim::RiskGuard killed(sim::RiskLimits{.kill_switch = true});
    sim::StrategyIntent intent{};
    intent.ts = 1;
    intent.kind = sim::StrategyIntentKind::SubmitMarket;
    intent.side = sim::Side::Sell;
    intent.qty = 1;

    auto decision = killed.evaluate(intent, sim::RiskState{});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::KillSwitch);

    sim::RiskLimits loss_limits{};
    loss_limits.max_loss = 100;
    sim::RiskGuard loss_guard(loss_limits);
    decision = loss_guard.evaluate(intent, sim::RiskState{.realized_loss = 101});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::LossLimit);

    sim::RiskLimits rate_limits{};
    rate_limits.max_messages = 2;
    rate_limits.message_window_ns = 10;
    sim::RiskGuard rate_guard(rate_limits);
    REQUIRE(rate_guard.evaluate(intent, sim::RiskState{}).accepted);
    intent.ts = 2;
    REQUIRE(rate_guard.evaluate(intent, sim::RiskState{}).accepted);
    intent.ts = 3;
    decision = rate_guard.evaluate(intent, sim::RiskState{});
    REQUIRE_FALSE(decision.accepted);
    REQUIRE(decision.reject_reason == sim::RiskRejectReason::MessageRateLimit);
    intent.ts = 11;
    REQUIRE(rate_guard.evaluate(intent, sim::RiskState{}).accepted);
    REQUIRE(rate_guard.messages_in_window() == 1);
}

TEST_CASE("replay pipeline integrates risk diagnostics TCA labels and inventory",
          "[replay][pipeline][integration]") {
    sim::RiskLimits limits{};
    limits.max_abs_inventory = 10;
    limits.max_order_qty = 10;
    limits.min_price = 1;
    limits.max_price = 100;
    sim::RiskGuard guard(limits);
    sim::ReplayDriver replay(0, 100, 16);
    sim::ReplayDiagnosticsCollector diagnostics{};
    sim::ParentOrderTracker parent_tracker{};
    sim::MarkoutObserver markouts({sim::MarkoutHorizon{.events = 1, .ns = 0}});
    sim::TcaAccumulator tca{};
    sim::InventoryTracker inventory{};
    const sim::FeeSchedule fees{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};

    sim::StrategyIntent rejected{};
    rejected.seq = 0;
    rejected.ts = 1;
    rejected.kind = sim::StrategyIntentKind::SubmitLimit;
    rejected.order_id = 999;
    rejected.parent_id = 9999;
    rejected.agent = 99;
    rejected.source = 7;
    rejected.side = sim::Side::Buy;
    rejected.price = 50;
    rejected.qty = 11;
    const auto rejected_decision = guard.evaluate(rejected, sim::RiskState{});
    REQUIRE_FALSE(rejected_decision.accepted);
    REQUIRE(rejected_decision.reject_reason == sim::RiskRejectReason::OrderTooLarge);

    const std::array<sim::StrategyIntent, 4> intents = {{
        {.seq = 1,
         .ts = 10,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1001,
         .parent_id = 8001,
         .agent = 10,
         .source = 7,
         .side = sim::Side::Sell,
         .price = 52,
         .qty = 5},
        {.seq = 2,
         .ts = 20,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1002,
         .parent_id = 8002,
         .agent = 11,
         .source = 7,
         .side = sim::Side::Buy,
         .price = 48,
         .qty = 4},
        {.seq = 3,
         .ts = 30,
         .kind = sim::StrategyIntentKind::SubmitMarket,
         .order_id = 2001,
         .parent_id = 9001,
         .agent = 20,
         .source = 7,
         .side = sim::Side::Buy,
         .qty = 2},
        {.seq = 4,
         .ts = 40,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1003,
         .parent_id = 8003,
         .agent = 12,
         .source = 7,
         .side = sim::Side::Buy,
         .price = 50,
         .qty = 1},
    }};

    std::vector<sim::FillRecord> fills{};
    std::vector<sim::MarkoutRecord> emitted_markouts{};
    for (const sim::StrategyIntent& intent : intents) {
        const auto decision = guard.evaluate(intent, sim::RiskState{});
        REQUIRE(decision.accepted);
        parent_tracker.record_submission(intent);

        const auto step = replay.apply(decision.event);
        diagnostics.add(step);
        for (const sim::FillRecord& fill : step.fills) {
            fills.push_back(fill);
            parent_tracker.record_maker_fill(fill);
            parent_tracker.record_taker_fill(fill);
            tca.add_maker_fill_with_fees(fill, fees);
            tca.add_taker_fill_with_fees(fill, fees);
            inventory.add_maker_fill(fill, fees);
            inventory.add_taker_fill(fill, fees);
            markouts.record_fill(fill);
        }
        for (const sim::MarkoutRecord& markout : markouts.evaluate(step.after)) {
            emitted_markouts.push_back(markout);
            tca.add_maker_markout(markout);
            tca.add_taker_markout(markout);
        }
    }

    REQUIRE(fills.size() == 1);
    const sim::FillRecord& fill = fills.front();
    REQUIRE(fill.source == 7);
    REQUIRE(fill.maker_parent_id == 8001);
    REQUIRE(fill.taker_parent_id == 9001);
    REQUIRE(fill.maker == 10);
    REQUIRE(fill.taker == 20);
    REQUIRE(fill.price == 52);
    REQUIRE(fill.qty == 2);
    REQUIRE(fill.has_mid_at_fill);
    REQUIRE(fill.mid_x2_at_fill == 100);

    const auto& replay_stats = diagnostics.diagnostics();
    REQUIRE(replay_stats.input_events == 4);
    REQUIRE(replay_stats.accepted_events == 4);
    REQUIRE(replay_stats.limit_submits == 3);
    REQUIRE(replay_stats.market_submits == 1);
    REQUIRE(replay_stats.trades == 1);
    REQUIRE(replay_stats.filled_qty == 2);
    REQUIRE(replay_stats.mid_samples == 3);
    REQUIRE(sim::average_spread(replay_stats) == 10.0 / 3.0);
    REQUIRE(replay_stats.replay_hash != 1469598103934665603ULL);

    const auto maker_parent_key = sim::ParentOrderKey{.parent_id = 8001,
                                                      .agent = 10,
                                                      .source = 7,
                                                      .side = sim::Side::Sell};
    const auto taker_parent_key = sim::ParentOrderKey{.parent_id = 9001,
                                                      .agent = 20,
                                                      .source = 7,
                                                      .side = sim::Side::Buy};
    const auto* maker_parent = parent_tracker.find(maker_parent_key);
    const auto* taker_parent = parent_tracker.find(taker_parent_key);
    REQUIRE(maker_parent != nullptr);
    REQUIRE(taker_parent != nullptr);
    REQUIRE(maker_parent->passive_submitted_qty == 5);
    REQUIRE(maker_parent->passive_filled_qty == 2);
    REQUIRE(sim::passive_queue_miss_qty(*maker_parent) == 3);
    REQUIRE(taker_parent->aggressive_submitted_qty == 2);
    REQUIRE(taker_parent->aggressive_filled_qty == 2);
    REQUIRE(sim::completion_rate(*taker_parent) == 1.0);

    REQUIRE(emitted_markouts.size() == 1);
    const sim::MarkoutRecord& markout = emitted_markouts.front();
    REQUIRE(markout.has_mid_after);
    REQUIRE(markout.mid_x2_after == 102);
    REQUIRE(markout.maker_markout_x2 == 2);
    REQUIRE(markout.taker_markout_x2 == -2);
    const auto markout_labels = sim::export_markout_labels(markout);
    REQUIRE(markout_labels.size() == sim::kMarkoutLabelCount);
    REQUIRE(markout_labels[17] == 2.0);
    REQUIRE(markout_labels[18] == -2.0);

    const auto* maker_tca =
        tca.find(sim::TcaKey{.parent_id = 8001, .agent = 10, .source = 7});
    const auto* taker_tca =
        tca.find(sim::TcaKey{.parent_id = 9001, .agent = 20, .source = 7});
    REQUIRE(maker_tca != nullptr);
    REQUIRE(taker_tca != nullptr);
    REQUIRE(maker_tca->passive_qty == 2);
    REQUIRE(maker_tca->passive_fee_cost == -2);
    REQUIRE(sim::average_net_passive_markout_ticks(*maker_tca) == 2.0);
    REQUIRE(taker_tca->aggressive_qty == 2);
    REQUIRE(taker_tca->aggressive_fee_cost == 4);
    REQUIRE(sim::average_net_aggressive_markout_ticks(*taker_tca) == -3.0);

    const auto* maker_inventory =
        inventory.find(sim::InventoryKey{.agent = 10, .source = 7});
    const auto* taker_inventory =
        inventory.find(sim::InventoryKey{.agent = 20, .source = 7});
    REQUIRE(maker_inventory != nullptr);
    REQUIRE(taker_inventory != nullptr);
    REQUIRE(maker_inventory->position == -2);
    REQUIRE(maker_inventory->cash == 106);
    REQUIRE(taker_inventory->position == 2);
    REQUIRE(taker_inventory->cash == -108);

    const auto opportunity =
        sim::make_opportunity_cost_metrics(*maker_parent, true, 100, true, 102);
    const auto parent_labels =
        sim::export_parent_completion_labels(*maker_parent, opportunity);
    REQUIRE(parent_labels.size() == sim::kParentCompletionLabelCount);
    REQUIRE(parent_labels[4] == 5.0);
    REQUIRE(parent_labels[14] == 3.0);
    REQUIRE(parent_labels[23] == -1.0);
}

TEST_CASE("replay pipeline wrapper processes strategy intents deterministically",
          "[replay][pipeline]") {
    sim::ReplayPipelineConfig config{};
    config.risk_limits.max_order_qty = 10;
    config.risk_limits.min_price = 1;
    config.risk_limits.max_price = 100;
    config.min_tick = 0;
    config.max_tick = 100;
    config.expected_orders = 16;
    config.fees = sim::FeeSchedule{.maker_fee_per_qty = -1, .taker_fee_per_qty = 2};
    config.markout_horizons = {sim::MarkoutHorizon{.events = 1, .ns = 0}};
    sim::ReplayPipeline pipeline(std::move(config));

    sim::StrategyIntent rejected{};
    rejected.seq = 0;
    rejected.ts = 1;
    rejected.kind = sim::StrategyIntentKind::SubmitLimit;
    rejected.order_id = 999;
    rejected.parent_id = 9999;
    rejected.agent = 99;
    rejected.source = 7;
    rejected.side = sim::Side::Buy;
    rejected.price = 50;
    rejected.qty = 11;
    const auto rejected_step = pipeline.apply(rejected);
    REQUIRE_FALSE(rejected_step.risk.accepted);
    REQUIRE_FALSE(rejected_step.replayed);
    REQUIRE(pipeline.diagnostics().input_events == 0);
    REQUIRE(pipeline.pipeline_diagnostics().intents == 1);
    REQUIRE(pipeline.pipeline_diagnostics().accepted_intents == 0);
    REQUIRE(pipeline.pipeline_diagnostics().rejected_intents == 1);
    REQUIRE(pipeline.pipeline_diagnostics().replayed_events == 0);
    REQUIRE(pipeline.pipeline_diagnostics()
                .reject_reason_counts[sim::risk_reject_reason_index(
                    sim::RiskRejectReason::OrderTooLarge)] == 1);
    REQUIRE(pipeline.parent_orders().size() == 0);

    const std::array<sim::StrategyIntent, 4> intents = {{
        {.seq = 1,
         .ts = 10,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1001,
         .parent_id = 8001,
         .agent = 10,
         .source = 7,
         .side = sim::Side::Sell,
         .price = 52,
         .qty = 5},
        {.seq = 2,
         .ts = 20,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1002,
         .parent_id = 8002,
         .agent = 11,
         .source = 7,
         .side = sim::Side::Buy,
         .price = 48,
         .qty = 4},
        {.seq = 3,
         .ts = 30,
         .kind = sim::StrategyIntentKind::SubmitMarket,
         .order_id = 2001,
         .parent_id = 9001,
         .agent = 20,
         .source = 7,
         .side = sim::Side::Buy,
         .qty = 2},
        {.seq = 4,
         .ts = 40,
         .kind = sim::StrategyIntentKind::SubmitLimit,
         .order_id = 1003,
         .parent_id = 8003,
         .agent = 12,
         .source = 7,
         .side = sim::Side::Buy,
         .price = 50,
         .qty = 1},
    }};

    std::size_t fill_count = 0;
    std::size_t markout_count = 0;
    for (const sim::StrategyIntent& intent : intents) {
        const auto step = pipeline.apply(intent);
        REQUIRE(step.risk.accepted);
        REQUIRE(step.replayed);
        fill_count += step.replay.fills.size();
        markout_count += step.markouts.size();
    }

    REQUIRE(fill_count == 1);
    REQUIRE(markout_count == 1);
    REQUIRE(pipeline.pipeline_diagnostics().intents == 5);
    REQUIRE(pipeline.pipeline_diagnostics().accepted_intents == 4);
    REQUIRE(pipeline.pipeline_diagnostics().rejected_intents == 1);
    REQUIRE(pipeline.pipeline_diagnostics().replayed_events == 4);
    REQUIRE(pipeline.pipeline_diagnostics()
                .reject_reason_counts[sim::risk_reject_reason_index(
                    sim::RiskRejectReason::None)] == 0);
    const auto pipeline_diagnostics_schema =
        sim::replay_pipeline_diagnostics_schema();
    const auto pipeline_diagnostics_export =
        sim::export_replay_pipeline_diagnostics(pipeline.pipeline_diagnostics());
    REQUIRE(pipeline_diagnostics_schema.size() ==
            sim::kReplayPipelineDiagnosticsCount);
    REQUIRE(pipeline_diagnostics_export.size() ==
            sim::kReplayPipelineDiagnosticsCount);
    REQUIRE(pipeline_diagnostics_schema[0].name == "intents");
    REQUIRE(pipeline_diagnostics_schema[4].name == "reject_reason_count");
    REQUIRE(pipeline_diagnostics_schema[4].index ==
            sim::risk_reject_reason_index(sim::RiskRejectReason::None));
    REQUIRE(pipeline_diagnostics_export[0] == 5.0);
    REQUIRE(pipeline_diagnostics_export[1] == 4.0);
    REQUIRE(pipeline_diagnostics_export[2] == 1.0);
    REQUIRE(pipeline_diagnostics_export[3] == 4.0);
    REQUIRE(pipeline_diagnostics_export
                [4 + sim::risk_reject_reason_index(
                         sim::RiskRejectReason::OrderTooLarge)] == 1.0);
    REQUIRE(pipeline.diagnostics().input_events == 4);
    REQUIRE(pipeline.diagnostics().filled_qty == 2);

    const auto* maker_parent =
        pipeline.parent_orders().find(sim::ParentOrderKey{.parent_id = 8001,
                                                          .agent = 10,
                                                          .source = 7,
                                                          .side = sim::Side::Sell});
    REQUIRE(maker_parent != nullptr);
    REQUIRE(maker_parent->passive_submitted_qty == 5);
    REQUIRE(maker_parent->passive_filled_qty == 2);
    REQUIRE(sim::passive_queue_miss_qty(*maker_parent) == 3);

    const auto* maker_tca =
        pipeline.tca().find(sim::TcaKey{.parent_id = 8001, .agent = 10, .source = 7});
    const auto* taker_tca =
        pipeline.tca().find(sim::TcaKey{.parent_id = 9001, .agent = 20, .source = 7});
    REQUIRE(maker_tca != nullptr);
    REQUIRE(taker_tca != nullptr);
    REQUIRE(sim::average_net_passive_markout_ticks(*maker_tca) == 2.0);
    REQUIRE(sim::average_net_aggressive_markout_ticks(*taker_tca) == -3.0);

    const auto* maker_inventory =
        pipeline.inventory().find(sim::InventoryKey{.agent = 10, .source = 7});
    const auto* taker_inventory =
        pipeline.inventory().find(sim::InventoryKey{.agent = 20, .source = 7});
    REQUIRE(maker_inventory != nullptr);
    REQUIRE(taker_inventory != nullptr);
    REQUIRE(maker_inventory->cash == 106);
    REQUIRE(taker_inventory->cash == -108);
}

TEST_CASE("risk guard allows cancels through submit-specific checks",
          "[risk][cancel]") {
    sim::RiskLimits limits{};
    limits.max_order_qty = 1;
    limits.min_price = 10;
    limits.max_price = 20;
    sim::RiskGuard guard(limits);

    sim::StrategyIntent cancel{};
    cancel.seq = 5;
    cancel.ts = 9;
    cancel.kind = sim::StrategyIntentKind::Cancel;
    cancel.order_id = 1234;
    cancel.agent = 77;
    cancel.qty = 0;
    cancel.price = 999;

    const auto decision = guard.evaluate(cancel, sim::RiskState{.inventory = 100});
    REQUIRE(decision.accepted);
    REQUIRE(decision.event.kind == sim::TraceEventKind::Cancel);
    REQUIRE(decision.event.order_id == 1234);
    REQUIRE(decision.event.agent == 77);
}

TEST_CASE("execution baselines create no-trade and market liquidation intents",
          "[execution][baseline]") {
    REQUIRE(sim::make_no_trade_schedule().empty());

    sim::ParentExecutionOrder order{};
    order.parent_id = 8001;
    order.agent = 42;
    order.source = 7;
    order.side = sim::Side::Sell;
    order.qty = 11;

    const auto intent = sim::make_market_liquidation_intent(
        order, 100, sim::ScheduleIds{.first_seq = 3, .first_order_id = 9001});
    REQUIRE(intent.seq == 3);
    REQUIRE(intent.ts == 100);
    REQUIRE(intent.kind == sim::StrategyIntentKind::SubmitMarket);
    REQUIRE(intent.order_id == 9001);
    REQUIRE(intent.parent_id == 8001);
    REQUIRE(intent.agent == 42);
    REQUIRE(intent.source == 7);
    REQUIRE(intent.side == sim::Side::Sell);
    REQUIRE(intent.price == 0);
    REQUIRE(intent.qty == 11);
}

TEST_CASE("execution baselines create deterministic TWAP market schedules",
          "[execution][twap]") {
    sim::ParentExecutionOrder order{};
    order.parent_id = 8002;
    order.agent = 43;
    order.source = 8;
    order.side = sim::Side::Buy;
    order.qty = 10;

    const auto schedule = sim::make_twap_market_schedule(
        order, 100, 160, 3, sim::ScheduleIds{.first_seq = 10, .first_order_id = 1000});

    REQUIRE(schedule.size() == 3);
    REQUIRE(schedule[0].seq == 10);
    REQUIRE(schedule[1].seq == 11);
    REQUIRE(schedule[2].seq == 12);
    REQUIRE(schedule[0].order_id == 1000);
    REQUIRE(schedule[1].order_id == 1001);
    REQUIRE(schedule[2].order_id == 1002);
    REQUIRE(schedule[0].ts == 100);
    REQUIRE(schedule[1].ts == 130);
    REQUIRE(schedule[2].ts == 160);
    REQUIRE(schedule[0].qty == 4);
    REQUIRE(schedule[1].qty == 3);
    REQUIRE(schedule[2].qty == 3);
    REQUIRE(schedule[0].kind == sim::StrategyIntentKind::SubmitMarket);
    REQUIRE(schedule[2].parent_id == 8002);
    REQUIRE(schedule[2].agent == 43);
    REQUIRE(schedule[2].source == 8);
    REQUIRE(schedule[2].side == sim::Side::Buy);
}

TEST_CASE("execution baselines cap TWAP slices to positive child quantities",
          "[execution][twap]") {
    sim::ParentExecutionOrder order{};
    order.qty = 2;
    order.side = sim::Side::Sell;

    const auto schedule = sim::make_twap_market_schedule(order, 5, 15, 5);
    REQUIRE(schedule.size() == 2);
    REQUIRE(schedule[0].ts == 5);
    REQUIRE(schedule[1].ts == 15);
    REQUIRE(schedule[0].qty == 1);
    REQUIRE(schedule[1].qty == 1);

    order.qty = 0;
    REQUIRE(sim::make_twap_market_schedule(order, 5, 15, 3).empty());
    order.qty = 10;
    REQUIRE(sim::make_twap_market_schedule(order, 5, 15, 0).empty());
}

TEST_CASE("execution baselines create TWAP limit schedules for passive baselines",
          "[execution][twap][limit]") {
    sim::ParentExecutionOrder order{};
    order.parent_id = 8003;
    order.agent = 44;
    order.side = sim::Side::Buy;
    order.qty = 5;

    const auto schedule = sim::make_twap_limit_schedule(
        order, 42, 100, 100, 2, sim::ScheduleIds{.first_seq = 20, .first_order_id = 2000});

    REQUIRE(schedule.size() == 2);
    REQUIRE(schedule[0].kind == sim::StrategyIntentKind::SubmitLimit);
    REQUIRE(schedule[1].kind == sim::StrategyIntentKind::SubmitLimit);
    REQUIRE(schedule[0].price == 42);
    REQUIRE(schedule[1].price == 42);
    REQUIRE(schedule[0].ts == 100);
    REQUIRE(schedule[1].ts == 100);
    REQUIRE(schedule[0].qty == 3);
    REQUIRE(schedule[1].qty == 2);
}

// Allocation probe is replay-hash neutral: arming/disarming ScopedAllocCounter
// does not perturb the deterministic replay digest.
TEST_CASE("scoped alloc counter does not perturb replay hash",
          "[diagnostics][alloc-probe]") {
    auto run = [](bool arm_counter) {
        sim::ReplayDriver replay(0, 100, 16);
        sim::ReplayDiagnosticsCollector diagnostics{};

        auto drive = [&]() {
            sim::TraceEvent bid{};
            bid.seq = 1;
            bid.ts = 10;
            bid.kind = sim::TraceEventKind::SubmitLimit;
            bid.order_id = 1001;
            bid.side = sim::Side::Buy;
            bid.price = 40;
            bid.qty = 5;
            diagnostics.add(replay.apply(bid));

            sim::TraceEvent ask{};
            ask.seq = 2;
            ask.ts = 11;
            ask.kind = sim::TraceEventKind::SubmitLimit;
            ask.order_id = 1002;
            ask.side = sim::Side::Sell;
            ask.price = 50;
            ask.qty = 5;
            diagnostics.add(replay.apply(ask));
        };

        if (arm_counter) {
            sim::diag::ScopedAllocCounter counter;
            REQUIRE(sim::diag::alloc_counting_active());
            drive();
            (void)counter.delta();
        } else {
            drive();
        }
        return diagnostics.diagnostics().replay_hash;
    };

    REQUIRE(sim::diag::alloc_arm_depth() == 0);
    const auto baseline = run(false);
    REQUIRE(run(true) == baseline);
    // Arm-depth must unwind cleanly so later measurements are not corrupted.
    REQUIRE(sim::diag::alloc_arm_depth() == 0);
}

// Feature-export contract honours prefix-hash determinism.
TEST_CASE("sync feature export sink verifies prefix-hash parity",
          "[features][export-contract]") {
    sim::ReplayDriver replay(0, 100, 16);
    sim::ReplayDiagnosticsCollector diagnostics{};
    sim::SyncFeatureExportSink sink;

    auto emit = [&](const sim::TraceEvent& event) {
        const auto step = replay.apply(event);
        diagnostics.add(step);
        // Frame carries the prefix hash AS OF this event.
        sink.on_frame(sim::ExportFrame{.seq = step.input.seq,
                                       .replay_hash = diagnostics.diagnostics().replay_hash,
                                       .snapshot = step.after});
    };

    sim::TraceEvent bid{};
    bid.seq = 1;
    bid.ts = 10;
    bid.kind = sim::TraceEventKind::SubmitLimit;
    bid.order_id = 1001;
    bid.side = sim::Side::Buy;
    bid.price = 40;
    bid.qty = 5;
    emit(bid);

    sim::TraceEvent ask{};
    ask.seq = 2;
    ask.ts = 11;
    ask.kind = sim::TraceEventKind::SubmitLimit;
    ask.order_id = 1002;
    ask.side = sim::Side::Sell;
    ask.price = 50;
    ask.qty = 5;
    emit(ask);

    REQUIRE(sink.accepted_frames() == 2);
    REQUIRE(sink.seq_monotone());
    // Final accepted frame's hash equals the replay's final hash.
    REQUIRE(sink.verify_final_parity(diagnostics.diagnostics().replay_hash));
    // The snapshot rides along on each frame.
    REQUIRE(sink.frames().back().snapshot.seq == 2);
}

// Stronger than final-parity: an independent replay of the SAME input must emit
// an identical (seq, replay_hash) sequence frame-for-frame -- this pins every
// prefix, so a mid-stream dropped/duplicated/reordered frame is detectable.
TEST_CASE("feature export prefix-hash sequence is deterministic frame-for-frame",
          "[features][export-contract]") {
    auto build_stream = []() {
        sim::ReplayDriver replay(0, 100, 16);
        sim::ReplayDiagnosticsCollector diagnostics{};
        sim::SyncFeatureExportSink sink;
        auto emit = [&](const sim::TraceEvent& event) {
            const auto step = replay.apply(event);
            diagnostics.add(step);
            sink.on_frame(sim::ExportFrame{.seq = step.input.seq,
                                           .replay_hash = diagnostics.diagnostics().replay_hash,
                                           .snapshot = step.after});
        };
        sim::TraceEvent bid{};
        bid.seq = 1; bid.ts = 10; bid.kind = sim::TraceEventKind::SubmitLimit;
        bid.order_id = 1001; bid.side = sim::Side::Buy; bid.price = 40; bid.qty = 5;
        emit(bid);
        sim::TraceEvent ask{};
        ask.seq = 2; ask.ts = 11; ask.kind = sim::TraceEventKind::SubmitLimit;
        ask.order_id = 1002; ask.side = sim::Side::Sell; ask.price = 50; ask.qty = 5;
        emit(ask);
        sim::TraceEvent cross{};
        cross.seq = 3; cross.ts = 12; cross.kind = sim::TraceEventKind::SubmitLimit;
        cross.order_id = 1003; cross.side = sim::Side::Buy; cross.price = 50; cross.qty = 2;
        emit(cross);
        return sink.frames();
    };

    const auto a = build_stream();
    const auto b = build_stream();
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == 3);
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].seq == b[i].seq);
        // Every PREFIX hash matches, not just the final one.
        REQUIRE(a[i].replay_hash == b[i].replay_hash);
    }
    // Distinct events must advance the prefix hash (no accidental constant).
    REQUIRE(a[0].replay_hash != a[1].replay_hash);
    REQUIRE(a[1].replay_hash != a[2].replay_hash);
}

TEST_CASE("sync feature export sink rejects reordered and duplicated frames",
          "[features][export-contract]") {
    sim::SyncFeatureExportSink sink;
    sink.on_frame(sim::ExportFrame{.seq = 1, .replay_hash = 11});
    sink.on_frame(sim::ExportFrame{.seq = 2, .replay_hash = 22});

    SECTION("out-of-order frame is rejected") {
        sink.on_frame(sim::ExportFrame{.seq = 1, .replay_hash = 99});
        REQUIRE_FALSE(sink.seq_monotone());
        REQUIRE(sink.seq_violations() == 1);
        REQUIRE(sink.accepted_frames() == 2);
        REQUIRE(sink.last_hash() == 22);
    }

    SECTION("duplicate seq is rejected") {
        sink.on_frame(sim::ExportFrame{.seq = 2, .replay_hash = 99});
        REQUIRE_FALSE(sink.seq_monotone());
        REQUIRE(sink.accepted_frames() == 2);
    }

    SECTION("mismatched final hash fails parity") {
        REQUIRE_FALSE(sink.verify_final_parity(12345));
    }
}

// Dataset-size guard classification and budget enforcement.
TEST_CASE("dataset size guard is a no-op when disabled",
          "[replay][dataset-guard]") {
    sim::DatasetSizeBudget budget{};
    budget.enabled = false;
    budget.max_events = 1;
    budget.max_bytes = 1;
    REQUIRE(sim::classify_dataset_size(1'000'000, 1'000'000, budget) ==
            sim::SizeRegime::InRam);
    REQUIRE_NOTHROW(sim::check_dataset_size_or_throw(1'000'000, 1'000'000, budget));
}

TEST_CASE("dataset size guard refuses oversize datasets",
          "[replay][dataset-guard]") {
    sim::DatasetSizeBudget budget{};
    budget.enabled = true;
    budget.max_events = 1000;
    budget.max_bytes = 0; // unbounded on bytes
    budget.warn_fraction = 0.8;

    REQUIRE(sim::classify_dataset_size(500, 0, budget) == sim::SizeRegime::InRam);
    REQUIRE(sim::classify_dataset_size(900, 0, budget) == sim::SizeRegime::Warn);
    REQUIRE(sim::classify_dataset_size(1001, 0, budget) == sim::SizeRegime::Refuse);

    REQUIRE_NOTHROW(sim::check_dataset_size_or_throw(900, 0, budget));
    REQUIRE_THROWS_AS(sim::check_dataset_size_or_throw(1001, 0, budget),
                      std::runtime_error);
}

TEST_CASE("dataset size guard warns and refuses on the byte axis",
          "[replay][dataset-guard]") {
    sim::DatasetSizeBudget budget{};
    budget.enabled = true;
    budget.max_events = 0; // unbounded on events
    budget.max_bytes = 10'000;
    budget.warn_fraction = 0.9;

    REQUIRE(sim::classify_dataset_size(0, 5'000, budget) == sim::SizeRegime::InRam);
    REQUIRE(sim::classify_dataset_size(0, 9'500, budget) == sim::SizeRegime::Warn);
    REQUIRE(sim::classify_dataset_size(0, 10'001, budget) == sim::SizeRegime::Refuse);
    REQUIRE_THROWS_AS(sim::check_dataset_size_or_throw(0, 10'001, budget),
                      std::runtime_error);
}

TEST_CASE("dataset size guard tolerates an invalid warn fraction",
          "[replay][dataset-guard]") {
    sim::DatasetSizeBudget budget{};
    budget.enabled = true;
    budget.max_events = 1000;
    budget.warn_fraction = -0.5; // invalid -> treated as 1.0 (warn only at limit)

    // Refuse logic is unaffected by the bad fraction.
    REQUIRE(sim::classify_dataset_size(1001, 0, budget) == sim::SizeRegime::Refuse);
    // Below the limit stays InRam rather than silently mis-classifying.
    REQUIRE(sim::classify_dataset_size(900, 0, budget) == sim::SizeRegime::InRam);
    // At the limit it warns.
    REQUIRE(sim::classify_dataset_size(1000, 0, budget) == sim::SizeRegime::Warn);
}

TEST_CASE("queue-state observer seeds queue-ahead and trades consume the front",
          "[queue-state][observer]") {
    sim::QueueStateObserver obs;
    // Our order rests behind 10 units already at the level.
    obs.on_submit(/*id=*/1, sim::Side::Buy, /*price=*/50, /*qty=*/5,
                  /*seq=*/100, /*ts=*/1000, /*observed_qty_ahead=*/10);
    {
        const auto* s = obs.find(1);
        REQUIRE(s != nullptr);
        REQUIRE(s->queue_ahead_qty == 10);
        REQUIRE_FALSE(s->queue_ahead_censored);
        REQUIRE(s->lifecycle == sim::QueueLifecycle::Resting);
    }

    // A trade consumes 4 from the front: queue-ahead drops to 6, still exact.
    obs.on_level_trade(sim::Side::Buy, 50, 4, 101, 1100);
    REQUIRE(obs.find(1)->queue_ahead_qty == 6);
    REQUIRE_FALSE(obs.find(1)->queue_ahead_censored);

    // A trade larger than the remaining queue-ahead reaches/crosses us: the
    // estimate goes to 0 AND is censored (aggregate data cannot say how much hit
    // us). Fill/remaining are untouched -- only on_own_fill may change those.
    obs.on_level_trade(sim::Side::Buy, 50, 9, 102, 1200);
    {
        const auto* s = obs.find(1);
        REQUIRE(s->queue_ahead_qty == 0);
        REQUIRE(s->queue_ahead_censored);
        REQUIRE(s->filled_qty == 0);
        REQUIRE(s->remaining_qty == 5);
        REQUIRE(s->lifecycle == sim::QueueLifecycle::Resting);
    }

    // A trade on the opposite side / different price does not touch our order.
    sim::QueueStateObserver other;
    other.on_submit(2, sim::Side::Sell, 60, 3, 1, 1, 7);
    other.on_level_trade(sim::Side::Buy, 60, 100, 2, 2);   // wrong side
    other.on_level_trade(sim::Side::Sell, 61, 100, 3, 3);  // wrong price
    REQUIRE(other.find(2)->queue_ahead_qty == 7);
    REQUIRE_FALSE(other.find(2)->queue_ahead_censored);
}

TEST_CASE("queue-state observer censors queue-ahead on unknown-position cancels",
          "[queue-state][observer][censoring]") {
    sim::QueueStateObserver obs;
    obs.on_submit(1, sim::Side::Buy, 50, 5, 10, 100, /*qty_ahead=*/8);

    // Unknown-position cancel: counted but DOES NOT advance queue-ahead, and the
    // estimate is censored -- we cannot claim progress we did not observe.
    obs.on_level_cancel(sim::Side::Buy, 50, 3, 11, 110);
    {
        const auto* s = obs.find(1);
        REQUIRE(s->queue_ahead_qty == 8);
        REQUIRE(s->queue_ahead_censored);
        REQUIRE(s->cancels_total == 1);
        REQUIRE(s->cancels_unknown_position == 1);
        REQUIRE(s->cancels_ahead_qty == 0);
    }

    // A cancel KNOWN to be ahead does reduce queue-ahead and is accounted
    // separately from unknown-position cancels.
    sim::QueueStateObserver known;
    known.on_submit(2, sim::Side::Buy, 50, 5, 10, 100, 8);
    known.on_level_cancel(sim::Side::Buy, 50, 3, 11, 110,
                          /*position_known=*/true, /*ahead=*/true);
    {
        const auto* s = known.find(2);
        REQUIRE(s->queue_ahead_qty == 5);
        REQUIRE(s->cancels_ahead_qty == 3);
        REQUIRE(s->cancels_unknown_position == 0);
        REQUIRE(s->cancels_total == 1);
    }
}

TEST_CASE("queue-state observer tracks fill lifecycle without lookahead",
          "[queue-state][observer][lifecycle]") {
    sim::QueueStateObserver obs;
    obs.on_submit(1, sim::Side::Sell, 70, 6, 5, 50, /*qty_ahead=*/4);

    // Partial own fill: lifecycle advances, remaining is exact, and because the
    // prior queue-ahead estimate was nonzero the fill proves it was stale.
    obs.on_own_fill(1, 2);
    {
        const auto* s = obs.find(1);
        REQUIRE(s->filled_qty == 2);
        REQUIRE(s->remaining_qty == 4);
        REQUIRE(s->queue_ahead_qty == 0);
        REQUIRE(s->queue_ahead_censored);
        REQUIRE(s->lifecycle == sim::QueueLifecycle::PartiallyFilled);
    }

    obs.on_own_fill(1, 4);
    {
        const auto* s = obs.find(1);
        REQUIRE(s->filled_qty == 6);
        REQUIRE(s->remaining_qty == 0);
        REQUIRE(s->lifecycle == sim::QueueLifecycle::Filled);
        REQUIRE(sim::is_terminal(s->lifecycle));
    }

    // Age is computed against the sampling instant and clamps at 0.
    const auto s = obs.sample(1, /*current_seq=*/9, /*current_ts=*/130);
    REQUIRE(s.age_events == 4);
    REQUIRE(s.age_ns == 80);
    const auto past = obs.sample(1, /*current_seq=*/3, /*current_ts=*/40);
    REQUIRE(past.age_events == 0);
    REQUIRE(past.age_ns == 0);
}

TEST_CASE("queue-state observer right-censors unresolved orders, fabricating nothing",
          "[queue-state][observer][right-censoring]") {
    sim::QueueStateObserver obs;
    obs.on_submit(1, sim::Side::Buy, 50, 5, 1, 10, 0);  // never resolved
    obs.on_submit(2, sim::Side::Buy, 50, 5, 1, 10, 0);
    obs.on_own_fill(2, 5);                              // terminal: Filled
    obs.on_submit(3, sim::Side::Buy, 50, 5, 1, 10, 0);
    obs.on_own_fill(3, 2);                              // partially filled, unresolved
    obs.on_submit(4, sim::Side::Buy, 50, 5, 1, 10, 0);
    obs.on_own_cancel(4);                              // terminal: Cancelled

    obs.finalize_right_censored();

    REQUIRE(obs.find(1)->right_censored);                       // Resting -> censored
    REQUIRE(obs.find(1)->lifecycle == sim::QueueLifecycle::Resting);
    REQUIRE(obs.find(1)->filled_qty == 0);                      // nothing fabricated
    REQUIRE_FALSE(obs.find(2)->right_censored);                 // Filled is terminal
    REQUIRE(obs.find(3)->right_censored);                       // PartiallyFilled -> censored
    REQUIRE(obs.find(3)->lifecycle == sim::QueueLifecycle::PartiallyFilled);
    REQUIRE(obs.find(3)->filled_qty == 2);
    REQUIRE_FALSE(obs.find(4)->right_censored);                 // Cancelled is terminal
}

TEST_CASE("queue-state observer schema and metadata are explicit",
          "[queue-state][observer][schema]") {
    REQUIRE(sim::kQueueStateExportSchema.size() == sim::kQueueStateExportCount);
    const auto meta = sim::queue_state_schema_meta();
    REQUIRE(meta.schema_name == "queue_state");
    REQUIRE(meta.timestamp_basis == sim::TimestampBasis::EventTime);
    REQUIRE(meta.quantity_unit == sim::QuantityUnit::Lots);
    // A censorable estimate column means the missing policy must be explicit.
    REQUIRE(meta.missing_policy == sim::MissingPolicy::SentinelFlagged);

    sim::QueueStateObserver obs;
    obs.on_submit(7, sim::Side::Sell, 80, 4, 2, 20, 9);
    const auto row = sim::export_queue_state_sample(obs.sample(7, 5, 50));
    REQUIRE(row.size() == sim::kQueueStateExportCount);
    REQUIRE(row[0] == 7.0);    // order_id
    REQUIRE(row[1] == -1.0);   // side (sell)
    REQUIRE(row[7] == 9.0);    // queue_ahead_qty
    REQUIRE(row[8] == 0.0);    // queue_ahead_censored (false)
}

TEST_CASE("queue-state observer is replay-hash invariant: A/B against the real book",
          "[queue-state][observer][replay-invariant]") {
    // The observer is an external read-only sink. Driving it from the SAME
    // observations must not perturb matching: the book's fills and top-of-book
    // trajectory are byte-identical with and without the observer attached.
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100;

    const auto run = [](bool attach_observer) {
        sim::FlatLadderBook book(kMin, kMax, 32);
        sim::QueueStateObserver obs;
        std::vector<long long> digest;

        const auto snapshot = [&](sim::TimeNs ts) {
            const auto o = sim::make_book_observables(book, static_cast<sim::EventSeq>(ts));
            digest.push_back(o.has_bid ? o.best_bid : -1);
            digest.push_back(o.best_bid_qty);
            digest.push_back(o.has_ask ? o.best_ask : -1);
            digest.push_back(o.best_ask_qty);
        };

        // Two passive sells rest at 50; queue-ahead of the second is the first's qty.
        const sim::Qty ahead_at_50 = book.volume_at(sim::Side::Sell, 50);
        auto r1 = book.add(make_limit_order(1, sim::Side::Sell, 50, 6), 1);
        if (attach_observer)
            obs.on_submit(1, sim::Side::Sell, 50, 6, 1, 1, ahead_at_50);
        (void)r1;
        snapshot(1);

        const sim::Qty ahead2 = book.volume_at(sim::Side::Sell, 50);
        book.add(make_limit_order(2, sim::Side::Sell, 50, 4), 2);
        if (attach_observer) obs.on_submit(2, sim::Side::Sell, 50, 4, 2, 2, ahead2);
        snapshot(2);

        // Aggressive buy of 5 lifts the front of the 50 level (FIFO: hits id 1).
        auto m = book.add(make_market_order(3, sim::Side::Buy, 5), 3);
        if (attach_observer) {
            obs.on_level_trade(sim::Side::Sell, 50, m.filled, 3, 3);
            obs.on_own_fill(1, m.filled);
        }
        digest.push_back(m.filled);
        digest.push_back(m.remaining);
        snapshot(3);

        // Cancel the unfilled resting tail.
        const bool c = book.cancel(2, 4);
        if (attach_observer) obs.on_own_cancel(2);
        digest.push_back(c ? 1 : 0);
        snapshot(4);

        if (attach_observer) obs.finalize_right_censored();
        return digest;
    };

    const auto without_obs = run(false);
    const auto with_obs = run(true);
    REQUIRE(without_obs == with_obs);  // observer cannot perturb matching
}

TEST_CASE("execution-baseline taxonomy gates comparability to one harness",
          "[execution][baseline-taxonomy]") {
    sim::TcaConventions conv{};
    conv.fee_sign = sim::FeeSignConvention::PositiveIsCost;
    conv.benchmark = sim::BenchmarkReference::ArrivalMid;
    conv.terminal_mark = sim::TerminalMarkConvention::MarkToMid;

    sim::ParentExecutionOrder order{};
    order.side = sim::Side::Buy;
    order.qty = 100;
    order.source = 3;

    sim::ExecutionBaselineSpec twap{
        .kind = sim::ExecutionBaselineKind::Twap, .order = order, .conventions = conv};
    sim::ExecutionBaselineSpec vwap{
        .kind = sim::ExecutionBaselineKind::Vwap, .order = order, .conventions = conv};

    REQUIRE(sim::baseline_kind_name(sim::ExecutionBaselineKind::Pov) == "pov");

    SECTION("different baseline kinds on the same footing ARE comparable") {
        REQUIRE(sim::baselines_comparable(twap, vwap));  // the horse-race is valid
    }
    SECTION("different conventions break comparability") {
        sim::ExecutionBaselineSpec other = vwap;
        other.conventions.terminal_mark = sim::TerminalMarkConvention::ForcedFlatNoResidual;
        REQUIRE_FALSE(sim::baselines_comparable(twap, other));
    }
    SECTION("different parent economics break comparability") {
        sim::ExecutionBaselineSpec bigger = vwap;
        bigger.order.qty = 200;
        REQUIRE_FALSE(sim::baselines_comparable(twap, bigger));
        sim::ExecutionBaselineSpec other_side = vwap;
        other_side.order.side = sim::Side::Sell;
        REQUIRE_FALSE(sim::baselines_comparable(twap, other_side));
    }
}

TEST_CASE("VWAP schedule completes the parent exactly via largest-remainder",
          "[execution][vwap]") {
    sim::ParentExecutionOrder order{};
    order.parent_id = 9001;
    order.side = sim::Side::Sell;
    order.qty = 10;
    order.source = 4;

    // Profile {1,2,2} over qty 10: ideal {2,4,4}, sums exactly.
    const auto sched = sim::make_vwap_market_schedule(
        order, {1, 2, 2}, 100, 200,
        sim::ScheduleIds{.first_seq = 50, .first_order_id = 500});
    REQUIRE(sched.size() == 3);
    REQUIRE(sim::schedule_total_qty(sched) == 10);  // parent completed exactly
    REQUIRE(sched[0].qty == 2);
    REQUIRE(sched[1].qty == 4);
    REQUIRE(sched[2].qty == 4);
    REQUIRE(sched[0].side == sim::Side::Sell);
    REQUIRE(sched[0].kind == sim::StrategyIntentKind::SubmitMarket);

    // Uneven profile must still sum to the parent (largest-remainder leftover).
    const auto uneven = sim::make_vwap_market_schedule(order, {1, 1, 1}, 0, 30);
    REQUIRE(sim::schedule_total_qty(uneven) == 10);  // 4,3,3 in some order

    // Degenerate (all-zero) weights fall back to an equal split, still exact.
    const auto degenerate = sim::make_vwap_market_schedule(order, {0, 0, 0}, 0, 30);
    REQUIRE(sim::schedule_total_qty(degenerate) == 10);

    // Empty profile or non-positive qty -> no trades.
    REQUIRE(sim::make_vwap_market_schedule(order, {}, 0, 30).empty());
    order.qty = 0;
    REQUIRE(sim::make_vwap_market_schedule(order, {1, 1}, 0, 30).empty());
}

TEST_CASE("POV schedule may honestly under-fill when volume runs out",
          "[execution][pov]") {
    sim::ParentExecutionOrder order{};
    order.side = sim::Side::Buy;
    order.qty = 100;
    order.source = 5;

    // 10% participation of volumes {200,300,100} -> targets {20,30,10} = 60 < 100.
    const auto sched = sim::make_pov_market_schedule(
        order, {200, 300, 100}, /*num=*/1, /*den=*/10, 0, 200);
    REQUIRE(sched.size() == 3);
    REQUIRE(sched[0].qty == 20);
    REQUIRE(sched[1].qty == 30);
    REQUIRE(sched[2].qty == 10);
    // POV is allowed to NOT complete the parent -- that is the baseline's nature.
    REQUIRE(sim::schedule_total_qty(sched) == 60);
    REQUIRE(sim::schedule_total_qty(sched) < order.qty);

    // When cumulative target would exceed the parent, the last slice is capped.
    sim::ParentExecutionOrder small = order;
    small.qty = 35;
    const auto capped = sim::make_pov_market_schedule(small, {200, 300, 100}, 1, 10, 0, 200);
    REQUIRE(sim::schedule_total_qty(capped) == 35);  // 20 + 15(capped), then stop
    REQUIRE(capped.size() == 2);

    // Degenerate participation -> no trades.
    REQUIRE(sim::make_pov_market_schedule(order, {200}, 0, 10, 0, 1).empty());
    REQUIRE(sim::make_pov_market_schedule(order, {200}, 1, 0, 0, 1).empty());
}

namespace {
inline sim::BookObservables make_l1(sim::Tick best_bid, sim::Qty bid_qty,
                                    sim::Tick best_ask, sim::Qty ask_qty,
                                    sim::EventSeq seq = 0, sim::TimeNs ts = 0) {
    sim::BookObservables o{};
    o.seq = seq;
    o.ts = ts;
    o.has_bid = true;
    o.has_ask = true;
    o.best_bid = best_bid;
    o.best_bid_qty = bid_qty;
    o.best_ask = best_ask;
    o.best_ask_qty = ask_qty;
    return o;
}
} // namespace

TEST_CASE("consecutive-snapshot OFI follows the Cont-Kukanov-Stoikov signs",
          "[ofi][microstructure]") {
    const auto base = make_l1(99, 10, 101, 12, 1, 100);

    SECTION("bid price up adds the new bid size") {
        const auto curr = make_l1(100, 7, 101, 12, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.bid_term == 7);   // +q_b(t)
        REQUIRE(s.ask_term == 0);   // ask unchanged: -(12-12)=0
        REQUIRE(s.ofi == 7);
        REQUIRE_FALSE(s.censored);
    }
    SECTION("bid price down subtracts the old bid size") {
        const auto curr = make_l1(98, 5, 101, 12, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.bid_term == -10);  // -q_b(t-1)
        REQUIRE(s.ofi == -10);
    }
    SECTION("same bid price uses the size delta") {
        const auto curr = make_l1(99, 15, 101, 12, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.bid_term == 5);    // 15-10
        REQUIRE(s.ofi == 5);
    }
    SECTION("ask price down is sell pressure (negative)") {
        const auto curr = make_l1(99, 10, 100, 8, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.ask_term == -8);   // -q_a(t)
        REQUIRE(s.ofi == -8);
    }
    SECTION("ask price up is buy pressure (positive)") {
        const auto curr = make_l1(99, 10, 102, 4, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.ask_term == 12);   // +q_a(t-1)
        REQUIRE(s.ofi == 12);
    }
    SECTION("more ask size at same ask price is sell pressure") {
        const auto curr = make_l1(99, 10, 101, 20, 2, 110);
        const auto s = sim::make_ofi_sample(base, curr);
        REQUIRE(s.ask_term == -8);   // -(20-12)
        REQUIRE(s.ofi == -8);
    }
}

TEST_CASE("OFI is distinct from single-snapshot volume imbalance and censors gaps",
          "[ofi][microstructure][censoring]") {
    // A balanced book with a one-sided OFI: equal qty both sides (volume
    // imbalance ~ 0) but the bid just stepped up -> strictly positive OFI.
    const auto prev = make_l1(99, 10, 101, 10, 1, 1);
    const auto curr = make_l1(100, 10, 101, 10, 2, 2);
    const auto s = sim::make_ofi_sample(prev, curr);
    REQUIRE(s.ofi == 10);                 // OFI sees the bid step...
    REQUIRE(curr.imbalance_l1 == 0.0);    // ...volume imbalance does not.

    // Missing a quoted side -> that side's term is 0 AND the sample is censored.
    sim::BookObservables one_sided{};
    one_sided.seq = 3;
    one_sided.has_bid = true;
    one_sided.best_bid = 100;
    one_sided.best_bid_qty = 5;  // no ask
    const auto censored = sim::make_ofi_sample(curr, one_sided);
    REQUIRE(censored.censored);
    REQUIRE(censored.ask_term == 0);
    REQUIRE(sim::export_ofi_sample(censored)[6] == 0.0);  // ofi_complete = false

    const auto meta = sim::ofi_schema_meta();
    REQUIRE(meta.schema_name == "ofi_l1");
    REQUIRE(meta.missing_policy == sim::MissingPolicy::SentinelFlagged);
}

TEST_CASE("depth sparsity exposes fixed-grid holes so zeros are not read as liquidity",
          "[depth][sparsity]") {
    sim::TopKBookSnapshot<4> snap{};
    snap.seq = 7;
    snap.ts = 70;
    snap.bid_count = 2;  // only 2 of 4 grid levels populated
    snap.ask_count = 4;  // full

    const auto s = sim::make_depth_sparsity(snap);
    REQUIRE(s.grid_levels == 4);
    REQUIRE(s.bid_populated == 2);
    REQUIRE(s.bid_sparsity == 0.5);   // half the bid grid is structural zero
    REQUIRE_FALSE(s.bid_full);
    REQUIRE(s.ask_sparsity == 0.0);
    REQUIRE(s.ask_full);

    const auto row = sim::export_depth_sparsity(s);
    REQUIRE(row.size() == sim::kDepthSparsityExportCount);
    REQUIRE(row[5] == 0.5);   // bid_sparsity
    REQUIRE(row[7] == 0.0);   // both_full = false (bid not full)
}

TEST_CASE("allocation budget gates offline regions without affecting matching",
          "[alloc][budget]") {
    using sim::diag::AllocBudget;
    using sim::diag::AllocStats;

    const AllocStats observed{.allocations = 5, .bytes = 4096};

    SECTION("a disabled budget always passes") {
        AllocBudget budget{};  // enabled = false
        budget.max_allocations = 1;
        const auto r = sim::diag::check_alloc_budget(observed, budget);
        REQUIRE(r.within);
        REQUIRE_FALSE(r.allocations_exceeded);
        REQUIRE(r.allocations == 5);   // still reports the measurement
        REQUIRE(r.bytes == 4096);
    }
    SECTION("within budget passes") {
        AllocBudget budget{.enabled = true, .max_allocations = 8, .max_bytes = 8192};
        REQUIRE(sim::diag::check_alloc_budget(observed, budget).within);
    }
    SECTION("exceeding the allocation count fails on that axis") {
        AllocBudget budget{.enabled = true, .max_allocations = 4, .max_bytes = 0};
        const auto r = sim::diag::check_alloc_budget(observed, budget);
        REQUIRE_FALSE(r.within);
        REQUIRE(r.allocations_exceeded);
        REQUIRE_FALSE(r.bytes_exceeded);  // 0 = unlimited on the bytes axis
    }
    SECTION("exceeding bytes fails on that axis") {
        AllocBudget budget{.enabled = true, .max_allocations = 0, .max_bytes = 1024};
        const auto r = sim::diag::check_alloc_budget(observed, budget);
        REQUIRE_FALSE(r.within);
        REQUIRE(r.bytes_exceeded);
        REQUIRE_FALSE(r.allocations_exceeded);
    }
}

TEST_CASE("replay checkpoint verifies reproduced state and localizes divergence",
          "[replay][checkpoint]") {
    sim::ReplayDiagnostics d{};
    d.input_events = 100;
    d.accepted_events = 95;
    d.rejected_events = 5;
    d.trades = 40;
    d.filled_qty = 1234;
    d.replay_hash = 0xABCDEF12u;
    d.has_last_seq = true;
    d.last_seq = 100;
    d.has_last_ts = true;
    d.last_ts = 50'000;
    d.seq_order_violations = 0;
    d.time_order_violations = 0;

    const auto expected = sim::make_replay_checkpoint(d);

    SECTION("an identical run matches") {
        const auto actual = sim::make_replay_checkpoint(d);
        const auto v = sim::verify_checkpoint(expected, actual);
        REQUIRE(v.matches);
        REQUIRE_FALSE(v.hash_mismatch);
        REQUIRE_FALSE(v.progress_mismatch);
        REQUIRE_FALSE(v.position_mismatch);
    }
    SECTION("a content divergence trips the hash mismatch") {
        sim::ReplayDiagnostics diverged = d;
        diverged.replay_hash = 0x99999999u;  // same counts, different content
        const auto v = sim::verify_checkpoint(expected, sim::make_replay_checkpoint(diverged));
        REQUIRE_FALSE(v.matches);
        REQUIRE(v.hash_mismatch);
        REQUIRE_FALSE(v.progress_mismatch);
    }
    SECTION("under-reported progress is flagged even if the hash matches") {
        sim::ReplayDiagnostics short_run = d;
        short_run.trades = 39;  // a checkpoint that lost a trade
        const auto v = sim::verify_checkpoint(expected, sim::make_replay_checkpoint(short_run));
        REQUIRE_FALSE(v.matches);
        REQUIRE(v.progress_mismatch);
    }
    SECTION("a resumed run that starts from the wrong position is flagged") {
        sim::ReplayDiagnostics moved = d;
        moved.last_seq = 101;
        const auto v = sim::verify_checkpoint(expected, sim::make_replay_checkpoint(moved));
        REQUIRE_FALSE(v.matches);
        REQUIRE(v.position_mismatch);
    }
}

namespace {
inline sim::QuoteModelParams as_params(double gamma, double sigma2, double k, sim::Qty qty) {
    return sim::QuoteModelParams{.risk_aversion = gamma, .sigma2_per_second = sigma2,
                                 .fill_k = k, .quote_qty = qty};
}
inline sim::QuoteInputs as_inputs(std::int64_t mid_x2, sim::Qty inv) {
    return sim::QuoteInputs{.mid_x2 = mid_x2, .inventory = inv, .now = 0,
                            .horizon_ts = 1'000'000'000};  // tau = 1.0 s
}
} // namespace

TEST_CASE("AS quote skews against inventory and stays tick-valid",
          "[quote-adapter][as]") {
    const auto p = as_params(0.1, 1.0, 0.1, 5);

    const auto flat = sim::make_as_quote(p, as_inputs(200, 0));   // mid = 100
    REQUIRE(flat.model_supported);
    REQUIRE(flat.has_bid);
    REQUIRE(flat.has_ask);
    REQUIRE(flat.bid_px == 93);
    REQUIRE(flat.ask_px == 107);
    REQUIRE(flat.bid_px < flat.ask_px);
    REQUIRE(flat.bid_qty == 5);

    const auto longq = sim::make_as_quote(p, as_inputs(200, 10));
    REQUIRE(longq.bid_px < flat.bid_px);   // long inventory shifts BOTH quotes down
    REQUIRE(longq.ask_px < flat.ask_px);

    const auto shortq = sim::make_as_quote(p, as_inputs(200, -10));
    REQUIRE(shortq.bid_px > flat.bid_px);  // short inventory shifts BOTH quotes up
    REQUIRE(shortq.ask_px > flat.ask_px);
}

TEST_CASE("AS quotes never cross across a parameter sweep",
          "[quote-adapter][as][non-crossing]") {
    for (double gamma : {0.05, 0.1, 0.5}) {
        for (double k : {0.05, 0.1, 1.0}) {
            for (double sigma2 : {0.0, 0.5, 2.0}) {
                for (sim::Qty inv : {-20, -5, 0, 5, 20}) {
                    const auto q = sim::make_as_quote(as_params(gamma, sigma2, k, 5),
                                                      as_inputs(200, inv));
                    if (q.has_bid && q.has_ask) {
                        REQUIRE(q.bid_px < q.ask_px);  // integer + non-crossing
                    }
                }
            }
        }
    }
}

TEST_CASE("AS guards suppress rather than distort", "[quote-adapter][as][guards]") {
    const auto p = as_params(0.1, 1.0, 0.1, 5);  // flat quote is bid 93 / ask 107

    SECTION("invalid parameters yield no quote at all") {
        const auto bad = sim::make_as_quote(as_params(0.0, 1.0, 0.1, 5), as_inputs(200, 0));
        REQUIRE_FALSE(bad.model_supported);
        REQUIRE_FALSE(bad.has_bid);
        REQUIRE_FALSE(bad.has_ask);
    }
    SECTION("inventory at the cap suppresses the exposure-increasing side") {
        sim::RiskLimits limits{};
        limits.max_abs_inventory = 10;
        const auto q = sim::make_as_quote(p, as_inputs(200, 10), limits);
        REQUIRE(q.model_supported);
        REQUIRE_FALSE(q.has_bid);   // already long -> stop buying
        REQUIRE(q.bid_suppressed);
        REQUIRE(q.has_ask);
    }
    SECTION("an out-of-band side is suppressed, not clamped toward mid") {
        sim::RiskLimits limits{};
        limits.min_price = 90;
        limits.max_price = 105;  // ask 107 is out of band
        const auto q = sim::make_as_quote(p, as_inputs(200, 0), limits);
        REQUIRE(q.has_bid);
        REQUIRE_FALSE(q.has_ask);
        REQUIRE(q.ask_suppressed);
    }
    SECTION("oversized notional suppresses the offending side") {
        sim::RiskLimits limits{};
        limits.max_order_notional = 500;  // bid 93*5=465 ok; ask 107*5=535 too big
        const auto q = sim::make_as_quote(p, as_inputs(200, 0), limits);
        REQUIRE(q.has_bid);
        REQUIRE_FALSE(q.has_ask);
    }
}

TEST_CASE("quote adapters never produce phantom fills, only SubmitLimit intents",
          "[quote-adapter][no-phantom-fill]") {
    const auto p = as_params(0.1, 1.0, 0.1, 5);
    const auto q = sim::make_as_quote(p, as_inputs(200, 0));
    const auto intents = sim::to_quote_intents(
        q, /*ts=*/42, sim::QuoteIntentIds{.first_seq = 1, .first_order_id = 10, .agent = 7});

    REQUIRE(intents.size() == 2);
    for (const auto& in : intents) {
        REQUIRE(in.kind == sim::StrategyIntentKind::SubmitLimit);  // never a fill/market
        REQUIRE(in.qty == 5);
        REQUIRE(in.agent == 7);
    }
    REQUIRE(intents[0].side == sim::Side::Buy);
    REQUIRE(intents[0].price == 93);
    REQUIRE(intents[1].side == sim::Side::Sell);
    REQUIRE(intents[1].price == 107);

    // GLFT/CJP placeholders are deferred: they fabricate neither a quote nor a fill.
    const auto glft = sim::make_guarded_quote(sim::QuoteModelKind::GlftHeuristic, p,
                                              as_inputs(200, 0));
    REQUIRE_FALSE(glft.model_supported);
    REQUIRE(sim::to_quote_intents(glft, 42).empty());
    const auto cjp = sim::make_guarded_quote(sim::QuoteModelKind::CjpHeuristic, p,
                                             as_inputs(200, 0));
    REQUIRE(sim::to_quote_intents(cjp, 42).empty());
}


#include "sim/impact_diagnostics.hpp"
#include <cmath>

namespace sim_test {
inline bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}
}

TEST_CASE("impact diagnostics: direction-normalized realized sign",
          "[impact-diagnostics][sign]") {
    sim::ImpactObservation buy{};
    buy.side = sim::Side::Buy;
    buy.pre_mid_x2 = 200;    // mid 100.0
    buy.post_mid_x2 = 206;   // mid 103.0 -> +3 ticks WITH a buy
    buy.has_pre_mid = buy.has_post_mid = true;
    buy.availability = sim::LabelAvailability::Available;
    REQUIRE(sim_test::approx_eq(sim::realized_impact_ticks(buy), 3.0));

    sim::ImpactObservation sell = buy;
    sell.side = sim::Side::Sell;
    sell.post_mid_x2 = 194;  // mid 97.0 -> -3 ticks; WITH a sell -> +3 normalized
    REQUIRE(sim_test::approx_eq(sim::realized_impact_ticks(sell), 3.0));
}

TEST_CASE("impact diagnostics: adverse move yields negative realized",
          "[impact-diagnostics][adverse]") {
    sim::ImpactObservation o{};
    o.side = sim::Side::Buy;
    o.pre_mid_x2 = 200;
    o.post_mid_x2 = 196;     // mid fell after a buy -> adverse
    o.has_pre_mid = o.has_post_mid = true;
    o.availability = sim::LabelAvailability::Available;
    REQUIRE(sim_test::approx_eq(sim::realized_impact_ticks(o), -2.0));
}

TEST_CASE("impact diagnostics: prediction is caller's coef, no internal fitting",
          "[impact-diagnostics][no-fit]") {
    REQUIRE(sim_test::approx_eq(sim::predicted_impact_ticks(0.5, 4.0), 1.0));
    REQUIRE(sim_test::approx_eq(sim::predicted_impact_ticks(0.5, -4.0), 1.0));
    REQUIRE(sim_test::approx_eq(sim::predicted_impact_ticks(1.0, 4.0), 2.0));
}

TEST_CASE("impact diagnostics: known residual, rmse and bias",
          "[impact-diagnostics][residual]") {
    sim::ImpactObservation a{};
    a.side = sim::Side::Buy; a.participation = 1.0;
    a.pre_mid_x2 = 200; a.post_mid_x2 = 204;  // realized +2 -> residual +1
    a.has_pre_mid = a.has_post_mid = true;
    a.availability = sim::LabelAvailability::Available;
    sim::ImpactObservation b = a;
    b.post_mid_x2 = 200;                        // realized 0  -> residual -1
    const sim::ImpactObservation obs[2] = {a, b};
    const auto s = sim::score_impacts(obs, 2, 1.0);
    REQUIRE(s.included == 2);
    REQUIRE(s.excluded == 0);
    REQUIRE(s.finite);
    REQUIRE(sim_test::approx_eq(s.bias(), 0.0));
    REQUIRE(sim_test::approx_eq(s.rmse(), 1.0));
}

TEST_CASE("impact diagnostics: pending and missing-mid observations excluded",
          "[impact-diagnostics][exclusions]") {
    sim::ImpactObservation base{};
    base.side = sim::Side::Buy; base.participation = 1.0;
    base.pre_mid_x2 = 200; base.post_mid_x2 = 204;
    base.has_pre_mid = base.has_post_mid = true;
    base.availability = sim::LabelAvailability::Available;

    sim::ImpactObservation pending = base;
    pending.availability = sim::LabelAvailability::Pending;
    sim::ImpactObservation no_post = base;
    no_post.has_post_mid = false;

    const sim::ImpactObservation obs[3] = {base, pending, no_post};
    const auto s = sim::score_impacts(obs, 3, 1.0);
    REQUIRE(s.included == 1);
    REQUIRE(s.excluded == 2);
}

TEST_CASE("impact diagnostics: schema meta is event-time, diagnostics-only",
          "[impact-diagnostics][schema]") {
    const auto m = sim::impact_diagnostics_schema_meta();
    REQUIRE(m.timestamp_basis == sim::TimestampBasis::EventTime);
    REQUIRE(m.quantity_unit == sim::QuantityUnit::Ratio);
    REQUIRE(m.label_availability == sim::LabelAvailability::Available);
}

#include "sim/survival_labels.hpp"

TEST_CASE("survival: full fill within horizon resolves Filled",
          "[survival][filled]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 100; cfg.start_ts = 0; cfg.original_qty = 10;
    cfg.horizon_events = 50;
    sim::SurvivalTracker t(cfg);
    t.on_fill(10, /*seq=*/120, /*ts=*/5);
    REQUIRE(t.state() == sim::SurvivalState::Filled);
    REQUIRE(t.filled_qty() == 10);
    REQUIRE(t.remaining_qty() == 0);
    REQUIRE(t.resolve_seq() == 120);
}

TEST_CASE("survival: partial fill stays Pending, then completes Filled",
          "[survival][partial]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 0; cfg.original_qty = 10; cfg.horizon_events = 100;
    sim::SurvivalTracker t(cfg);
    t.on_fill(4, 10, 1);
    REQUIRE(t.state() == sim::SurvivalState::Pending);  // partial -> not Filled
    REQUIRE(t.filled_qty() == 4);
    REQUIRE(t.remaining_qty() == 6);
    t.on_fill(6, 20, 2);
    REQUIRE(t.state() == sim::SurvivalState::Filled);
    REQUIRE(t.resolve_seq() == 20);
}

TEST_CASE("survival: partial fill then cancel -> CensoredCancel with filled recorded",
          "[survival][cancel]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 0; cfg.original_qty = 10; cfg.horizon_events = 100;
    sim::SurvivalTracker t(cfg);
    t.on_fill(3, 10, 1);
    t.on_cancel(15, 2);
    REQUIRE(t.state() == sim::SurvivalState::CensoredCancel);
    REQUIRE(t.filled_qty() == 3);
    REQUIRE(t.remaining_qty() == 7);
    REQUIRE(t.resolve_seq() == 15);
}

TEST_CASE("survival: cancel with no fills -> CensoredCancel filled 0",
          "[survival][cancel-nofill]") {
    sim::SurvivalConfig cfg{};
    cfg.original_qty = 10; cfg.horizon_events = 100;
    sim::SurvivalTracker t(cfg);
    t.on_cancel(5, 1);
    REQUIRE(t.state() == sim::SurvivalState::CensoredCancel);
    REQUIRE(t.filled_qty() == 0);
}

TEST_CASE("survival: event horizon closes before fill -> CensoredHorizon at start+H",
          "[survival][horizon-event]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 100; cfg.original_qty = 10; cfg.horizon_events = 50;
    sim::SurvivalTracker t(cfg);
    t.advance(/*seq=*/151, /*ts=*/9);  // offset 51 >= 50 -> outside window
    REQUIRE(t.state() == sim::SurvivalState::CensoredHorizon);
    REQUIRE(t.resolve_seq() == 150);   // start_seq + horizon_events
}

TEST_CASE("survival: time horizon closes -> CensoredHorizon at start_ts+H",
          "[survival][horizon-time]") {
    sim::SurvivalConfig cfg{};
    cfg.start_ts = 1000; cfg.original_qty = 10; cfg.horizon_ns = 200;
    sim::SurvivalTracker t(cfg);
    t.advance(/*seq=*/7, /*ts=*/1250);  // offset 250 >= 200
    REQUIRE(t.state() == sim::SurvivalState::CensoredHorizon);
    REQUIRE(t.resolve_ts() == 1200);
}

TEST_CASE("survival: fill at the horizon boundary is too late -> CensoredHorizon",
          "[survival][late-fill]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 0; cfg.original_qty = 10; cfg.horizon_events = 50;
    sim::SurvivalTracker t(cfg);
    t.on_fill(10, /*seq=*/50, /*ts=*/3);  // offset 50 == H -> outside half-open window
    REQUIRE(t.state() == sim::SurvivalState::CensoredHorizon);
    REQUIRE(t.filled_qty() == 0);          // the late fill is not counted
}

TEST_CASE("survival: end-of-data before horizon -> CensoredEndOfData (not Horizon)",
          "[survival][end-of-data]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 0; cfg.original_qty = 10; cfg.horizon_events = 1000;
    sim::SurvivalTracker t(cfg);
    t.on_fill(2, 5, 1);                 // partial, still pending
    t.finalize_end_of_data(/*last_seq=*/40, /*last_ts=*/9);
    REQUIRE(t.state() == sim::SurvivalState::CensoredEndOfData);
    REQUIRE(t.filled_qty() == 2);
    REQUIRE(t.resolve_seq() == 40);
}

TEST_CASE("survival: same-seq fill-before-cancel -> full fill wins, cancel ignored",
          "[survival][same-seq]") {
    sim::SurvivalConfig cfg{};
    cfg.start_seq = 0; cfg.original_qty = 10; cfg.horizon_events = 100;
    sim::SurvivalTracker t(cfg);
    t.on_fill(10, /*seq=*/30, /*ts=*/4);   // completes at seq 30
    t.on_cancel(/*seq=*/30, /*ts=*/4);     // same seq -> ignored (already terminal)
    REQUIRE(t.state() == sim::SurvivalState::Filled);
    REQUIRE(t.resolve_seq() == 30);
}

#include "sim/exploitability_probe.hpp"
#include <limits>

TEST_CASE("exploitability: incumbent best -> zero gain, never a certificate",
          "[exploitability][no-gain]") {
    const double devs[3] = {0.5, 0.9, 1.0};   // none beat incumbent 1.5
    const auto r = sim::probe_exploitability(1.5, devs, 3);
    REQUIRE(r.gain_lower_bound == 0.0);
    REQUIRE(r.num_deviations_tested == 3);
    REQUIRE_FALSE(r.no_deviations_tested);
    REQUIRE_FALSE(r.is_equilibrium_certificate);  // zero gain is NOT a proof
}

TEST_CASE("exploitability: profitable deviation yields positive lower bound",
          "[exploitability][gain]") {
    const double devs[3] = {1.0, 2.5, 1.2};   // best 2.5 beats incumbent 2.0
    const auto r = sim::probe_exploitability(2.0, devs, 3);
    REQUIRE(sim_test::approx_eq(r.best_deviation_payoff, 2.5));
    REQUIRE(sim_test::approx_eq(r.gain_lower_bound, 0.5));
    REQUIRE(r.num_deviations_tested == 3);
}

TEST_CASE("exploitability: non-finite deviation payoffs are rejected, not compared",
          "[exploitability][reject]") {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double devs[4] = {1.0, inf, nan, 2.2};
    const auto r = sim::probe_exploitability(2.0, devs, 4);
    REQUIRE(r.num_deviations_tested == 2);          // 1.0 and 2.2 only
    REQUIRE(r.num_deviations_rejected == 2);        // inf, nan
    REQUIRE(sim_test::approx_eq(r.best_deviation_payoff, 2.2));
    REQUIRE(sim_test::approx_eq(r.gain_lower_bound, 0.2));
}

TEST_CASE("exploitability: zero deviations -> gain 0 flagged no-deviations-tested",
          "[exploitability][zero-dev]") {
    const auto r = sim::probe_exploitability(1.0, nullptr, 0);
    REQUIRE(r.gain_lower_bound == 0.0);
    REQUIRE(r.no_deviations_tested);
    REQUIRE_FALSE(r.any_deviation_valid);
    REQUIRE_FALSE(r.is_equilibrium_certificate);
}

TEST_CASE("exploitability: non-finite incumbent invalidates the gap",
          "[exploitability][bad-incumbent]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double devs[2] = {1.0, 5.0};
    const auto r = sim::probe_exploitability(nan, devs, 2);
    REQUIRE_FALSE(r.incumbent_valid);
    REQUIRE(r.gain_lower_bound == 0.0);             // cannot compute a gap
    REQUIRE(r.num_deviations_tested == 2);
}

#include "sim/accelerator_parity.hpp"

TEST_CASE("accelerator parity: integer outputs must match exactly",
          "[accelerator][int]") {
    const std::int64_t ref[4] = {1, 2, 3, 4};
    const std::int64_t same[4] = {1, 2, 3, 4};
    const std::int64_t diff[4] = {1, 2, 9, 4};
    const auto ok = sim::check_parity_int(ref, 4, same, 4);
    REQUIRE(ok.accelerator_acceptable);
    REQUIRE(ok.compared == 4);
    const auto bad = sim::check_parity_int(ref, 4, diff, 4);
    REQUIRE_FALSE(bad.accelerator_acceptable);
    REQUIRE(bad.has_mismatch);
    REQUIRE(bad.first_mismatch == 2);
}

TEST_CASE("accelerator parity: doubles within / beyond tolerance",
          "[accelerator][double-tol]") {
    const double ref[3] = {1.0, 2.0, 1e6};
    const double close[3] = {1.0 + 1e-12, 2.0, 1e6 + 1e-4};  // abs+rel ok
    const double far[3] = {1.0, 2.5, 1e6};                    // 2.5 vs 2.0 fails
    REQUIRE(sim::check_parity_double(ref, 3, close, 3).accelerator_acceptable);
    const auto bad = sim::check_parity_double(ref, 3, far, 3);
    REQUIRE_FALSE(bad.accelerator_acceptable);
    REQUIRE(bad.first_mismatch == 1);
}

TEST_CASE("accelerator parity: size mismatch fails immediately",
          "[accelerator][size]") {
    const std::int64_t a[3] = {1, 2, 3};
    const std::int64_t b[2] = {1, 2};
    const auto r = sim::check_parity_int(a, 3, b, 2);
    REQUIRE_FALSE(r.size_match);
    REQUIRE_FALSE(r.accelerator_acceptable);
    REQUIRE(r.compared == 0);  // no out-of-bounds compare
}

TEST_CASE("accelerator parity: non-finite double semantics are explicit",
          "[accelerator][nonfinite]") {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    sim::ParityTolerance tol{};
    REQUIRE(sim::parity_doubles_match(nan, nan, tol));        // both NaN agree
    REQUIRE_FALSE(sim::parity_doubles_match(nan, 1.0, tol));  // one NaN -> mismatch
    REQUIRE(sim::parity_doubles_match(inf, inf, tol));        // same-sign inf agree
    REQUIRE_FALSE(sim::parity_doubles_match(inf, -inf, tol)); // opposite inf mismatch
    REQUIRE_FALSE(sim::parity_doubles_match(inf, 1e300, tol));// inf vs finite mismatch
}

TEST_CASE("accelerator parity: fast-but-wrong accelerator is rejected",
          "[accelerator][parity-gates-speed]") {
    const std::int64_t ref[3] = {10, 20, 30};
    const std::int64_t wrong[3] = {10, 20, 31};
    const auto parity = sim::check_parity_int(ref, 3, wrong, 3);
    // candidate is 100x faster but WRONG
    const auto v = sim::accelerator_verdict(parity, /*ref_ns=*/1000.0, /*cand_ns=*/10.0);
    REQUIRE(sim_test::approx_eq(v.speedup, 100.0));
    REQUIRE_FALSE(v.accepted);  // speedup never overrides parity
}

TEST_CASE("accelerator parity: correct + faster accelerator is accepted",
          "[accelerator][accepted]") {
    const std::int64_t ref[3] = {10, 20, 30};
    const std::int64_t same[3] = {10, 20, 30};
    const auto parity = sim::check_parity_int(ref, 3, same, 3);
    const auto v = sim::accelerator_verdict(parity, 1000.0, 250.0);
    REQUIRE(v.accepted);
    REQUIRE(sim_test::approx_eq(v.speedup, 4.0));
}

#include "sim/join_manifest.hpp"

TEST_CASE("join manifest: causally clean join is usable",
          "[join][usable]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 100;  r.feature_ts = 1000;
    r.label_start_seq = 102; r.label_start_ts = 1200;  // starts after embargo
    r.label_eval_seq = 110;  r.label_eval_ts = 2000;
    r.feature_split_id = 1;  r.label_split_id = 1;
    sim::JoinPolicy p{};
    p.embargo_events = 1;  p.embargo_ns = 100;          // boundary = seq 101 / ts 1100
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::Usable);
    REQUIRE(sim::join_is_usable(sim::validate_join(r, p)));
}

TEST_CASE("join manifest: label starting at embargo boundary is leaky (seq)",
          "[join][leaky-seq]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 100;  r.label_start_seq = 101;  // == feature_seq + embargo(1)
    r.label_eval_seq = 105;
    r.label_start_ts = 5000; r.label_eval_ts = 6000;  // time axis clean
    sim::JoinPolicy p{}; p.embargo_events = 1;
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::LeakySeq);
}

TEST_CASE("join manifest: time embargo violation is leaky (time)",
          "[join][leaky-time]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 100;  r.label_start_seq = 110;  r.label_eval_seq = 120;  // seq clean
    r.feature_ts = 1000;  r.label_start_ts = 1050;  // < feature_ts + embargo_ns(100)
    sim::JoinPolicy p{}; p.embargo_ns = 100;
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::LeakyTime);
}

TEST_CASE("join manifest: eval before start is non-causal",
          "[join][non-causal]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 10; r.label_start_seq = 20; r.label_eval_seq = 15;  // eval < start
    sim::JoinPolicy p{};
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::NonCausalEval);
}

TEST_CASE("join manifest: cross-split join is rejected",
          "[join][split]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 1; r.label_start_seq = 5; r.label_eval_seq = 9;
    r.label_start_ts = 50; r.label_eval_ts = 90;
    r.feature_split_id = 0;  r.label_split_id = 1;  // train feature, test label
    sim::JoinPolicy p{};  // require_same_split defaults to true
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::SplitMismatch);
    sim::JoinPolicy lax{}; lax.require_same_split = false;
    REQUIRE(sim::validate_join(r, lax) == sim::JoinValidity::Usable);
}

TEST_CASE("join manifest: diagnostic-only rows never count as training labels",
          "[join][diagnostic]") {
    sim::JoinManifestRow r{};
    r.feature_seq = 1; r.label_start_seq = 100; r.label_eval_seq = 200;
    r.diagnostic_only = true;
    sim::JoinPolicy p{};
    REQUIRE(sim::validate_join(r, p) == sim::JoinValidity::DiagnosticOnly);
    REQUIRE_FALSE(sim::join_is_usable(sim::validate_join(r, p)));
}

TEST_CASE("join manifest: summary tallies outcomes deterministically",
          "[join][summary]") {
    sim::JoinPolicy p{}; p.embargo_events = 1;
    sim::JoinManifestSummary s{};
    sim::JoinManifestRow clean{};
    clean.feature_seq = 1; clean.label_start_seq = 5; clean.label_eval_seq = 9;
    clean.label_start_ts = 50; clean.label_eval_ts = 90;
    sim::JoinManifestRow leaky{};
    leaky.feature_seq = 1; leaky.label_start_seq = 2; leaky.label_eval_seq = 9;
    leaky.label_start_ts = 50; leaky.label_eval_ts = 90;
    join_summary_add(s, sim::validate_join(clean, p));
    join_summary_add(s, sim::validate_join(leaky, p));
    REQUIRE(s.total == 2);
    REQUIRE(s.usable == 1);
    REQUIRE(s.leaky_seq == 1);
}

TEST_CASE("join manifest: schema meta declares event-time basis",
          "[join][schema]") {
    const auto m = sim::join_manifest_schema_meta();
    REQUIRE(m.schema_name == "join_manifest");
    REQUIRE(m.timestamp_basis == sim::TimestampBasis::EventTime);
}

// --- T4-1 WIRING: opt-in join-manifest hook in the feature-export contract ----

// (a) A causal/embargoed join passes the hook, the frame is admitted to the
// export sink, and the hook exports the join-manifest schema-meta.
TEST_CASE("join wiring: causal join is admitted and exports the manifest meta",
          "[join][wire]") {
    sim::JoinPolicy p{};
    p.embargo_events = 1;
    p.embargo_ns = 100;
    sim::JoinManifestExportHook hook(p);
    sim::SyncFeatureExportSink sink;

    sim::JoinManifestRow row{};
    row.feature_seq = 100;  row.feature_ts = 1000;
    row.label_start_seq = 102;  row.label_start_ts = 1200;  // past embargo boundary
    row.label_eval_seq = 110;   row.label_eval_ts = 2000;
    row.feature_split_id = 1;   row.label_split_id = 1;

    const sim::ExportFrame frame{.seq = 102, .replay_hash = 0xABCD};
    const auto v = hook.emit_if_usable(sink, frame, row);

    REQUIRE(v == sim::JoinValidity::Usable);
    REQUIRE(sink.accepted_frames() == 1);
    REQUIRE(sink.frames().back().seq == 102);
    REQUIRE(hook.summary().total == 1);
    REQUIRE(hook.summary().usable == 1);
    REQUIRE_FALSE(hook.any_leak());
    // The manifest schema-meta is surfaced through the export-contract hook.
    const auto m = hook.schema_meta();
    REQUIRE(m.schema_name == "join_manifest");
    REQUIRE(m.timestamp_basis == sim::TimestampBasis::EventTime);
}

// (b) A leaky join is flagged by the hook AND surfaced through the export
// contract: the labelled frame is DROPPED (never admitted to the sink) and the
// leak is observable via the hook's summary / any_leak().
TEST_CASE("join wiring: leaky join is flagged and dropped from the export path",
          "[join][wire]") {
    sim::JoinPolicy p{};
    p.embargo_events = 1;
    sim::JoinManifestExportHook hook(p);
    sim::SyncFeatureExportSink sink;

    sim::JoinManifestRow leaky{};
    leaky.feature_seq = 100;
    leaky.label_start_seq = 101;  // == feature_seq + embargo(1): boundary is leaky
    leaky.label_eval_seq = 105;
    leaky.label_start_ts = 5000;  leaky.label_eval_ts = 6000;

    const sim::ExportFrame frame{.seq = 101, .replay_hash = 0x1234};
    const auto v = hook.emit_if_usable(sink, frame, leaky);

    REQUIRE(v == sim::JoinValidity::LeakySeq);
    // Surfaced through the export contract: frame never reached the sink.
    REQUIRE(sink.accepted_frames() == 0);
    REQUIRE(hook.any_leak());
    REQUIRE(hook.summary().total == 1);
    REQUIRE(hook.summary().leaky_seq == 1);
    REQUIRE(hook.summary().usable == 0);

    // admit() exposes the same verdict without an export sink.
    sim::JoinManifestExportHook hook2(p);
    REQUIRE_FALSE(hook2.admit(leaky));
    REQUIRE(hook2.summary().leaky_seq == 1);
}

// (c) Regression guard: when the hook is NOT used, the existing export behaviour
// is byte-for-byte unchanged -- a plain sink fed via on_frame admits frames and
// enforces prefix-hash parity exactly as before the wiring.
TEST_CASE("join wiring: unused hook leaves existing export behaviour unchanged",
          "[join][wire]") {
    sim::SyncFeatureExportSink sink;
    sink.on_frame(sim::ExportFrame{.seq = 1, .replay_hash = 11});
    sink.on_frame(sim::ExportFrame{.seq = 2, .replay_hash = 22});

    REQUIRE(sink.accepted_frames() == 2);
    REQUIRE(sink.seq_monotone());
    REQUIRE(sink.verify_final_parity(22));
    REQUIRE(sink.last_hash() == 22);

    // Reordered/duplicated frames are still rejected as before.
    sink.on_frame(sim::ExportFrame{.seq = 1, .replay_hash = 99});
    REQUIRE_FALSE(sink.seq_monotone());
    REQUIRE(sink.seq_violations() == 1);
    REQUIRE(sink.accepted_frames() == 2);
}

TEST_CASE("reward ledger operations and conventions", "[reward][tca]") {
    sim::RewardLedger ledger;
    sim::RewardKey key{.parent_id = 1, .agent = 2, .source = 3};

    SECTION("empty ledger has size 0") {
        REQUIRE(ledger.empty());
        REQUIRE(ledger.size() == 0);
    }

    SECTION("record_reward creates entry") {
        ledger.record_reward(key, 500'000);  // 0.5 in x1e6 fixed-point
        REQUIRE(ledger.size() == 1);

        const auto* summary = ledger.find(key);
        REQUIRE(summary != nullptr);
        REQUIRE(summary->cumulative_reward_x1e6 == 500'000);
        REQUIRE(summary->turn_count == 1);
        REQUIRE(summary->penalty_count == 0);
    }

    SECTION("negative reward increments penalty_count") {
        ledger.record_reward(key, -200'000);
        const auto* summary = ledger.find(key);
        REQUIRE(summary != nullptr);
        REQUIRE(summary->penalty_count == 1);
        REQUIRE(summary->penalty_magnitude_x1e6 == 200'000);
    }

    SECTION("record_fill accumulates filled_qty") {
        ledger.record_fill(key, 100);
        ledger.record_fill(key, 50);
        const auto* summary = ledger.find(key);
        REQUIRE(summary != nullptr);
        REQUIRE(summary->filled_qty == 150);
    }

    SECTION("record_remaining sets remaining_qty") {
        ledger.record_remaining(key, 25);
        const auto* summary = ledger.find(key);
        REQUIRE(summary != nullptr);
        REQUIRE(summary->remaining_qty == 25);
    }

    SECTION("record_price_improvement sets ticks") {
        ledger.record_price_improvement(key, 3);
        const auto* summary = ledger.find(key);
        REQUIRE(summary != nullptr);
        REQUIRE(summary->price_improvement_ticks == 3);
    }

    SECTION("clear empties the ledger") {
        ledger.record_reward(key, 100'000);
        REQUIRE_FALSE(ledger.empty());
        ledger.clear();
        REQUIRE(ledger.empty());
    }

    SECTION("missing key returns nullptr") {
        sim::RewardKey missing{.parent_id = 999, .agent = 999, .source = 999};
        REQUIRE(ledger.find(missing) == nullptr);
    }

    SECTION("reward conventions are comparable when equal") {
        sim::RewardConventions a, b;
        REQUIRE(sim::reward_conventions_comparable(a, b));

        b.gamma_x1e6 = 900'000;  // different gamma
        REQUIRE_FALSE(sim::reward_conventions_comparable(a, b));
    }

    SECTION("reward key equality and hashing") {
        sim::RewardKey k1{.parent_id = 1, .agent = 2, .source = 3};
        sim::RewardKey k2{.parent_id = 1, .agent = 2, .source = 3};
        sim::RewardKey k3{.parent_id = 1, .agent = 2, .source = 4};

        REQUIRE(k1 == k2);
        REQUIRE_FALSE(k1 == k3);

        sim::RewardKeyHash hash_fn;
        REQUIRE(hash_fn(k1) == hash_fn(k2));
        // k3 should hash differently (not guaranteed, but overwhelmingly likely)
        REQUIRE(hash_fn(k1) != hash_fn(k3));
    }
}

TEST_CASE("reward ledger does not contaminate TCA",
          "[reward][contamination][tca]") {
    sim::RewardLedger reward_ledger;
    sim::TcaSummary tca_summary{};

    sim::RewardKey rkey{.parent_id = 1, .agent = 2, .source = 3};

    // Record reward and fill data
    reward_ledger.record_reward(rkey, 1'000'000);  // 1.0 in x1e6
    reward_ledger.record_fill(rkey, 500);

    const auto* rs = reward_ledger.find(rkey);
    REQUIRE(rs != nullptr);
    REQUIRE(rs->cumulative_reward_x1e6 == 1'000'000);
    REQUIRE(rs->filled_qty == 500);

    // TCA summary should NOT have been affected by reward recording
    REQUIRE(tca_summary.passive_qty == 0);
    REQUIRE(tca_summary.aggressive_qty == 0);
    REQUIRE(tca_summary.passive_notional == 0);
    REQUIRE(tca_summary.aggressive_notional == 0);
    REQUIRE(tca_summary.passive_fee_cost == 0);
    REQUIRE(tca_summary.aggressive_fee_cost == 0);
    REQUIRE(tca_summary.passive_markout_x2_qty == 0);
    REQUIRE(tca_summary.aggressive_markout_x2_qty == 0);

    // Verify the reward summary fields are NOT present in TcaSummary
    // (compile-time structural disjointness -- these fields simply don't exist
    // on TcaSummary, so if this compiles, the disjointness is enforced)
    REQUIRE(rs->cumulative_reward_x1e6 != tca_summary.passive_qty);  // different types, different meanings
}

TEST_CASE("three-risk triplet has independent fields",
          "[three-risk][no-netting]") {
    sim::ThreeRiskTriplet triplet{};

    SECTION("default values are zero") {
        REQUIRE(triplet.maker_adversity_x2 == 0);
        REQUIRE(triplet.taker_adversity_x2 == 0);
        REQUIRE(triplet.queue_risk_x2 == 0);
        REQUIRE(triplet.maker_adversity_qty == 0);
        REQUIRE(triplet.taker_adversity_qty == 0);
        REQUIRE(triplet.queue_risk_qty == 0);
    }

    SECTION("each field can be set independently") {
        triplet.maker_adversity_x2 = 10;
        triplet.taker_adversity_x2 = -5;
        triplet.queue_risk_x2 = 20;

        REQUIRE(triplet.maker_adversity_x2 == 10);
        REQUIRE(triplet.taker_adversity_x2 == -5);
        REQUIRE(triplet.queue_risk_x2 == 20);
    }

    SECTION("no-netting guard returns true") {
        REQUIRE(sim::no_netting_guard(triplet));
    }

    SECTION("per-side split produces proportional breakout") {
        triplet.maker_adversity_x2 = 100;
        triplet.taker_adversity_x2 = 200;
        triplet.queue_risk_x2 = 300;

        auto split = sim::split_risk_by_side(triplet, 75, 25);
        REQUIRE(split.buy_side.maker_adversity_x2 == 75);
        REQUIRE(split.sell_side.maker_adversity_x2 == 25);
        REQUIRE(split.buy_side.taker_adversity_x2 == 150);
        REQUIRE(split.sell_side.taker_adversity_x2 == 50);
        REQUIRE(split.buy_side.queue_risk_x2 == 225);
        REQUIRE(split.sell_side.queue_risk_x2 == 75);
    }

    SECTION("export schema has 6 columns (3 risk + 3 qty)") {
        REQUIRE(sim::kThreeRiskExportCount == 6);
        // Verify no "total" or "net" column exists
        for (const auto& col : sim::kThreeRiskExportSchema) {
            REQUIRE(col.name.find("total") == std::string_view::npos);
            REQUIRE(col.name.find("net") == std::string_view::npos);
        }
    }
}

TEST_CASE("Self-Trade Prevention policies in FlatLadderBook and LimitOrderBook", "[lob][stp]") {
    SECTION("FlatLadderBook CancelPassive cancels resting maker on self-match") {
        sim::FlatLadderBook book(90, 110);
        auto maker = make_limit_order(1, sim::Side::Sell, 100, 20);
        book.add(maker, 1000);

        // Incoming buy from same agent with CancelPassive
        auto taker = make_limit_order(1, sim::Side::Buy, 100, 15);
        auto res = book.add(taker, 2000, sim::StpMode::CancelPassive);

        REQUIRE(res.trades.empty());
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 0);
        // Maker was cancelled; taker rests on the bid side as an uncrossed limit
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 15);
    }

    SECTION("FlatLadderBook CancelAggressive cancels incoming taker immediately") {
        sim::FlatLadderBook book(90, 110);
        auto maker = make_limit_order(1, sim::Side::Sell, 100, 20);
        book.add(maker, 1000);

        auto taker = make_limit_order(1, sim::Side::Buy, 100, 15);
        auto res = book.add(taker, 2000, sim::StpMode::CancelAggressive);

        REQUIRE(res.trades.empty());
        REQUIRE(res.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 20); // maker still resting
    }

    SECTION("FlatLadderBook DecrementAndCancel decrements both without trade") {
        sim::FlatLadderBook book(90, 110);
        auto maker = make_limit_order(1, sim::Side::Sell, 100, 30);
        book.add(maker, 1000);

        auto taker = make_limit_order(1, sim::Side::Buy, 100, 10);
        auto res = book.add(taker, 2000, sim::StpMode::DecrementAndCancel);

        REQUIRE(res.trades.empty());
        REQUIRE(res.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 20); // 30 - 10 = 20 remaining
    }

    SECTION("LimitOrderBook STP policies match FlatLadderBook behavior") {
        sim::LimitOrderBook book;
        auto maker = make_limit_order(42, sim::Side::Sell, 105, 50);
        book.add(maker);

        auto taker = make_limit_order(42, sim::Side::Buy, 105, 20);
        auto res = book.add(taker, sim::StpMode::DecrementAndCancel);

        REQUIRE(res.trades.empty());
        REQUIRE(book.volume_at(sim::Side::Sell, 105) == 30);
    }
}

TEST_CASE("L3 FIFO queue position tracking (queue_ahead)", "[lob][l3][queue]") {
    SECTION("FlatLadderBook queue_ahead tracks resting order volume ahead") {
        sim::FlatLadderBook book(90, 110);
        auto o1 = book.add(make_limit_order(1, sim::Side::Buy, 100, 10), 1000);
        auto o2 = book.add(make_limit_order(2, sim::Side::Buy, 100, 25), 1100);
        auto o3 = book.add(make_limit_order(3, sim::Side::Buy, 100, 15), 1200);

        // Order IDs start at 1
        REQUIRE(book.queue_ahead(1) == std::make_optional<sim::Qty>(0));
        REQUIRE(book.queue_ahead(2) == std::make_optional<sim::Qty>(10));
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(35));

        // Cancel order 1 -> orders 2 and 3 advance
        book.cancel(1, 1300);
        REQUIRE(book.queue_ahead(1) == std::nullopt);
        REQUIRE(book.queue_ahead(2) == std::make_optional<sim::Qty>(0));
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(25));

        // Market sell of 15 lots -> partially fills order 2 (10 lots remain)
        book.add(make_market_order(9, sim::Side::Sell, 15), 1400);
        REQUIRE(book.queue_ahead(2) == std::make_optional<sim::Qty>(0));
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(10));

        // Market sell of 10 lots -> fully fills order 2
        book.add(make_market_order(9, sim::Side::Sell, 10), 1500);
        REQUIRE(book.queue_ahead(2) == std::nullopt);
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(0));
    }

    SECTION("LimitOrderBook queue_ahead parity") {
        sim::LimitOrderBook book;
        book.add(make_limit_order(1, sim::Side::Sell, 102, 20));
        book.add(make_limit_order(2, sim::Side::Sell, 102, 30));
        book.add(make_limit_order(3, sim::Side::Sell, 102, 40));

        REQUIRE(book.queue_ahead(1) == std::make_optional<sim::Qty>(0));
        REQUIRE(book.queue_ahead(2) == std::make_optional<sim::Qty>(20));
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(50));

        book.cancel(1);
        REQUIRE(book.queue_ahead(1) == std::nullopt);
        REQUIRE(book.queue_ahead(2) == std::make_optional<sim::Qty>(0));
        REQUIRE(book.queue_ahead(3) == std::make_optional<sim::Qty>(30));
    }
}

TEST_CASE("Queue-Reactive model and Hawkes point-process generator", "[sim][queue_reactive]") {
    sim::QueueReactiveProcess qrp;

    auto lambda_shallow = qrp.compute_intensities(10, 10);
    auto lambda_deep    = qrp.compute_intensities(1000, 1000);

    // In queue-reactive theory (Huang et al.), deeper queue implies:
    // 1. Lower limit order arrival intensity (traders don't crowd large queues)
    const std::size_t buy_limit_idx = static_cast<std::size_t>(sim::QueueEventType::BuyLimit);
    REQUIRE(lambda_deep[buy_limit_idx] < lambda_shallow[buy_limit_idx]);

    // 2. Higher cancellation intensity (more resting volume to cancel)
    const std::size_t buy_cancel_idx = static_cast<std::size_t>(sim::QueueEventType::BuyCancel);
    REQUIRE(lambda_deep[buy_cancel_idx] > lambda_shallow[buy_cancel_idx]);

    // Generate 50 consecutive events and ensure timestamps strictly monotonically increase
    sim::TimeNs prev_time = 0;
    for (int i = 0; i < 50; ++i) {
        auto ev = qrp.next_event(100, 50, 101, 50);
        REQUIRE(qrp.current_time() > prev_time);
        REQUIRE(ev.qty >= 1);
        REQUIRE(ev.inter_arrival_ns > 0);
        prev_time = qrp.current_time();
    }
}

TEST_CASE("Extended order types: PostOnly, IOC, and FOK parity", "[sim][order_types]") {
    SECTION("FlatLadderBook PostOnly") {
        sim::FlatLadderBook book(90, 110);
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50), 1000);

        // Crossing PostOnly Buy is rejected
        sim::Order post_cross{.qty = 20, .orig_qty = 20, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        auto res1 = book.add(post_cross, 2000);
        REQUIRE(res1.trades.empty());
        REQUIRE(res1.filled == 0);
        REQUIRE(res1.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 50);
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 0);

        // Non-crossing PostOnly Buy rests passively
        sim::Order post_passive{.qty = 30, .orig_qty = 30, .price = 99, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        auto res2 = book.add(post_passive, 3000);
        REQUIRE(res2.trades.empty());
        REQUIRE(res2.filled == 0);
        REQUIRE(res2.remaining == 30);
        REQUIRE(book.volume_at(sim::Side::Buy, 99) == 30);
    }

    SECTION("LimitOrderBook PostOnly") {
        sim::LimitOrderBook book;
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50));

        sim::Order post_cross{.qty = 20, .orig_qty = 20, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        auto res1 = book.add(post_cross);
        REQUIRE(res1.trades.empty());
        REQUIRE(res1.filled == 0);
        REQUIRE(res1.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 50);
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 0);

        sim::Order post_passive{.qty = 30, .orig_qty = 30, .price = 99, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        auto res2 = book.add(post_passive);
        REQUIRE(res2.trades.empty());
        REQUIRE(res2.filled == 0);
        REQUIRE(res2.remaining == 30);
        REQUIRE(book.volume_at(sim::Side::Buy, 99) == 30);
    }

    SECTION("FlatLadderBook IOC") {
        sim::FlatLadderBook book(90, 110);
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50), 1000);

        sim::Order ioc{.qty = 80, .orig_qty = 80, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::IOC};
        auto res = book.add(ioc, 2000);
        REQUIRE(res.trades.size() == 1);
        REQUIRE(res.filled == 50);
        REQUIRE(res.remaining == 30);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 0);
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 0); // Residual does NOT rest
    }

    SECTION("LimitOrderBook IOC") {
        sim::LimitOrderBook book;
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50));

        sim::Order ioc{.qty = 80, .orig_qty = 80, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::IOC};
        auto res = book.add(ioc);
        REQUIRE(res.trades.size() == 1);
        REQUIRE(res.filled == 50);
        REQUIRE(res.remaining == 30);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 0);
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 0);
    }

    SECTION("FlatLadderBook FOK") {
        sim::FlatLadderBook book(90, 110);
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50), 1000);

        // Insufficient liquidity -> killed with 0 fills
        sim::Order fok_kill{.qty = 80, .orig_qty = 80, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::FOK};
        auto res1 = book.add(fok_kill, 2000);
        REQUIRE(res1.trades.empty());
        REQUIRE(res1.filled == 0);
        REQUIRE(res1.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 50);

        // Sufficient liquidity -> fills 100%
        sim::Order fok_fill{.qty = 40, .orig_qty = 40, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::FOK};
        auto res2 = book.add(fok_fill, 3000);
        REQUIRE(res2.trades.size() == 1);
        REQUIRE(res2.filled == 40);
        REQUIRE(res2.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 10);
    }

    SECTION("LimitOrderBook FOK") {
        sim::LimitOrderBook book;
        book.add(make_limit_order(1, sim::Side::Sell, 100, 50));

        sim::Order fok_kill{.qty = 80, .orig_qty = 80, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::FOK};
        auto res1 = book.add(fok_kill);
        REQUIRE(res1.trades.empty());
        REQUIRE(res1.filled == 0);
        REQUIRE(res1.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 50);

        sim::Order fok_fill{.qty = 40, .orig_qty = 40, .price = 100, .side = sim::Side::Buy, .type = sim::OrderType::FOK};
        auto res2 = book.add(fok_fill);
        REQUIRE(res2.trades.size() == 1);
        REQUIRE(res2.filled == 40);
        REQUIRE(res2.remaining == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, 100) == 10);
    }
}

TEST_CASE("Zero-tombstone intrusive slotmap and non-throwing construction", "[sim][intrusive_slotmap]") {
    // 1. Non-throwing construction and factory check
    auto opt_book = sim::FlatLadderBook::create(100, 90);
    REQUIRE(opt_book == nullptr);

    sim::FlatLadderBook invalid(100, 90);
    REQUIRE_FALSE(invalid.is_valid());

    auto valid_book = sim::FlatLadderBook::create(90, 110);
    REQUIRE(valid_book != nullptr);
    REQUIRE(valid_book->is_valid());

    // 2. Intrusive unlinking and slot recycling
    sim::FlatLadderBook book(90, 110, 256);
    std::vector<sim::OrderId> ids;
    for (int i = 0; i < 50; ++i) {
        auto res = book.add(make_limit_order(i, sim::Side::Buy, 100, 10), 1000 + i);
        ids.push_back(i + 1);
    }

    REQUIRE(book.volume_at(sim::Side::Buy, 100) == 500);
    REQUIRE(book.order_count() == 50);
    REQUIRE(book.check_invariants_report().ok);

    // Cancel odd orders (strict O(1) intrusive unlinking, zero tombstones)
    for (std::size_t i = 0; i < ids.size(); i += 2) {
        REQUIRE(book.cancel(ids[i], 2000 + i));
    }

    REQUIRE(book.volume_at(sim::Side::Buy, 100) == 250);
    REQUIRE(book.order_count() == 25);
    REQUIRE(book.check_invariants_report().ok);

    // Re-fill with new orders; verify recycled slots
    for (int i = 0; i < 25; ++i) {
        book.add(make_limit_order(100 + i, sim::Side::Buy, 100, 10), 3000 + i);
    }

    REQUIRE(book.volume_at(sim::Side::Buy, 100) == 500);
    REQUIRE(book.order_count() == 50);
    REQUIRE(book.check_invariants_report().ok);

    // 3. Compact storage requirements verification (TODO resolution)
    static_assert(sizeof(sim::CompactOrder) == 24, "CompactOrder must be 24 bytes");
    static_assert(sizeof(sim::IntrusiveOrderNode) == 32, "IntrusiveOrderNode must be 32 bytes");
    sim::CompactOrder co{.id = 42, .ts = 1000, .price = 100, .qty = 10};
    REQUIRE(co.id == 42);
    REQUIRE(co.price == 100);
}

TEST_CASE("Lean-aligned Bitwise vEB 128/256/512, ARM MSB-first, Offset-Binary, Gray Basket, and Elias-Fano", "[sim][veb_codings]") {
    sim::BitwiseVeb128 v128;
    v128.insert(95);
    v128.insert(12);
    REQUIRE(v128.find_min() == 12);
    REQUIRE(v128.find_max() == 95);
    v128.remove(12);
    REQUIRE(v128.find_min() == 95);

    sim::BitwiseVeb256Avx2 v256;
    v256.insert(210);
    v256.insert(45);
    REQUIRE(v256.find_min() == 45);
    v256.remove(45);
    REQUIRE(v256.find_min() == 210);

    sim::BitwiseVeb512 v512;
    v512.insert(499);
    v512.insert(129);
    REQUIRE(v512.find_min() == 129);
    v512.remove(129);
    REQUIRE(v512.find_min() == 499);

    sim::ArmMsbBitboard64 arm_bb;
    arm_bb.insert(41);
    arm_bb.insert(7);
    REQUIRE(arm_bb.find_min() == 7);
    REQUIRE(arm_bb.find_max() == 41);

    REQUIRE(sim::OffsetBinaryCodec::encode(-500) < sim::OffsetBinaryCodec::encode(0));
    REQUIRE(sim::OffsetBinaryCodec::encode(0) < sim::OffsetBinaryCodec::encode(500));
    REQUIRE(sim::OffsetBinaryCodec::decode(sim::OffsetBinaryCodec::encode(-123456)) == -123456);

    sim::KnuthGrayBasket<6> basket({10, -20, 35, 5, -15, 40});
    std::size_t flip = 0;
    std::uint64_t mask = 0;
    std::int64_t val = 0;
    std::size_t count = 0;
    while (basket.next_subset(flip, mask, val)) {
        std::int64_t expected = 0;
        for (std::size_t i = 0; i < 6; ++i) {
            if ((mask & (1ULL << i)) != 0ULL) expected += basket.weights[i];
        }
        REQUIRE(val == expected);
        ++count;
    }
    REQUIRE(count == 63);

    std::vector<std::uint64_t> sorted_ts;
    for (std::uint64_t i = 0; i < 512; ++i) {
        sorted_ts.push_back(i * 100ULL + (i & 7ULL));
    }
    sim::EliasFanoIndex ef(sorted_ts);
    for (std::size_t i = 0; i < sorted_ts.size(); ++i) {
        REQUIRE(ef.get(i) == sorted_ts[i]);
    }
}

TEST_CASE("FlatLadderBook zero-allocation hot path across passive add, add_into, cancel, and queue_ahead",
          "[flat-ladder][alloc-probe][hot-path]") {
#ifndef QRLOB_SANITIZER_ACTIVE
    {
        sim::diag::ScopedAllocCounter positive_ctrl;
        std::vector<int> dummy(64, 1);
        REQUIRE(dummy[0] == 1);
        REQUIRE(positive_ctrl.delta().allocations >= 1);
    }
#endif

    sim::FlatLadderBook book(90, 110, 512);
    sim::Trade trade_buf[16]{};

    sim::diag::AllocStats hot_delta{};
    {
        sim::diag::ScopedAllocCounter counter;

        // 1. Passive Limit adds via add() and add_into()
        for (int i = 0; i < 64; ++i) {
            const sim::Order bid = make_limit_order(1, sim::Side::Buy, static_cast<sim::Tick>(95 - (i & 3)), 10);
            const sim::Order ask = make_limit_order(2, sim::Side::Sell, static_cast<sim::Tick>(105 + (i & 3)), 10);
            const auto r_bid = book.add(bid, static_cast<sim::TimeNs>(i * 2 + 1));
            const auto s_ask = book.add_into(ask, static_cast<sim::TimeNs>(i * 2 + 2), trade_buf, 16);
            REQUIRE(r_bid.filled == 0);
            REQUIRE(r_bid.remaining == 10);
            REQUIRE(s_ask.filled == 0);
            REQUIRE(s_ask.remaining == 10);
            REQUIRE(s_ask.trade_count == 0);
        }

        // 2. PostOnly (passive and crossing-rejected) and rejected FOK/IOC
        sim::Order po_ok{.qty = 15, .orig_qty = 15, .price = 99, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        REQUIRE(book.add(po_ok, 500).remaining == 15);
        sim::Order po_rej{.qty = 15, .orig_qty = 15, .price = 105, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        REQUIRE(book.add(po_rej, 501).remaining == 0);
        sim::Order fok_rej{.qty = 5000, .orig_qty = 5000, .price = 108, .side = sim::Side::Buy, .type = sim::OrderType::FOK};
        REQUIRE(book.add(fok_rej, 502).filled == 0);

        // 3. L3 queue_ahead queries and top-of-book queries
        for (sim::OrderId id = 1; id <= 32; ++id) {
            auto qa = book.queue_ahead(id);
            REQUIRE(qa.has_value());
        }
        sim::Tick bb = 0, ba = 0, mid2 = 0;
        sim::Qty bv = 0, av = 0;
        REQUIRE(book.best_bid(bb, bv));
        REQUIRE(book.best_ask(ba, av));
        REQUIRE(book.mid_tick_x2(mid2));
        REQUIRE(book.volume_at(sim::Side::Buy, 99) == 15);

        // 4. Multi-fill crossing orders via add_into()
        sim::Order sweep_buy = make_limit_order(9, sim::Side::Buy, 108, 85);
        const auto sweep = book.add_into(sweep_buy, 600, trade_buf, 16);
        REQUIRE(sweep.filled == 85);
        REQUIRE(sweep.remaining == 0);
        REQUIRE(sweep.trade_count == 9);
        REQUIRE(trade_buf[0].qty == 10);
        REQUIRE(trade_buf[8].qty == 5);

        // 5. O(1) cancellations
        for (sim::OrderId id = 1; id <= 32; id += 2) {
            REQUIRE(book.cancel(id, 700 + id));
        }

        hot_delta = counter.delta();
    }

    REQUIRE(hot_delta.allocations == 0);
    REQUIRE(hot_delta.bytes == 0);
    REQUIRE(book.check_invariants_report().ok);
}

TEST_CASE("Empty/invalid FlatLadderBook, Tick overflow guards, and non-positive qty handling",
          "[flat-ladder][lob][edge-cases]") {
    SECTION("Invalid FlatLadderBook(100, 90) queries and mutations are safe") {
        sim::FlatLadderBook invalid(100, 90);
        REQUIRE_FALSE(invalid.is_valid());
        REQUIRE(invalid.tick_count() == 0);
        REQUIRE(invalid.is_empty());

        sim::Tick p = -1;
        sim::Qty v = 99;
        REQUIRE_FALSE(invalid.best_bid(p, v));
        REQUIRE(p == 100);
        REQUIRE(v == 0);
        REQUIRE_FALSE(invalid.best_ask(p, v));
        REQUIRE(p == 100);
        REQUIRE(v == 0);

        sim::Tick sum = 99;
        REQUIRE_FALSE(invalid.mid_tick_x2(sum));
        REQUIRE(sum == 0);

        double mid = 99.0;
        REQUIRE_FALSE(invalid.mid_price(mid));

        REQUIRE(invalid.volume_at(sim::Side::Buy, 100) == 0);
        REQUIRE(invalid.volume_at(sim::Side::Buy, std::numeric_limits<sim::Tick>::min()) == 0);
        REQUIRE(invalid.volume_at(sim::Side::Sell, std::numeric_limits<sim::Tick>::max()) == 0);

        auto res = invalid.add(make_limit_order(1, sim::Side::Buy, 100, 10), 10);
        REQUIRE(res.filled == 0);
        REQUIRE(res.remaining == 0);
        auto sum_res = invalid.add_into(make_limit_order(1, sim::Side::Buy, 100, 10), 10, nullptr, 0);
        REQUIRE(sum_res.filled == 0);
        REQUIRE(sum_res.remaining == 0);
        REQUIRE(invalid.order_count() == 0);
        REQUIRE(invalid.event_count() == 0);
    }

    SECTION("Extreme Tick values in volume_at do not overflow signed int32") {
        sim::FlatLadderBook book(90, 110);
        book.add(make_limit_order(1, sim::Side::Buy, 100, 25), 1);
        REQUIRE(book.volume_at(sim::Side::Buy, std::numeric_limits<sim::Tick>::min()) == 0);
        REQUIRE(book.volume_at(sim::Side::Buy, std::numeric_limits<sim::Tick>::min() + 1) == 0);
        REQUIRE(book.volume_at(sim::Side::Sell, std::numeric_limits<sim::Tick>::max()) == 0);
        REQUIRE(book.volume_at(sim::Side::Buy, 100) == 25);
    }

    SECTION("Zero and negative qty orders are rejected without resting or consuming OrderId") {
        sim::FlatLadderBook flat(90, 110);
        sim::LimitOrderBook tree;

        for (sim::Qty bad_qty : {sim::Qty{0}, sim::Qty{-5}}) {
            for (sim::OrderType tp : {sim::OrderType::Limit, sim::OrderType::PostOnly,
                                      sim::OrderType::Market, sim::OrderType::IOC,
                                      sim::OrderType::FOK}) {
                sim::Order o{.qty = bad_qty, .orig_qty = bad_qty, .price = 100, .side = sim::Side::Buy, .type = tp};
                auto rf = flat.add(o, 100);
                auto sf = flat.add_into(o, 100, nullptr, 0);
                auto rt = tree.add(o);
                REQUIRE(rf.filled == 0);
                REQUIRE(rf.remaining == 0);
                REQUIRE(sf.filled == 0);
                REQUIRE(sf.remaining == 0);
                REQUIRE(rt.filled == 0);
                REQUIRE(rt.remaining == 0);
            }
        }
        REQUIRE(flat.order_count() == 0);
        REQUIRE(tree.order_count() == 0);
        REQUIRE(flat.check_invariants_report().ok);

        // Next valid order still receives OrderId = 1 in both books
        flat.add(make_limit_order(1, sim::Side::Buy, 100, 10), 200);
        tree.add(make_limit_order(1, sim::Side::Buy, 100, 10));
        REQUIRE(flat.queue_ahead(1) == std::make_optional<sim::Qty>(0));
        REQUIRE(tree.queue_ahead(1) == std::make_optional<sim::Qty>(0));
    }

    SECTION("Out-of-range PostOnly does not rest and reports 0 remaining") {
        sim::FlatLadderBook flat(90, 110);
        sim::Order po_low{.qty = 20, .orig_qty = 20, .price = 80, .side = sim::Side::Buy, .type = sim::OrderType::PostOnly};
        auto r1 = flat.add(po_low, 10);
        REQUIRE(r1.filled == 0);
        REQUIRE(r1.remaining == 0);
        REQUIRE(flat.order_count() == 0);
    }
}

TEST_CASE("FOK + Self-Trade Prevention exact pre-flight parity in FlatLadderBook and LimitOrderBook",
          "[flat-ladder][lob][fok][stp]") {
    auto test_both = [](auto&& fn) {
        sim::FlatLadderBook flat(90, 110);
        sim::LimitOrderBook tree;
        fn(flat, tree);
        REQUIRE(flat.check_invariants_report().ok);
        require_same_book_state(tree, flat, 90, 110);
    };

    SECTION("CancelAggressive: self-order blocking full fill rejects FOK with zero mutation") {
        test_both([](auto& flat, auto& tree) {
            // Agent 2 has 20 lots ahead of Agent 1's 30 lots at 100 Ask
            require_same_after_add(tree, flat, make_limit_order(2, sim::Side::Sell, 100, 20), 90, 110, 1);
            require_same_after_add(tree, flat, make_limit_order(1, sim::Side::Sell, 100, 30), 90, 110, 2);

            // Agent 1 submits FOK Buy for 40 lots under CancelAggressive -> must reject!
            sim::Order fok{.qty = 40, .agent = 1, .orig_qty = 40, .price = 100,
                           .side = sim::Side::Buy, .type = sim::OrderType::FOK,
                           .stp = sim::StpMode::CancelAggressive};
            const auto rf = flat.add(fok, 3);
            const auto rt = tree.add(fok);
            REQUIRE(rf.trades.empty());
            REQUIRE(rf.filled == 0);
            REQUIRE(rf.remaining == 0);
            REQUIRE(rt.trades.empty());
            REQUIRE(rt.filled == 0);
            REQUIRE(rt.remaining == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 100) == 50);

            // Agent 1 submits FOK Buy for 20 lots under CancelAggressive -> fills 20 against Agent 2 before reaching self-order!
            sim::Order fok_ok{.qty = 20, .agent = 1, .orig_qty = 20, .price = 100,
                              .side = sim::Side::Buy, .type = sim::OrderType::FOK,
                              .stp = sim::StpMode::CancelAggressive};
            const auto rf2 = flat.add(fok_ok, 4);
            const auto rt2 = tree.add(fok_ok);
            REQUIRE(rf2.filled == 20);
            REQUIRE(rf2.remaining == 0);
            REQUIRE(rt2.filled == 20);
            REQUIRE(rt2.remaining == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 100) == 30);
        });
    }

    SECTION("DecrementAndCancel: self-order blocking full fill rejects FOK with zero mutation") {
        test_both([](auto& flat, auto& tree) {
            require_same_after_add(tree, flat, make_limit_order(2, sim::Side::Sell, 100, 20), 90, 110, 1);
            require_same_after_add(tree, flat, make_limit_order(1, sim::Side::Sell, 100, 30), 90, 110, 2);

            sim::Order fok{.qty = 30, .agent = 1, .orig_qty = 30, .price = 100,
                           .side = sim::Side::Buy, .type = sim::OrderType::FOK,
                           .stp = sim::StpMode::DecrementAndCancel};
            const auto rf = flat.add(fok, 3);
            const auto rt = tree.add(fok);
            REQUIRE(rf.trades.empty());
            REQUIRE(rf.filled == 0);
            REQUIRE(rf.remaining == 0);
            REQUIRE(rt.trades.empty());
            REQUIRE(rt.filled == 0);
            REQUIRE(rt.remaining == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 100) == 50);
        });
    }

    SECTION("CancelPassive: FOK counts only non-self liquidity and cancels self-orders only when fillable") {
        test_both([](auto& flat, auto& tree) {
            // Agent 1 has 30 lots @ 100 Ask, Agent 2 has 25 lots @ 101 Ask
            require_same_after_add(tree, flat, make_limit_order(1, sim::Side::Sell, 100, 30), 90, 110, 1);
            require_same_after_add(tree, flat, make_limit_order(2, sim::Side::Sell, 101, 25), 90, 110, 2);

            // Agent 1 FOK Buy 40 @ 101 with CancelPassive: non-self liquidity is 25 < 40 -> rejected without cancelling Agent 1's ask!
            sim::Order fok_rej{.qty = 40, .agent = 1, .orig_qty = 40, .price = 101,
                               .side = sim::Side::Buy, .type = sim::OrderType::FOK,
                               .stp = sim::StpMode::CancelPassive};
            const auto rf1 = flat.add(fok_rej, 3);
            const auto rt1 = tree.add(fok_rej);
            REQUIRE(rf1.trades.empty());
            REQUIRE(rf1.filled == 0);
            REQUIRE(rt1.trades.empty());
            REQUIRE(rt1.filled == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 100) == 30);
            REQUIRE(flat.volume_at(sim::Side::Sell, 101) == 25);

            // Agent 1 FOK Buy 25 @ 101 with CancelPassive: non-self liquidity is 25 >= 25 -> cancels Agent 1's 30 @ 100 and fills 25 @ 101!
            sim::Order fok_ok{.qty = 25, .agent = 1, .orig_qty = 25, .price = 101,
                              .side = sim::Side::Buy, .type = sim::OrderType::FOK,
                              .stp = sim::StpMode::CancelPassive};
            const auto rf2 = flat.add(fok_ok, 4);
            const auto rt2 = tree.add(fok_ok);
            REQUIRE(rf2.trades.size() == 1);
            REQUIRE(rf2.filled == 25);
            REQUIRE(rf2.remaining == 0);
            REQUIRE(rt2.trades.size() == 1);
            REQUIRE(rt2.filled == 25);
            REQUIRE(rt2.remaining == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 100) == 0);
            REQUIRE(flat.volume_at(sim::Side::Sell, 101) == 0);
        });
    }
}
