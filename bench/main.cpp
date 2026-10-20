#include <benchmark/benchmark.h>

#include "sim/alloc_probe.hpp"
#include "sim/flat_ladder_book.hpp"
#include "sim/lob.hpp"
#include "sim/queue_reactive.hpp"
#include "sim/rng.hpp"

#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

// Global allocation-operator replacement for ScopedAllocCounter telemetry.
// Active only while a ScopedAllocCounter is armed on the calling thread.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
namespace {
inline void* aligned_alloc_impl(std::size_t size, std::size_t align) noexcept {
    if (align < sizeof(void*)) align = sizeof(void*);
    void* p = nullptr;
    if (posix_memalign(&p, align, size == 0 ? 1 : size) != 0) return nullptr;
    return p;
}
} // namespace

void* operator new(std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) {
#if defined(__cpp_exceptions)
        throw std::bad_alloc();
#else
        std::abort();
#endif
    }
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) {
#if defined(__cpp_exceptions)
        throw std::bad_alloc();
#else
        std::abort();
#endif
    }
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new(std::size_t size, std::align_val_t al) {
    void* p = aligned_alloc_impl(size, static_cast<std::size_t>(al));
    if (!p) {
#if defined(__cpp_exceptions)
        throw std::bad_alloc();
#else
        std::abort();
#endif
    }
    if (sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size, std::align_val_t al) {
    void* p = aligned_alloc_impl(size, static_cast<std::size_t>(al));
    if (!p) {
#if defined(__cpp_exceptions)
        throw std::bad_alloc();
#else
        std::abort();
#endif
    }
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
void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    void* p = aligned_alloc_impl(size, static_cast<std::size_t>(al));
    if (p && sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}
void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    void* p = aligned_alloc_impl(size, static_cast<std::size_t>(al));
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
void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept { std::free(ptr); }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

inline sim::Order make_limit_order(sim::AgentId agent,
                                   sim::Side side,
                                   sim::Tick price,
                                   sim::Qty qty) {
    sim::Order o{};
    // Keep assignment order aligned with struct field order in sim::Order.
    o.agent    = agent;
    o.qty      = qty;
    o.orig_qty = qty;
    o.price    = price;
    o.side     = side;
    o.type     = sim::OrderType::Limit;
    return o;
}

// Adapter helpers -- the FlatLadderBook hot path requires `ts`; LOB does not.
// Defaulting ts=0 here is safe in BENCH-ONLY code: production callers still
// must pass an explicit timestamp.
template <typename Book>
inline auto add_at(Book& b, const sim::Order& o, sim::TimeNs ts = 0) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        return b.add(o, ts);
    } else {
        (void)ts;
        return b.add(o);
    }
}

template <typename Book>
inline bool cancel_at(Book& b, sim::OrderId id, sim::TimeNs ts = 0) {
    if constexpr (std::is_same_v<Book, sim::FlatLadderBook>) {
        return b.cancel(id, ts);
    } else {
        (void)ts;
        return b.cancel(id);
    }
}

template <typename MakeBook>
void run_add_no_cross(benchmark::State& state,
                      MakeBook&& make_book,
                      sim::Tick mid,
                      sim::Qty qty) {
    const int batch = static_cast<int>(state.range(0));
    std::uint64_t sink = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto book = make_book(batch);
        state.ResumeTiming();

        for (int i = 0; i < batch; ++i) {
            const sim::Order buy = make_limit_order(
                1,
                sim::Side::Buy,
                static_cast<sim::Tick>(mid - 10 - (i & 7)),
                qty);
            const sim::Order sell = make_limit_order(
                2,
                sim::Side::Sell,
                static_cast<sim::Tick>(mid + 10 + (i & 7)),
                qty);
            const sim::TimeNs ts = static_cast<sim::TimeNs>(i);
            sink += static_cast<std::uint64_t>(add_at(book, buy, ts * 2).filled);
            sink += static_cast<std::uint64_t>(add_at(book, sell, ts * 2 + 1).filled);
        }

        benchmark::DoNotOptimize(sink);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch) * 2);
}

template <typename MakeBook>
void run_cancel_cycle(benchmark::State& state,
                      MakeBook&& make_book,
                      sim::Tick price,
                      sim::Qty qty) {
    const int n = static_cast<int>(state.range(0));
    std::uint64_t cancelled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto book = make_book(n);
        std::vector<sim::OrderId> ids;
        ids.reserve(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            const sim::Order o = make_limit_order(7, sim::Side::Buy, price, qty);
            add_at(book, o, static_cast<sim::TimeNs>(i));
            ids.push_back(static_cast<sim::OrderId>(i + 1));
        }
        state.ResumeTiming();

        sim::TimeNs cts = static_cast<sim::TimeNs>(n);
        for (sim::OrderId id : ids) {
            cancelled += static_cast<std::uint64_t>(cancel_at(book, id, ++cts));
        }

        benchmark::DoNotOptimize(cancelled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}


// Arena-sized cancel cycle helper: parameterizes arena computation and counter reporting.
template <typename ComputeArena>
void run_cancel_cycle_arena(benchmark::State& state,
                             ComputeArena&& compute_arena,
                             sim::Tick price,
                             sim::Qty qty) {
    const int n = static_cast<int>(state.range(0));
    std::uint64_t cancelled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        const auto [arena_bytes, counter_value] = compute_arena(n);
        sim::LimitOrderBook book(
            static_cast<std::size_t>(n + 64),
            /*enable_compaction=*/true,
            {},
            arena_bytes);
        std::vector<sim::OrderId> ids;
        ids.reserve(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            const sim::Order o = make_limit_order(7, sim::Side::Buy, price, qty);
            add_at(book, o);
            ids.push_back(static_cast<sim::OrderId>(i + 1));
        }
        state.ResumeTiming();

        for (sim::OrderId id : ids) {
            cancelled += static_cast<std::uint64_t>(cancel_at(book, id));
        }

        benchmark::DoNotOptimize(cancelled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
    // Compute arena config once to report counter (use first iteration's result)
    const auto [final_arena_bytes, final_counter_value] = compute_arena(n);
    state.counters["arena_kib"] = final_counter_value;
}

template <typename MakeBook>
void run_crossing_match(benchmark::State& state,
                        MakeBook&& make_book,
                        sim::Tick ask_price,
                        sim::Qty qty) {
    const int depth = static_cast<int>(state.range(0));
    std::uint64_t filled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto book = make_book(depth);

        for (int i = 0; i < depth; ++i) {
            const sim::Order ask = make_limit_order(11, sim::Side::Sell, ask_price, qty);
            add_at(book, ask, static_cast<sim::TimeNs>(i));
        }
        state.ResumeTiming();

        const sim::Order buy_cross = make_limit_order(
            99,
            sim::Side::Buy,
            ask_price,
            static_cast<sim::Qty>(depth));
        const auto result = add_at(book, buy_cross, static_cast<sim::TimeNs>(depth));
        filled += static_cast<std::uint64_t>(result.filled);

        benchmark::DoNotOptimize(filled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(depth));
}

template <typename MakeBook>
void run_out_of_band_passive(benchmark::State& state,
                             MakeBook&& make_book,
                             sim::Tick mid,
                             sim::Qty qty) {
    const int n = static_cast<int>(state.range(0));
    std::uint64_t sink = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto book = make_book(n);
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            const sim::Order o = make_limit_order(
                21,
                sim::Side::Buy,
                static_cast<sim::Tick>(mid - 100 - (i & 3)),
                qty);
            sink += static_cast<std::uint64_t>(add_at(book, o, static_cast<sim::TimeNs>(i)).remaining);
        }

        sink += static_cast<std::uint64_t>(book.order_count());
        benchmark::DoNotOptimize(sink);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

template <typename MakeBook>
void run_compaction_stress(benchmark::State& state,
                           MakeBook&& make_book,
                           sim::Tick price,
                           sim::Qty qty) {
    const int n = static_cast<int>(state.range(0));
    const int rounds = static_cast<int>(state.range(1));
    std::uint64_t cancelled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        auto book = make_book(n);
        sim::OrderId next_id = 1;
        state.ResumeTiming();

        for (int r = 0; r < rounds; ++r) {
            for (int i = 0; i < n; ++i) {
                const sim::Order o = make_limit_order(42, sim::Side::Buy, price, qty);
                benchmark::DoNotOptimize(add_at(book, o));
            }

            for (int i = 0; i < n; ++i) {
                cancelled += static_cast<std::uint64_t>(cancel_at(book, next_id++));
            }
        }

        benchmark::DoNotOptimize(cancelled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n) * rounds * 2);
}

} // namespace

static void BM_RngNextU64(benchmark::State& state) {
    sim::Rng rng(42);
    std::uint64_t sink = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(sink ^= rng.next_u64());
    }

    benchmark::DoNotOptimize(sink);
}

BENCHMARK(BM_RngNextU64);

static void BM_LOB_AddNoCross(benchmark::State& state) {
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    run_add_no_cross(
        state,
        [](int batch) {
            return sim::LimitOrderBook(static_cast<std::size_t>(batch * 2 + 64));
        },
        kMid,
        kQty);
}

static void BM_FlatLadder_AddNoCross(benchmark::State& state) {
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    run_add_no_cross(
        state,
        [=](int batch) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(batch * 2 + 64));
        },
        kMid,
        kQty);
}

static void BM_LOB_CancelCycle(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_cancel_cycle(
        state,
        [](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64));
        },
        kPrice,
        kQty);
}

static void BM_FlatLadder_CancelCycle(benchmark::State& state) {
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_cancel_cycle(
        state,
        [=](int n) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(n + 64));
        },
        kPrice,
        kQty);
}

static void BM_LOB_CancelCycle_NoCompaction(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_cancel_cycle(
        state,
        [](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64), /*enable_compaction=*/false);
        },
        kPrice,
        kQty);
}

static void BM_LOB_CrossingMatch(benchmark::State& state) {
    constexpr sim::Tick kAsk = 10'001;
    constexpr sim::Qty kQty = 1;
    run_crossing_match(
        state,
        [](int depth) {
            return sim::LimitOrderBook(static_cast<std::size_t>(depth + 64));
        },
        kAsk,
        kQty);
}

static void BM_FlatLadder_CrossingMatch(benchmark::State& state) {
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kAsk = 10'001;
    constexpr sim::Qty kQty = 1;
    run_crossing_match(
        state,
        [=](int depth) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(depth + 64));
        },
        kAsk,
        kQty);
}

// Compare out-of-band passive limit handling:
// tree book rests those orders at off-band prices, flat ladder drops remainder.
static void BM_LOB_OutOfBandPassive(benchmark::State& state) {
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    run_out_of_band_passive(
        state,
        [](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64));
        },
        kMid,
        kQty);
}

static void BM_FlatLadder_OutOfBandPassive(benchmark::State& state) {
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    run_out_of_band_passive(
        state,
        [=](int n) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(n + 64));
        },
        kMid,
        kQty);
}

// Repeatedly build/cancel dense same-price queues to exercise tombstone
// accumulation and compaction policy over long horizons.
static void BM_LOB_CompactionStress(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_compaction_stress(
        state,
        [](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64));
        },
        kPrice,
        kQty);
}

static void BM_LOB_CompactionStress_NoCompaction(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_compaction_stress(
        state,
        [](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64), /*enable_compaction=*/false);
        },
        kPrice,
        kQty);
}

static void BM_LOB_CompactionStress_MemoryProfile(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    // More eager compaction (memory-first): compact sooner and at lower dead ratio.
    const sim::CompactionConfig memory_profile{
        .dead_min = 256,
        .total_min = 512,
        .dead_num = 1,
        .dead_den = 2,
    };

    run_compaction_stress(
        state,
        [=](int n) {
            return sim::LimitOrderBook(static_cast<std::size_t>(n + 64), true, memory_profile);
        },
        kPrice,
        kQty);
}

static void BM_LOB_CompactionStress_ArenaAuto(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    const int n = static_cast<int>(state.range(0));
    const int rounds = static_cast<int>(state.range(1));
    const std::size_t expected = static_cast<std::size_t>(n) + std::size_t{64};
    const std::size_t arena_bytes = sim::suggest_arena_bytes(expected);

    std::uint64_t cancelled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        sim::LimitOrderBook book(
            static_cast<std::size_t>(n + 64),
            true,
            {},
            arena_bytes);
        sim::OrderId next_id = 1;
        state.ResumeTiming();

        for (int r = 0; r < rounds; ++r) {
            for (int i = 0; i < n; ++i) {
                const sim::Order o = make_limit_order(42, sim::Side::Buy, kPrice, kQty);
                benchmark::DoNotOptimize(book.add(o));
            }

            for (int i = 0; i < n; ++i) {
                cancelled += static_cast<std::uint64_t>(book.cancel(next_id++));
            }
        }

        benchmark::DoNotOptimize(cancelled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n) * rounds * 2);
    state.counters["arena_kib"] = static_cast<double>(arena_bytes) / 1024.0;
}

// Arena-size sweep for allocator locality tuning.
// Args: {orders_per_cycle, arena_kib}
static void BM_LOB_CancelCycle_ArenaSweep(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_cancel_cycle_arena(
        state,
        [&state](int) {
            const std::size_t arena_bytes = static_cast<std::size_t>(state.range(1)) * 1024;
            return std::make_pair(arena_bytes, static_cast<double>(state.range(1)));
        },
        kPrice,
        kQty);
}

// Auto-sized arena based on expected order count helper.
static void BM_LOB_CancelCycle_ArenaAuto(benchmark::State& state) {
    constexpr sim::Tick kPrice = 10'000;
    constexpr sim::Qty kQty = 1;
    run_cancel_cycle_arena(
        state,
        [](int n) {
            const std::size_t expected = static_cast<std::size_t>(n) + std::size_t{64};
            const std::size_t arena_bytes = sim::suggest_arena_bytes(expected);
            return std::make_pair(arena_bytes, static_cast<double>(arena_bytes) / 1024.0);
        },
        kPrice,
        kQty);
}

// ---------------------------------------------------------------------------
// Deep-ladder benchmarks
//
// Wide tick band (~100k ticks, ~1.5k bitmap words). Stresses the
// bitmap-accelerated refresh_best_*: a linear scan over a sparse 100k-entry
// vector would dominate cancel-on-inside cost; the bitmap collapses that to
// O(N/64) word scans + a single intrinsic.
// ---------------------------------------------------------------------------

// Deep cancel-cycle: rest at the inside of a wide band, then drain. Each
// cancel that empties the inside-best level forces a refresh -- the path
// the bitmap most accelerates.
static void BM_FlatLadder_DeepCancelCycle(benchmark::State& state) {
    constexpr sim::Tick kMin   = 0;
    constexpr sim::Tick kMax   = 100'000;
    constexpr sim::Tick kPrice = 50'000;
    constexpr sim::Qty  kQty   = 1;
    run_cancel_cycle(
        state,
        [=](int n) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(n + 64));
        },
        kPrice,
        kQty);
}

// Deep refresh-heavy: rest at sparse extreme prices, then cancel each. Every
// cancel collapses the cached best and triggers a long refresh window across
// the bitmap. The constant-factor win vs. linear scan grows with band width.
static void BM_FlatLadder_DeepRefreshHeavy(benchmark::State& state) {
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100'000;
    constexpr sim::Qty  kQty = 1;
    const int n = static_cast<int>(state.range(0));
    std::uint64_t cancelled = 0;

    for (auto _ : state) {
        state.PauseTiming();
        sim::FlatLadderBook book(kMin, kMax, static_cast<std::size_t>(n + 64));
        std::vector<sim::OrderId> ids;
        ids.reserve(static_cast<std::size_t>(n));
        // Spread orders across the full ladder so the inside is always
        // "deep" relative to the next non-empty level.
        const sim::Tick step = static_cast<sim::Tick>((kMax - kMin) / std::max(1, n));
        for (int i = 0; i < n; ++i) {
            const sim::Tick px = static_cast<sim::Tick>(kMin + 1 + i * step);
            const sim::Order o = make_limit_order(7, sim::Side::Buy, px, kQty);
            book.add(o, static_cast<sim::TimeNs>(i));
            ids.push_back(static_cast<sim::OrderId>(i + 1));
        }
        state.ResumeTiming();

        // Cancel from the inside outward -- each cancel triggers a full
        // refresh sweep. Without the bitmap this is O(N) per cancel.
        // Reverse traversal is load-bearing (must always cancel the current
        // inside), so the modernize-to-range-for hint does not apply here.
        sim::TimeNs cts = static_cast<sim::TimeNs>(n);
        // NOLINTNEXTLINE(modernize-loop-convert)
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
            cancelled += static_cast<std::uint64_t>(book.cancel(*it, ++cts));
        }

        benchmark::DoNotOptimize(cancelled);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// Deep market vs limit: confirms the IsLimit template specializations carry no
// regression on a wide-band workload (and exercises both kBuy / kSell crosses).
static void BM_FlatLadder_DeepCrossingMatch(benchmark::State& state) {
    constexpr sim::Tick kMin = 0;
    constexpr sim::Tick kMax = 100'000;
    constexpr sim::Tick kAsk = 50'000;
    constexpr sim::Qty  kQty = 1;
    run_crossing_match(
        state,
        [=](int depth) {
            return sim::FlatLadderBook(kMin, kMax, static_cast<std::size_t>(depth + 64));
        },
        kAsk,
        kQty);
}

// Allocation-probe benchmarks: measure steady-state heap allocations across
// passive and crossing hot paths after book construction.
static void BM_LOB_AddNoCross_AllocProbe(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    std::uint64_t total_allocs = 0;
    std::uint64_t total_bytes = 0;

    for (auto _ : state) {
        state.PauseTiming();
        sim::LimitOrderBook book(static_cast<std::size_t>(batch * 2 + 64));
        state.ResumeTiming();
        std::uint64_t sink = 0;
        sim::diag::AllocStats delta{};
        {
            sim::diag::ScopedAllocCounter counter;
            for (int i = 0; i < batch; ++i) {
                const sim::Order buy = make_limit_order(
                    1, sim::Side::Buy, static_cast<sim::Tick>(kMid - 10 - (i & 7)), kQty);
                const sim::Order sell = make_limit_order(
                    2, sim::Side::Sell, static_cast<sim::Tick>(kMid + 10 + (i & 7)), kQty);
                sink += static_cast<std::uint64_t>(book.add(buy).filled);
                sink += static_cast<std::uint64_t>(book.add(sell).filled);
            }
            delta = counter.delta();
        }
        benchmark::DoNotOptimize(sink);
        total_allocs += delta.allocations;
        total_bytes += delta.bytes;
    }

    state.counters["allocs"] =
        benchmark::Counter(static_cast<double>(total_allocs), benchmark::Counter::kAvgIterations);
    state.counters["alloc_bytes"] =
        benchmark::Counter(static_cast<double>(total_bytes), benchmark::Counter::kAvgIterations);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch) * 2);
}

static void BM_FlatLadder_AddNoCross_AllocProbe(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kMid = 10'000;
    constexpr sim::Qty kQty = 1;
    std::uint64_t total_allocs = 0;
    std::uint64_t total_bytes = 0;

    for (auto _ : state) {
        state.PauseTiming();
        sim::FlatLadderBook book(kMin, kMax, static_cast<std::size_t>(batch * 2 + 64));
        state.ResumeTiming();
        std::uint64_t sink = 0;
        sim::diag::AllocStats delta{};
        {
            sim::diag::ScopedAllocCounter counter;
            for (int i = 0; i < batch; ++i) {
                const sim::Order buy = make_limit_order(
                    1, sim::Side::Buy, static_cast<sim::Tick>(kMid - 10 - (i & 7)), kQty);
                const sim::Order sell = make_limit_order(
                    2, sim::Side::Sell, static_cast<sim::Tick>(kMid + 10 + (i & 7)), kQty);
                const sim::TimeNs ts = static_cast<sim::TimeNs>(i);
                sink += static_cast<std::uint64_t>(book.add(buy, ts * 2).filled);
                sink += static_cast<std::uint64_t>(book.add(sell, ts * 2 + 1).filled);
            }
            delta = counter.delta();
        }
        benchmark::DoNotOptimize(sink);
        total_allocs += delta.allocations;
        total_bytes += delta.bytes;
    }

    state.counters["allocs"] =
        benchmark::Counter(static_cast<double>(total_allocs), benchmark::Counter::kAvgIterations);
    state.counters["alloc_bytes"] =
        benchmark::Counter(static_cast<double>(total_bytes), benchmark::Counter::kAvgIterations);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch) * 2);
}

static void BM_FlatLadder_AddIntoCrossing_AllocProbe(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    constexpr sim::Tick kMin = 9'980;
    constexpr sim::Tick kMax = 10'020;
    constexpr sim::Tick kAsk = 10'000;
    constexpr sim::Qty kQty = 1;
    std::uint64_t total_allocs = 0;
    std::uint64_t total_bytes = 0;

    for (auto _ : state) {
        state.PauseTiming();
        sim::FlatLadderBook book(kMin, kMax, static_cast<std::size_t>(batch * 2 + 64));
        for (int i = 0; i < batch; ++i) {
            const sim::Order ask = make_limit_order(
                1, sim::Side::Sell, static_cast<sim::Tick>(kAsk + (i & 7)), kQty);
            book.add_into(ask, static_cast<sim::TimeNs>(i), nullptr, 0);
        }
        state.ResumeTiming();

        sim::Trade trade_buf[16]{};
        std::uint64_t sink = 0;
        sim::diag::AllocStats delta{};
        {
            sim::diag::ScopedAllocCounter counter;
            for (int i = 0; i < batch; ++i) {
                const sim::Order buy = make_limit_order(
                    2, sim::Side::Buy, static_cast<sim::Tick>(kAsk + 7), kQty);
                const auto summary = book.add_into(
                    buy, static_cast<sim::TimeNs>(batch + i), trade_buf, 16);
                sink += static_cast<std::uint64_t>(summary.filled) + summary.trade_count;
            }
            delta = counter.delta();
        }
        benchmark::DoNotOptimize(sink);
        benchmark::DoNotOptimize(trade_buf);
        total_allocs += delta.allocations;
        total_bytes += delta.bytes;
    }

    state.counters["allocs"] =
        benchmark::Counter(static_cast<double>(total_allocs), benchmark::Counter::kAvgIterations);
    state.counters["alloc_bytes"] =
        benchmark::Counter(static_cast<double>(total_bytes), benchmark::Counter::kAvgIterations);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
}

BENCHMARK(BM_LOB_AddNoCross)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_LOB_AddNoCross_AllocProbe)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_FlatLadder_AddNoCross)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_FlatLadder_AddNoCross_AllocProbe)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_FlatLadder_AddIntoCrossing_AllocProbe)->Arg(256)->Arg(1024)->Arg(4096);
BENCHMARK(BM_LOB_CancelCycle)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_FlatLadder_CancelCycle)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_LOB_CancelCycle_NoCompaction)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_LOB_CrossingMatch)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_FlatLadder_CrossingMatch)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_LOB_OutOfBandPassive)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_FlatLadder_OutOfBandPassive)->Arg(1024)->Arg(4096)->Arg(16384);
// Deep-ladder benchmarks across a 100,001-tick band.
BENCHMARK(BM_FlatLadder_DeepCancelCycle)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_FlatLadder_DeepRefreshHeavy)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_FlatLadder_DeepCrossingMatch)->Arg(1024)->Arg(4096)->Arg(16384);
BENCHMARK(BM_LOB_CompactionStress)
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({4096, 16});
BENCHMARK(BM_LOB_CompactionStress_NoCompaction)
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({4096, 16});
BENCHMARK(BM_LOB_CompactionStress_MemoryProfile)
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({4096, 16});
BENCHMARK(BM_LOB_CompactionStress_ArenaAuto)
    ->Args({1024, 8})
    ->Args({4096, 8})
    ->Args({4096, 16});
BENCHMARK(BM_LOB_CancelCycle_ArenaSweep)
    ->Args({4096, 256})
    ->Args({4096, 1024})
    ->Args({4096, 4096});
BENCHMARK(BM_LOB_CancelCycle_ArenaAuto)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384);

void BM_FlatLadder_STP_CancelPassive(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    for (auto _ : state) {
        state.PauseTiming();
        sim::FlatLadderBook book(90, 110, batch * 2);
        for (int i = 0; i < batch; ++i) {
            book.add(make_limit_order(1, sim::Side::Sell, 100, 10), i);
        }
        state.ResumeTiming();
        for (int i = 0; i < batch; ++i) {
            book.add(make_limit_order(1, sim::Side::Buy, 100, 10), batch + i, sim::StpMode::CancelPassive);
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * batch);
}

void BM_FlatLadder_QueueAhead(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    sim::FlatLadderBook book(90, 110, batch * 2);
    for (int i = 0; i < batch; ++i) {
        book.add(make_limit_order(static_cast<sim::AgentId>(i + 1), sim::Side::Buy, 100, 10), i);
    }
    sim::Qty sink = 0;
    for (auto _ : state) {
        for (int i = 1; i <= batch; ++i) {
            auto q = book.queue_ahead(static_cast<sim::OrderId>(i));
            if (q) sink += *q;
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * batch);
}

void BM_QueueReactive_EventGeneration(benchmark::State& state) {
    const int batch = static_cast<int>(state.range(0));
    sim::QueueReactiveProcess qrp;
    for (auto _ : state) {
        for (int i = 0; i < batch; ++i) {
            auto ev = qrp.next_event(100, 50, 101, 50);
            benchmark::DoNotOptimize(ev);
        }
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * batch);
}

BENCHMARK(BM_FlatLadder_STP_CancelPassive)->Arg(1024)->Arg(4096);
BENCHMARK(BM_FlatLadder_QueueAhead)->Arg(1024)->Arg(4096);
BENCHMARK(BM_QueueReactive_EventGeneration)->Arg(1024)->Arg(4096);

BENCHMARK_MAIN();
