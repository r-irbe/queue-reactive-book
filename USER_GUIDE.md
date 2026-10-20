# Queue-Reactive Limit Order Book -- User Guide

This guide covers building the simulator and test binaries with CMake or Make, running the standalone CLI and benchmark suites, and understanding the architecture of the `O(1)` intrusive limit order book (`FlatLadderBook`) and the Queue-Reactive + Hawkes arrival engine (`QueueReactiveProcess`).

---

## 1. Building the Project

The project requires a standard C++20 compiler (`g++ >= 11` or `clang++ >= 14`) and either GNU Make or CMake (`>= 3.16`). Self-contained header shims under `include/test_shim/` provide the Catch2 and Google Benchmark interfaces when external packages are not installed.

### Building with CMake (Recommended for CI)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

To build without the experimental offline replay / SPSC / Hawkes calibration surface:

```bash
cmake -B build_core -DCMAKE_BUILD_TYPE=Release -DQRLOB_BUILD_EXPERIMENTAL=OFF
cmake --build build_core
ctest --test-dir build_core --output-on-failure
```

### Building with GNU Make

```bash
make              # Build standalone simulation CLI (build/sim)
make test         # Build and run full test suite (build/sim_test, 165 test cases)
make test_core    # Build and run core-only tests (QRLOB_BUILD_EXPERIMENTAL=OFF)
make bench        # Build and run microbenchmarks (build/sim_bench)
make example      # Build and run examples/basic_sim.cpp (build/sim_example)
```

Diagnostic sanitizer and LTO modes:

```bash
make asan         # AddressSanitizer + UndefinedBehaviorSanitizer
make ubsan        # UndefinedBehaviorSanitizer
make tsan         # ThreadSanitizer
make debug        # Unoptimized debug build (-O0 -g3)
make release-lto  # Build build/sim_bench with -O3 -flto=auto -march=native
```

---

## 2. Running the Simulator CLI

The `build/sim` binary demonstrates self-trade prevention (`CancelPassive`), L3 FIFO queue priority inspection (`queue_ahead`), and a zero-allocation Queue-Reactive simulation loop verified by `qr::alloc_guard::ScopedAllocCounter`.

```bash
./build/sim --help
./build/sim --steps 50000 --seed 42
```

### CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-h`, `--help` | Print CLI usage and exit | -- |
| `--steps <N>` | Number of simulation events to generate and match (minimum `100`) | `10000` |
| `--seed <S>` | 64-bit seed (decimal or `0x` hex) for `Xoshiro256StarStar` PRNG | `0x9E3779B97F4A7C15` |

Positional `<STEPS>` is also accepted for backwards compatibility (`./build/sim 25000`).

---

## 3. Running the Benchmark Suites

The benchmark binary `build/sim_bench` runs microbenchmarks across both `FlatLadderBook` and `LimitOrderBook`, including deep-ladder bitmap sweeps and steady-state heap allocation probes.

```bash
# Run all benchmarks
./build/sim_bench

# Filter benchmarks by substring
./build/sim_bench --benchmark_filter=BM_FlatLadder
./build/sim_bench --benchmark_filter=AllocProbe
```

Key benchmark families:
- `BM_FlatLadder_AddNoCross` / `BM_LOB_AddNoCross`: Passive limit order placement without crossing.
- `BM_FlatLadder_CancelCycle` / `BM_LOB_CancelCycle`: Order insertion followed by full FIFO cancellation.
- `BM_FlatLadder_CrossingMatch` / `BM_LOB_CrossingMatch`: Aggressive sweep across resting depth.
- `BM_FlatLadder_DeepCancelCycle` / `BM_FlatLadder_DeepRefreshHeavy`: Sparse orders across a 100,000-tick band exercising the 64-bit word summary bitmap (`ctzll`/`clzll`).
- `*_AllocProbe`: Steady-state heap allocation verification via `include/qr/alloc_guard.hpp` (`ScopedAllocCounter`), asserting `0` allocations per operation on `FlatLadderBook::add_into`.

---

## 4. Architecture Overview

### 4.1 `FlatLadderBook`: `O(1)` Intrusive Limit Order Book

`FlatLadderBook` (`include/sim/flat_ladder_book.hpp`, `src/flat_ladder_book.cpp`) maps a contiguous tick band `[min_price, max_price]` onto a flat array of `PriceLevel` headers backed by a pre-reserved pool of `IntrusiveOrderNode` slots:

- **Intrusive Doubly-Linked FIFO Queues**: Each `PriceLevel` stores `head_slot` and `tail_slot` indices into the pre-allocated node vector. Resting orders are linked via integer slot indices (`prev_slot`, `next_slot`), avoiding pointer chasing and node allocations.
- **`O(1)` Cancellation**: `OrderId` maps directly to its slot index via a flat lookup table. Cancelling an order unlinks its `IntrusiveOrderNode` in `O(1)` time without leaving tombstones or requiring periodic compaction.
- **Two-Level 64-Bit Bitmap Best-Price Discovery**: Active price levels set bits in a 64-bit word array. Refreshing `best_bid` or `best_ask` after a level drains uses hardware bit-scan intrinsics (`__builtin_ctzll` / `__builtin_clzll`) to locate the next populated tick in `O(W)` word steps rather than scanning empty ticks linearly.
- **Zero-Allocation Hot Path (`add_into`)**: Passing a caller-owned `sim::Trade*` buffer to `add_into(...)` records fills in place without touching `std::vector` or the heap.
- **Order Types & Self-Trade Prevention**: Supports `Limit`, `Market`, `PostOnly`, `IOC`, and `FOK` orders alongside `None`, `CancelPassive`, `CancelAggressive`, and `DecrementAndCancel` self-trade prevention (`StpMode`).

### 4.2 `LimitOrderBook`: Tree-Backed Reference Engine

`LimitOrderBook` (`include/sim/lob.hpp`, `src/lob.cpp`) stores price levels in `std::pmr::map` backed by a `monotonic_buffer_resource` and `unsynchronized_pool_resource`. It handles arbitrary unbounded tick prices and compacts cancelled tombstone slots lazily according to `CompactionConfig`.

### 4.3 Queue-Reactive & Hawkes Arrival Engine

`QueueReactiveProcess` (`include/sim/queue_reactive.hpp`) combines:
1. **Queue-Reactive State Dependence (Huang, Lehalle & Rosenbaum, 2015)**: Limit order arrival rates decay with resting queue depth ($\lambda_L(q) = \lambda_0 / (1 + c_L q^\gamma)$), cancellation rates scale with resting depth ($\lambda_C(q) = \lambda_{C,0} q^\delta$), and market order rates increase logarithmically with opposite-side depth.
2. **Mutually-Exciting Hawkes Feedback**: A $6 \times 6$ excitation matrix (`HawkesMatrix`) over `{BuyLimit, SellLimit, BuyCancel, SellCancel, BuyMarket, SellMarket}` captures momentum bursts (market orders exciting same-side market orders) and fleeting liquidity (market orders exciting opposite-side cancellations).
3. **Deterministic PRNG (`Xoshiro256StarStar`)**: All inter-arrival times and event selections draw from a seedable `sim::Rng` with `jump()` / `long_jump()` stream partitioning for reproducible parallel rollouts.

### 4.4 Zero-Allocation Telemetry (`include/qr/alloc_guard.hpp`)

All entrypoints (`src/main.cpp`, `bench/main.cpp`, `tests/main.cpp`) include `include/qr/alloc_guard.hpp`, which overrides global `operator new` / `operator delete` (respecting `-fno-exceptions` via `#if defined(__cpp_exceptions)` and disabling overrides when AddressSanitizer or ThreadSanitizer is active) and routes active allocation counts to thread-local `sim::diag::ScopedAllocCounter` instances.
