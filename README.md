# Queue-Reactive Order Book

C++20 price-time priority limit order book and queue-reactive market-making simulator.

The matching core compiles cleanly under `-fno-exceptions` and runs without heap allocations on the steady-state matching and cancel paths once the book is constructed for its expected order capacity. Two self-contained header shims (`include/test_shim/`) cover the Catch2 and Google Benchmark interfaces used by `sim_test` and `sim_bench`, so the repository builds with a standard C++20 toolchain and no external package manager.

## Research Question

How does order-entry latency shift the balance between spread capture and adverse selection?

Faster cancels shed resting quotes before a mid-price move sweeps them. Slower cancels hold queue priority longer when the inside level remains stable. Holding the matching rules fixed while sweeping quote latency against spread width and inventory penalties exposes how those quantities trade off on the same order book.

## Book Implementations

Two order book engines share the same price-time (FIFO) matching rules across all supported order types and self-trade prevention modes:

1. `FlatLadderBook` (`include/sim/flat_ladder_book.hpp`, `src/flat_ladder_book.cpp`) maps a bounded `[min_price, max_price]` tick range onto a flat `std::pmr` level array paired with a two-level 64-bit word summary bitmap (`ctzll`/`clzll` inside-price discovery). Resting orders live in a pre-reserved intrusive doubly-linked slot pool (`IntrusiveOrderNode`), so `cancel(OrderId)` unlinks in `O(1)` time without tombstones or compaction sweeps. When callers supply a pre-allocated `Trade*` buffer via `add_into(...)`, both passive placement and multi-fill crosses execute with zero heap allocations.
2. `LimitOrderBook` (`include/sim/lob.hpp`, `src/lob.cpp`) stores active price levels in a `std::pmr::map` backed by an upstream `monotonic_buffer_resource` and `unsynchronized_pool_resource`. Cancelled orders mark tombstone slots that compact lazily when dead slots exceed configurable thresholds (`CompactionConfig`), making it a general-purpose reference engine for unbounded tick ranges.

Both engines support five order types (`Limit`, `Market`, `PostOnly`, `IOC`, `FOK`) and four self-trade prevention policies (`None`, `CancelPassive`, `CancelAggressive`, `DecrementAndCancel`). Calling `FlatLadderBook::queue_ahead(OrderId)` walks the intrusive FIFO list from `head_slot` to the target node in `O(k)` steps (where `k` is the number of resting orders ahead at that tick) to report exact volume ahead of a resting quote.

## Modules

Headers under `include/sim/` group into four layers:

- Core matching and arrival process: `types.hpp`, `flat_ladder_book.hpp`, `lob.hpp`, and `queue_reactive.hpp` (`QueueReactiveProcess`, coupling Huang-Lehalle-Rosenbaum queue-size tables with a dyadic Hawkes arrival intensity).
- Observation and quoting controls: `queue_state_observer.hpp` (normalized queue depletion and touch imbalance), `ofi_sparsity_features.hpp` (multi-level order flow imbalance and gap ratios), `quote_adapters.hpp` (discrete tick-offset and continuous Avellaneda-Stoikov targets), and `risk_guard.hpp` (hard position limits paired with inventory skew and drawdown circuit breakers).
- Evaluation and transaction-cost accounting: `reward_ledger.hpp` (per-step PnL decomposition into cash flow and mark-to-mid changes net of inventory penalties and terminal liquidation), `execution_baselines.hpp` (TWAP schedules alongside passive-pegged and immediate-sweep benchmarks), `tca.hpp` and `tca_diagnostics.hpp` (implementation shortfall and adverse selection attribution), `impact_diagnostics.hpp` (response and cross-impact curves), `survival_labels.hpp` (competing-risks fill vs. cancel labels), and `exploitability_probe.hpp` (frozen-opponent best-response probes).
- Replay and schema contracts: `replay_checkpoint.hpp` (deterministic book and PRNG snapshot validation), `export_schema.hpp`, `feature_export_contract.hpp`, `join_manifest.hpp`, `dataset_size_guard.hpp`, `alloc_probe.hpp`, and `bitwise_veb_codings.hpp`.

Offline replay and cross-implementation parity harnesses live in [`experimental/`](experimental/README.md) behind the `QRLOB_BUILD_EXPERIMENTAL` build flag. Nothing in `src/` or `include/sim/` depends on `experimental/`.

## Benchmarks

Measured with `./build/sim_bench` on a dual AMD EPYC 7453 (56 cores / 112 threads, Linux x86_64, GCC 16):

| Benchmark (`N = 4096` unless noted) | Default (`-O2`) | Release LTO (`-O3 -flto=auto -march=native`) | Heap Allocs / Op |
| --- | --- | --- | --- |
| `BM_FlatLadder_CancelCycle/4096` | 104.4 M ops/s (9.6 ns/op) | 150.7 M ops/s (6.6 ns/op) | 0 |
| `BM_FlatLadder_CrossingMatch/4096` | 33.9 M ops/s (29.5 ns/op) | 96.3 M ops/s (10.4 ns/op) | - |
| `BM_FlatLadder_AddIntoCrossing_AllocProbe/4096` | 22.1 M ops/s (45.2 ns/op) | 37.8 M ops/s (26.5 ns/op) | 0 (`0.0 B`) |
| `BM_FlatLadder_AddNoCross_AllocProbe/4096` | 19.5 M ops/s (51.4 ns/op) | 37.3 M ops/s (26.8 ns/op) | 0 (`0.0 B`) |
| `BM_FlatLadder_DeepRefreshHeavy/4096` (100k-tick band) | 45.9 M ops/s (21.8 ns/op) | 80.6 M ops/s (12.4 ns/op) | 0 |
| `BM_FlatLadder_STP_CancelPassive/4096` | 18.4 M ops/s (54.4 ns/op) | 24.6 M ops/s (40.6 ns/op) | 0 |
| `BM_FlatLadder_QueueAhead/1024` (all 1024 on 1 tick, `O(k)` walk) | 0.58 M queries/s (1.72 us/op) | 0.73 M queries/s (1.37 us/op) | 0 |

Reproduce on your machine:

```bash
make bench
make release-lto && ./build/sim_bench
```

## Build and Test

Requires a C++20 compiler (`g++` or `clang++`) and `make` (or CMake >= 3.16).

```bash
make              # Build standalone simulation CLI (build/sim)
./build/sim       # Run 10,000-event queue-reactive simulation with allocation check
make test         # Build and run full test suite including experimental surface (build/sim_test)
make test_core    # Build and run core-only tests (QRLOB_BUILD_EXPERIMENTAL=OFF)
make bench        # Build and run microbenchmarks (build/sim_bench)
```

Sanitizer and CMake builds:

```bash
make asan         # AddressSanitizer + UndefinedBehaviorSanitizer
make ubsan        # UndefinedBehaviorSanitizer
make tsan         # ThreadSanitizer
make debug        # Unoptimized debug build (-O0 -g3)

cmake -B build_cmake -DQRLOB_BUILD_EXPERIMENTAL=ON
cmake --build build_cmake
ctest --test-dir build_cmake --output-on-failure
```

## References

- Huang, Lehalle & Rosenbaum (2015), *Simulating and analyzing order book data: The queue-reactive model*, JASA / arXiv:1312.0563
- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance
- Gueant, Lehalle & Fernandez-Tapia (2013), *Dealing with the Inventory Risk*, arXiv:1105.3115
- Gao & Wang (2018), *Optimal Market Making in the Presence of Latency*, arXiv:1806.05849
- Spooner et al. (2018), *Market Making via Reinforcement Learning*, arXiv:1804.04216

## License

Dual-licensed under [`LICENSE`](LICENSE):

1. **PolyForm Noncommercial License 1.0.0** for non-commercial academic research and educational evaluation with attribution.
2. **Commercial License & AI/Automated-Translation Provenance Notice** for commercial or proprietary trading deployment (including quantitative fund production use and automated or AI-assisted code translation). Commercial inquiries: Radu Irbe (GitHub: `r-irbe`).
