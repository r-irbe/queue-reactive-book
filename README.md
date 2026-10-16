# Queue-reactive order book

C++20 zero-allocation limit order book and queue-reactive market-making simulator.
Engineered for nanosecond-scale HFT simulation and deterministic matching.

## Architectural Highlights

- **Zero-Exception Core (`-fno-exceptions`)**: In high-frequency trading (HFT) and
  low-latency systems, stack unwinding and runtime exception tables introduce jitter,
  cache pollution, and non-deterministic overhead. The entire matching core is
  compiled and verified clean under `-fno-exceptions`.
- **Zero-Tombstone Intrusive Ladder**: Unlike traditional array ladders that leave
  tombstones requiring compaction pauses, `FlatLadderBook` implements an intrusive
  doubly-linked slotmap pool. Order cancellation achieves deterministic O(1) unlinking
  and instant slot recycling with zero compaction pauses.
- **Extended Order Types**:
  - `Limit`: Standard resting or crossing limit order.
  - `Market`: Immediate taker execution against top of book; residual drops.
  - `PostOnly`: Guaranteed maker order; rejected if it would cross the spread.
  - `IOC` (Immediate-Or-Cancel): Fills available depth up to limit price; residual cancels.
  - `FOK` (Fill-Or-Kill): Two-phase execution; fills 100% of order volume or cancels completely.
- **Self-Trade Prevention (STP)**: Full support for `None`, `CancelPassive`,
  `CancelAggressive`, and `DecrementAndCancel` policies at order and engine levels.
- **L3 FIFO Queue Priority Tracking**: Exact, ground-truth ahead volume calculation
  (`queue_ahead(OrderId)`) for real-time queue-depletion and cancel-risk modeling.
- **Queue-Reactive Intensity Model**: Built-in state-dependent queue-reactive process
  (Huang, Lehalle, Rosenbaum 2015) coupled with a multi-dimensional Hawkes arrival process.
- **Zero External Dependencies**: Self-contained Catch2 and Google Benchmark shims
  enable immediate out-of-the-box building without vcpkg or external package managers.

## Research Question

How does order-entry latency change the trade-off between spread capture and
inventory risk?

A market maker that is faster to cancel stale quotes leaks less inventory when
the mid moves; a market maker that is slower captures more spread when the
queue is stable. The simulator holds the matching rules fixed and varies
latency, spread, and inventory constraints so those three quantities can be
read against each other on the same book.

## Two Book Implementations

- `LimitOrderBook` (`include/sim/lob.hpp`, `src/lob.cpp`): `std::map`-based
  price levels, general-purpose, simple to reason about.
- `FlatLadderBook` (`include/sim/flat_ladder_book.hpp`, `src/flat_ladder_book.cpp`):
  Bounded tick ladder with intrusive doubly-linked slots, O(N/64) bitmap level search,
  deterministic O(1) cancel unlinking without tombstones, and arena-backed flat memory.

## Benchmarks

Measured on local hardware (dual AMD EPYC 7453 56-core, 112 threads):

| Workload | Throughput | Mean Latency |
| --- | --- | --- |
| `CancelCycle` (Intrusive Slotmap) | >85 M items/s | <12 ns/op |
| `QueueAhead` (L3 Position Query) | >120 M items/s | <8 ns/op |
| `STP Matching` (CancelPassive) | >35 M matches/s | <28 ns/op |

Run benchmarks:

```bash
make bench
```

Or run release with LTO:

```bash
make release-lto
```

## Papers & Foundational References

- Huang, Lehalle & Rosenbaum, *Simulating and analyzing order book data: The queue-reactive model*, arXiv:1312.0563
- Gao & Wang, *Optimal Market Making in the Presence of Latency*, arXiv:1806.05849
- Gueant, Lehalle & Fernandez-Tapia, *Dealing with the Inventory Risk*, arXiv:1105.3115
- Spooner et al., *Market Making via Reinforcement Learning*, arXiv:1804.04216

## Build & Test

Requires a C++20 compiler (`g++` or `clang++`) and `make`.

```bash
make          # Build CLI simulation driver (build/sim)
make test     # Build and run full unit test suite (build/sim_test)
make bench    # Build and run microbenchmarks (build/sim_bench)
./build/sim   # Run standalone queue-reactive simulation
```

Sanitizers and diagnostic modes:

```bash
make test MODE=asan    # AddressSanitizer + UBSan
make test MODE=ubsan   # UndefinedBehaviorSanitizer
make test MODE=tsan    # ThreadSanitizer
make test MODE=debug   # Debug symbols (-O0 -g3)
```

## Layout

```text
include/sim/       Matching engine headers (flat ladder, lob, types, queue-reactive)
include/test_shim/ Zero-dependency Catch2 and Google Benchmark compatibility shims
src/               Matching engine implementations and simulation driver CLI
tests/             Catch2 unit tests (STP, queue position, order types, invariants)
bench/             Google Benchmark microbenchmarks
```

## License

This software is licensed under the **PolyForm Noncommercial License 1.0.0**
with an explicit **Automated Code Synthesis & AI Reimplementation Covenant**.

- **Academic, Educational, and Non-Commercial Research**: Free and open to use,
  modify, and redistribute with source attribution.
- **Commercial Use & Commercial AI Reimplementation**: Prohibited without a separate
  commercial license from the author. Using this software to prompt or instruct AI/LLM
  systems to generate commercial derivative works or functional clones constitutes
  an unlicensed derivative work.
- **Commercial Licensing Inquiries**: Contact Radu Irbe (GitHub: `r-irbe`).
