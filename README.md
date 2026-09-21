# Queue-reactive order book

C++20 limit-order-book and queue-reactive market-making simulator.
Throughput figures are machine-local and workload-scoped.

## Research question

How does order-entry latency change the trade-off between spread capture and
inventory risk?

A market maker that is faster to cancel stale quotes leaks less inventory when
the mid moves; a market maker that is slower captures more spread when the
queue is stable. The simulator holds the matching rules fixed and varies
latency, spread, and inventory constraints so those three quantities can be
read against each other on the same book.

## Two book implementations

- `LimitOrderBook` (`include/sim/lob.hpp`, `src/lob.cpp`): `std::map`-based
  price levels, general-purpose, simple to reason about.
- `FlatLadderBook` (`include/sim/flat_ladder_book.hpp`, `src/flat_ladder_book.cpp`):
  bounded tick ladder, vector-backed levels, FIFO plus tombstones, cached best
  pointers. Compaction is enabled by default (speed-first profile:
  `dead_min=1024`, `total_min=2048`, `dead/total >= 2/3`). Arena sizing is
  automatic via `suggest_arena_bytes(expected_orders)`.

## Scoped benches

Observed on one development machine, one snapshot. Do not treat these as a
cross-machine ranking.

| Workload | Throughput |
| --- | --- |
| `CancelCycle` | ~58–65 M items/s |
| `CompactionStress` | ~28–30 M items/s |

For `CancelCycle` at 4096 orders, arena sweeps favored ~1 MiB; auto-sizing
stayed close across scales.

```bash
make bench
```

## Papers

- Gao & Wang, *Optimal Market Making in the Presence of Latency*, arXiv:1806.05849
- Guéant, Lehalle & Fernandez-Tapia, *Dealing with the Inventory Risk*, arXiv:1105.3115
- Huang, Lehalle & Rosenbaum, *Simulating and analyzing order book data: The queue-reactive model*, arXiv:1312.0563
- Spooner et al., *Market Making via Reinforcement Learning*, arXiv:1804.04216

## Build

Requires a C++20 compiler, `make`, and `git`.

```bash
# once, local only
git clone --depth 1 https://github.com/microsoft/vcpkg .vcpkg
.vcpkg/bootstrap-vcpkg.sh -disableMetrics

make deps          # triplet auto-detected (arm64-osx, x64-osx, x64-linux)
make build
make test
make bench
make example
make clean
```

Override the triplet with `VCPKG_TRIPLET=...`. Custom vcpkg:
`make deps VCPKG=/path/to/vcpkg VCPKG_ROOT=/path/to/repo`.

Sanitizer / debug knobs: `make test MODE=asan` (also `ubsan`, `tsan`, `debug`).
`make bench MODE=release-lto` builds the benchmark binary with `-O3 -flto
-march=native`.

## Layout

```text
include/sim/     book types, replay, features, TCA
include/offline/ header-only diagnostic replay fixtures
include/gov/     governance-label replay (offline, cold path)
src/             LimitOrderBook + FlatLadderBook
tests/           Catch2 suite
bench/           Google Benchmark workloads
examples/        RapidJSON config example
```

## License

Apache-2.0.
