# Experimental Replay and Parity Surface

Headers in this directory provide offline calibration, shared-memory transport, and cross-implementation parity checks. They are compiled into `sim_test` when `QRLOB_BUILD_EXPERIMENTAL=ON` (the default for `make test`) and excluded when `QRLOB_BUILD_EXPERIMENTAL=OFF` (`make test_core`).

## Isolation Boundary

Nothing in `include/sim/` or `src/` includes headers from `experimental/`. The matching core (`FlatLadderBook` and `LimitOrderBook`) compiles without exceptions (`-fno-exceptions`) and performs no heap allocations on its pre-sized hot path. Data flows strictly outward from matching engine events or recorded traces into these offline tools.

## Headers

- `experimental/spsc_ring_bridge.hpp` implements a lock-free POSIX shared-memory single-producer single-consumer ring (`ShmIpcEvent`, `ShmIpcHeader`, `ShmRingChannel`, `ShmSpscProducer`, `ShmSpscConsumer`, `ShmSpscEndpoint`) using a 64-byte cache-line-aligned event layout.
- `experimental/offline_hawkes_calibrator.hpp` provides offline dyadic Hawkes intensity replay (`ArrivalPointProcessReplay`), queue-reactive state-bucketed arrival calibration (`StateDependentHawkesCalibrator`), and equal-volume VPIN toxicity calculation (`VpinToxicityReplay`).
- `experimental/flat_ladder_parity_harness.hpp` verifies event-by-event replay equivalence between `sim::FlatLadderBook`, `sim::LimitOrderBook`, a reference tombstone-array ladder (`ByteFrozenTombstoneLadder`), and a static van Emde Boas tree layout (`VebStaticSearch`).
