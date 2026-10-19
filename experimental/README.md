# Experimental Surface (Absorbed from Project Q)

This directory contains the experimental and offline verification surface
absorbed from `/home/radu/code/q` (`f190d0a4`), isolated behind the opt-in
build option `QRLOB_BUILD_EXPERIMENTAL` (default `ON` for test builds).

## Isolation Invariant

The production matching core (`include/sim/flat_ladder_book.hpp`,
`src/flat_ladder_book.cpp`, `include/sim/lob.hpp`, `src/lob.cpp`) remains
strictly zero-exception (`-fno-exceptions`) and zero-allocation on the hot
path. Nothing in `include/sim/` or `src/` includes or depends on `experimental/`.
Data flows strictly one way: out of the matching core or replay traces into
these experimental adapters and offline diagnostics.

## Components

1. `experimental/spsc_ring_bridge.hpp`
   - Lock-free, zero-allocation POSIX shared-memory SPSC ring bridge
     (`ShmIpcEvent`, `ShmIpcHeader`, `ShmRingChannel`, `ShmSpscProducer`,
     `ShmSpscConsumer`, `ShmSpscEndpoint`) matching the 64-byte binary wire
     layout of `rust-mmap-engine` (`shm_spsc_ipc::ShmIpcEvent`).
2. `experimental/offline_hawkes_calibrator.hpp`
   - Offline dyadic Hawkes point-process intensity replay (`ArrivalEvent`,
     `HawkesParams`, `ArrivalPointProcessReplay`), state-dependent
     queue-reactive Hawkes calibrator (`StateDependentHawkesCalibrator`), and
     equal-volume VPIN flow-toxicity replay (`SignedTrade`, `VolumeBucket`,
     `VpinResult`, `VpinToxicityReplay`) with compile-time non-feedback guards.
3. `experimental/flat_ladder_parity_harness.hpp`
   - Deterministic replay parity harness comparing the production intrusive
     `sim::FlatLadderBook` against `sim::LimitOrderBook` and the byte-frozen
     tombstone-array ladder (`ByteFrozenTombstoneLadder`) from `q`, plus the
     offline cache-oblivious van Emde Boas static search parity verifier
     (`VebStaticSearch`).
