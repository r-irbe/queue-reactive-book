#pragma once

// Allocation probe (CMU-002, post-Tranche-4 code synthesis).
//
// Offline, opt-in measurement utility for counting heap allocations across a
// scoped region. It is part of the around-the-book OBSERVABILITY layer and has
// no effect on matching/replay results: the matching kernel
// (`flat_ladder_book.{hpp,cpp}`) is frozen and never includes this header.
//
// Design (per rubber-duck critique):
//   * This header defines ONLY a thread-local counter, an arm-depth, and a
//     RAII `ScopedAllocCounter`. Reading/incrementing the counter is the only
//     thing it does -- it never replaces global `operator new`/`operator
//     delete`.
//   * The actual global allocation-operator replacement lives ONLY in the
//     benchmark translation unit (`bench/main.cpp`), which is a standalone
//     binary. The test and library binaries are NOT affected, so Catch2's
//     static registration and the `sim` library keep the platform allocator.
//   * Counting is active only while `alloc_arm_depth() > 0` (i.e. inside a
//     live `ScopedAllocCounter`). When disarmed, the replacement operator is a
//     thin pass-through and the count is untouched.
//
// Because arming/disarming and reading only touch a thread-local integer, a
// `ScopedAllocCounter` is allocation-hash neutral: wrapping a replay in one
// does not change `ReplayDiagnostics::replay_hash`.

#include <cstddef>
#include <cstdint>

namespace sim::diag {

struct AllocStats {
    std::uint64_t allocations = 0;
    std::uint64_t bytes = 0;
};

// Counting is active when the arm-depth is positive. Nested scopes increment
// the depth, so counting stays active until the outermost scope ends.
inline int& alloc_arm_depth() noexcept {
    static thread_local int depth = 0;
    return depth;
}

inline AllocStats& alloc_tls() noexcept {
    static thread_local AllocStats stats;
    return stats;
}

inline bool alloc_counting_active() noexcept { return alloc_arm_depth() > 0; }

// Called by the benchmark TU's replacement `operator new` when armed.
inline void note_alloc(std::size_t n) noexcept {
    AllocStats& stats = alloc_tls();
    ++stats.allocations;
    stats.bytes += n;
}

// RAII region counter. Snapshots the running totals on construction and arms
// counting; `delta()` returns allocations/bytes observed within the scope.
class ScopedAllocCounter {
public:
    ScopedAllocCounter() noexcept : start_(alloc_tls()) { ++alloc_arm_depth(); }

    ~ScopedAllocCounter() {
        if (alloc_arm_depth() > 0) --alloc_arm_depth();
    }

    ScopedAllocCounter(const ScopedAllocCounter&) = delete;
    ScopedAllocCounter& operator=(const ScopedAllocCounter&) = delete;
    ScopedAllocCounter(ScopedAllocCounter&&) = delete;
    ScopedAllocCounter& operator=(ScopedAllocCounter&&) = delete;

    AllocStats delta() const noexcept {
        const AllocStats now = alloc_tls();
        return AllocStats{now.allocations - start_.allocations,
                          now.bytes - start_.bytes};
    }

private:
    AllocStats start_;
};

// --- Allocation budget gate (Tier-2 PROTOTYPE, offline/diagnostics only) -------
//
// Turns the allocation probe from a passive counter into a GATE for offline /
// replay paths: an offline region may declare a budget (max allocations and/or
// bytes) and assert it stayed within it. This is diagnostics-only -- it is
// evaluated AFTER a scoped region from `ScopedAllocCounter::delta()`, never on
// the matching hot path, and the frozen kernel never includes this header. A
// disabled budget always passes, so wiring a budget in is allocation-hash
// neutral.

struct AllocBudget {
    bool enabled = false;
    std::uint64_t max_allocations = 0;  // 0 = unlimited for this axis
    std::uint64_t max_bytes = 0;        // 0 = unlimited for this axis
};

struct AllocBudgetResult {
    bool within = true;
    bool allocations_exceeded = false;
    bool bytes_exceeded = false;
    std::uint64_t allocations = 0;
    std::uint64_t bytes = 0;
};

inline AllocBudgetResult check_alloc_budget(const AllocStats& stats,
                                            const AllocBudget& budget) noexcept {
    AllocBudgetResult out{};
    out.allocations = stats.allocations;
    out.bytes = stats.bytes;
    if (!budget.enabled) return out;  // disabled budget never fails
    if (budget.max_allocations != 0 && stats.allocations > budget.max_allocations) {
        out.allocations_exceeded = true;
    }
    if (budget.max_bytes != 0 && stats.bytes > budget.max_bytes) {
        out.bytes_exceeded = true;
    }
    out.within = !out.allocations_exceeded && !out.bytes_exceeded;
    return out;
}

} // namespace sim::diag
