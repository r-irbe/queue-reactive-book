#pragma once

// Thread-local heap allocation counter and budget gate for scoped regions.
// Binaries that replace global `operator new` forward active allocations to
// `note_alloc(size)` while `ScopedAllocCounter` is armed (`alloc_arm_depth() > 0`).

#include <cstddef>
#include <cstdint>

namespace sim::diag {

struct AllocStats {
    std::uint64_t allocations = 0;
    std::uint64_t bytes = 0;
};

inline int& alloc_arm_depth() noexcept {
    constinit static thread_local int depth = 0;
    return depth;
}

inline AllocStats& alloc_tls() noexcept {
    constinit static thread_local AllocStats stats{};
    return stats;
}

inline bool alloc_counting_active() noexcept { return alloc_arm_depth() > 0; }

inline void note_alloc(std::size_t n) noexcept {
    AllocStats& stats = alloc_tls();
    ++stats.allocations;
    stats.bytes += n;
}

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

// Evaluates observed `AllocStats` against an optional allocation/byte ceiling.
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
    if (!budget.enabled) return out;
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
