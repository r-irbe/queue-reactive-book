// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "sim/alloc_probe.hpp"

#include <cstdlib>
#include <new>

namespace qr::alloc_guard {

using AllocStats = sim::diag::AllocStats;
using ScopedAllocCounter = sim::diag::ScopedAllocCounter;
using AllocBudget = sim::diag::AllocBudget;
using AllocBudgetResult = sim::diag::AllocBudgetResult;

inline bool alloc_counting_active() noexcept {
    return sim::diag::alloc_counting_active();
}

inline void note_alloc(std::size_t n) noexcept {
    sim::diag::note_alloc(n);
}

inline AllocBudgetResult check_alloc_budget(const AllocStats& stats,
                                            const AllocBudget& budget) noexcept {
    return sim::diag::check_alloc_budget(stats, budget);
}

inline void* aligned_alloc_impl(std::size_t size, std::size_t align) noexcept {
    if (align < sizeof(void*)) align = sizeof(void*);
    void* p = nullptr;
    if (posix_memalign(&p, align, size == 0 ? 1 : size) != 0) return nullptr;
    return p;
}

} // namespace qr::alloc_guard

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define QRLOB_SANITIZER_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define QRLOB_SANITIZER_ACTIVE 1
#endif
#endif

#if !defined(QRLOB_SANITIZER_ACTIVE) && !defined(QRLOB_NO_GLOBAL_NEW_OVERRIDE)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

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
    void* p = qr::alloc_guard::aligned_alloc_impl(size, static_cast<std::size_t>(al));
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
    void* p = qr::alloc_guard::aligned_alloc_impl(size, static_cast<std::size_t>(al));
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
    void* p = qr::alloc_guard::aligned_alloc_impl(size, static_cast<std::size_t>(al));
    if (p && sim::diag::alloc_counting_active()) sim::diag::note_alloc(size);
    return p;
}

void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t&) noexcept {
    void* p = qr::alloc_guard::aligned_alloc_impl(size, static_cast<std::size_t>(al));
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
#endif
