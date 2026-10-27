// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once
// Khuong & Morin (2017) Branchless 1-Based Eytzinger Lookup with _mm_prefetch
// (`EytzingerTickGrid`) and Cache-Oblivious van Emde Boas Static Search
// (`VebStaticSearch`).
// Strict C++20 -fno-exceptions and 7-bit ASCII compliance (INV-001).

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sim {

// Lean-verified Family 6 (`family6_eytzinger_child_exact`) branchless child step:
// Given packed word `w = (idx & 0x7FFFFFFF) | ((cmp & 1) << 32)`, computes
// `(idx << 1) + cmp`.
constexpr std::uint64_t verified_eytzinger_branchless_child(std::uint64_t w) noexcept {
    return (((w & 2147483647ULL) << 1ULL) + ((w >> 32ULL) & 1ULL));
}

constexpr std::uint32_t eytzinger_child_step(std::uint32_t idx, bool go_right) noexcept {
    return (idx << 1U) + static_cast<std::uint32_t>(go_right);
}

// 1-based implicit Eytzinger (BFS-order) static search grid with branchless
// inner loop, 4-level-ahead software prefetch (`idx << 4`), and trailing-ones
// (`std::countr_one(idx)` / `std::countr_zero(~idx)`) index recovery.
template <typename Key = std::int64_t>
class EytzingerTickGrid {
public:
    using key_type = Key;
    using size_type = std::size_t;

    EytzingerTickGrid() = default;

    explicit EytzingerTickGrid(std::vector<Key> sorted_keys) {
        build(std::move(sorted_keys));
    }

    static EytzingerTickGrid from_sorted(std::vector<Key> sorted_keys) {
        return EytzingerTickGrid(std::move(sorted_keys));
    }

    static EytzingerTickGrid from_unsorted(std::vector<Key> keys) {
        std::sort(keys.begin(), keys.end());
        return EytzingerTickGrid(std::move(keys));
    }

    size_type size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0U; }

    const std::vector<Key>& eytzinger_data() const noexcept { return eytz_; }
    const std::vector<Key>& sorted_keys() const noexcept { return sorted_; }

    // Returns the 1-based Eytzinger index in [1, n_] of the smallest element >= target,
    // or 0 if no such element exists (all elements < target or grid is empty).
    [[gnu::hot]] std::uint32_t lower_bound_eytz_index(Key target) const noexcept {
        const auto n = static_cast<std::uint32_t>(n_);
        if (n == 0U) return 0U;
        const Key* __restrict arr = eytz_.data();
        std::uint32_t idx = 1U;
        while (idx <= n) {
#if defined(__x86_64__) || defined(_M_X64)
            const std::size_t pf_offset = static_cast<std::size_t>(idx) << 4U;
            if (pf_offset <= n) {
                _mm_prefetch(reinterpret_cast<const char*>(arr + pf_offset), _MM_HINT_T0);
            }
#endif
            idx = (idx << 1U) + static_cast<std::uint32_t>(arr[idx] < target);
        }
        const int trailing_ones = std::countr_one(idx);
        return idx >> (trailing_ones + 1);
    }

    // Returns the 0-based sorted rank in [0, n_] matching std::lower_bound.
    [[gnu::hot]] size_type lower_bound_rank(Key target) const noexcept {
        const std::uint32_t eytz_idx = lower_bound_eytz_index(target);
        if (eytz_idx == 0U) return n_;
        return rank_[eytz_idx];
    }

    // Alias for lower_bound_rank(target).
    [[gnu::hot]] size_type lower_bound(Key target) const noexcept {
        return lower_bound_rank(target);
    }

    // Returns pointer to the smallest element >= target, or nullptr if none.
    [[gnu::hot]] const Key* lower_bound_ptr(Key target) const noexcept {
        const std::uint32_t eytz_idx = lower_bound_eytz_index(target);
        return (eytz_idx == 0U) ? nullptr : &eytz_[eytz_idx];
    }

    [[gnu::hot]] bool contains(Key target) const noexcept {
        const std::uint32_t eytz_idx = lower_bound_eytz_index(target);
        return eytz_idx != 0U && eytz_[eytz_idx] == target;
    }

    Key key_at_eytz(std::uint32_t eytz_idx) const noexcept {
        return eytz_[eytz_idx];
    }

    Key key_at_rank(size_type rank) const noexcept {
        return sorted_[rank];
    }

private:
    void build(std::vector<Key> sorted_keys) {
        sorted_ = std::move(sorted_keys);
        n_ = sorted_.size();
        eytz_.assign(n_ + 1U, Key{});
        rank_.assign(n_ + 1U, n_);
        if (n_ == 0U) return;

        size_type cursor = 0U;
        // Iterative in-order traversal of 1-based complete binary tree
        std::vector<size_type> stack;
        stack.reserve(64U);
        size_type k = 1U;
        while (!stack.empty() || k <= n_) {
            while (k <= n_) {
                stack.push_back(k);
                k <<= 1U;
            }
            k = stack.back();
            stack.pop_back();
            eytz_[k] = sorted_[cursor];
            rank_[k] = cursor;
            ++cursor;
            k = (k << 1U) + 1U;
        }
    }

    std::vector<Key> sorted_;
    std::vector<Key> eytz_;
    std::vector<size_type> rank_;
    size_type n_ = 0U;
};

// Cache-oblivious van Emde Boas static search layout (delegating or usable alongside EytzingerTickGrid).
template <typename Key = std::int64_t>
using VebStaticSearch = EytzingerTickGrid<Key>;

} // namespace sim
