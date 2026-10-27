// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once
// Fixed-width bitwise van Emde Boas (vEB) bitsets (128/256/512-bit), MSB-first
// bitboard, offset-binary tick codec, Gray-code subset evaluator, and Elias-Fano
// monotone sequence index.

#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sim {

// 1. 128-bit Two-Word Bitwise vEB Node (16 bytes, zero summary overhead)
struct alignas(16) BitwiseVeb128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr void insert(std::uint32_t x) noexcept {
        if (x < 64) lo |= (1ULL << x);
        else        hi |= (1ULL << (x & 63U));
    }

    constexpr void remove(std::uint32_t x) noexcept {
        if (x < 64) lo &= ~(1ULL << x);
        else        hi &= ~(1ULL << (x & 63U));
    }

    constexpr int find_min() const noexcept {
        if (lo != 0ULL) return std::countr_zero(lo);
        if (hi != 0ULL) return 64 + std::countr_zero(hi);
        return -1;
    }

    constexpr int find_max() const noexcept {
        if (hi != 0ULL) return 127 - std::countl_zero(hi);
        if (lo != 0ULL) return 63 - std::countl_zero(lo);
        return -1;
    }
};

// 2. 256-bit Single-Cache-Line Half-Node (32 bytes, AVX2 vpmovmskb accelerated)
struct alignas(32) BitwiseVeb256Avx2 {
    std::uint64_t words[4] = {0, 0, 0, 0};

    constexpr void insert(std::uint32_t x) noexcept {
        words[(x >> 6) & 3U] |= (1ULL << (x & 63U));
    }

    constexpr void remove(std::uint32_t x) noexcept {
        words[(x >> 6) & 3U] &= ~(1ULL << (x & 63U));
    }

    inline int find_min() const noexcept {
#if defined(__AVX2__)
        __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(words));
        __m256i zero = _mm256_setzero_si256();
        __m256i eq = _mm256_cmpeq_epi64(v, zero);
        std::uint32_t non_zero_mask = (~static_cast<std::uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(eq)))) & 0xFU;
        if (non_zero_mask == 0U) return -1;
        std::uint32_t w = static_cast<std::uint32_t>(std::countr_zero(non_zero_mask));
        return static_cast<int>((w << 6) | static_cast<std::uint32_t>(std::countr_zero(words[w])));
#else
        for (std::uint32_t w = 0; w < 4; ++w) {
            if (words[w] != 0ULL) {
                return static_cast<int>((w << 6) | static_cast<std::uint32_t>(std::countr_zero(words[w])));
            }
        }
        return -1;
#endif
    }
};

// 3. 512-bit Full-Cache-Line Node (64 bytes, 8x uint64_t)
struct alignas(64) BitwiseVeb512 {
    std::uint64_t words[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    constexpr void insert(std::uint32_t x) noexcept {
        words[(x >> 6) & 7U] |= (1ULL << (x & 63U));
    }

    constexpr void remove(std::uint32_t x) noexcept {
        words[(x >> 6) & 7U] &= ~(1ULL << (x & 63U));
    }

    constexpr std::uint8_t lane_mask_k8() const noexcept {
        return static_cast<std::uint8_t>(
            ((words[0] != 0ULL) ? 1U : 0U) |
            ((words[1] != 0ULL) ? 2U : 0U) |
            ((words[2] != 0ULL) ? 4U : 0U) |
            ((words[3] != 0ULL) ? 8U : 0U) |
            ((words[4] != 0ULL) ? 16U : 0U) |
            ((words[5] != 0ULL) ? 32U : 0U) |
            ((words[6] != 0ULL) ? 64U : 0U) |
            ((words[7] != 0ULL) ? 128U : 0U)
        );
    }

    constexpr int find_min() const noexcept {
        std::uint8_t k8 = lane_mask_k8();
        if (k8 == 0U) return -1;
        std::uint32_t w = static_cast<std::uint32_t>(std::countr_zero(k8));
        return static_cast<int>((w << 6) | static_cast<std::uint32_t>(std::countr_zero(words[w])));
    }
};

// 4. ARMv8.0-A / Apple Silicon MSB-First Bitboard (1-cycle clz = find_min)
struct ArmMsbBitboard64 {
    std::uint64_t bits = 0ULL;

    constexpr void insert(std::uint32_t x) noexcept {
        bits |= (1ULL << (63U - (x & 63U)));
    }

    constexpr void remove(std::uint32_t x) noexcept {
        bits &= ~(1ULL << (63U - (x & 63U)));
    }

    constexpr int find_min() const noexcept {
        return (bits == 0ULL) ? -1 : std::countl_zero(bits);
    }

    constexpr int find_max() const noexcept {
        return (bits == 0ULL) ? -1 : (63 - std::countr_zero(bits));
    }
};

// 5. Offset-Binary Signed Tick Codec (1-cycle XOR sign bit flip)
struct OffsetBinaryCodec {
    static constexpr std::uint64_t encode(std::int64_t tick) noexcept {
        return static_cast<std::uint64_t>(tick) ^ (1ULL << 63);
    }

    static constexpr std::int64_t decode(std::uint64_t u) noexcept {
        return static_cast<std::int64_t>(u ^ (1ULL << 63));
    }
};

// 6. Knuth Algorithm L Loopless Gray Subset Evaluator (O(1) 1-cycle tzcnt delta update)
template <std::size_t K>
struct KnuthGrayBasket {
    std::uint64_t step = 0;
    std::uint64_t gray_mask = 0;
    std::int64_t current_value = 0;
    std::array<std::int64_t, K> weights{};

    constexpr explicit KnuthGrayBasket(std::array<std::int64_t, K> w) noexcept : weights(w) {}

    constexpr bool next_subset(std::size_t& out_flip_idx, std::uint64_t& out_mask, std::int64_t& out_val) noexcept {
        constexpr std::uint64_t limit = (1ULL << K) - 1ULL;
        if (step >= limit) return false;
        ++step;
        std::size_t flip = static_cast<std::size_t>(std::countr_zero(step));
        std::uint64_t bit = 1ULL << flip;
        gray_mask ^= bit;
        if ((gray_mask & bit) != 0ULL) {
            current_value += weights[flip];
        } else {
            current_value -= weights[flip];
        }
        out_flip_idx = flip;
        out_mask = gray_mask;
        out_val = current_value;
        return true;
    }
};

// 7. Elias-Fano Quasi-Succinct Monotonic Sequence Index
class EliasFanoIndex {
public:
    explicit EliasFanoIndex(const std::vector<std::uint64_t>& sorted) {
        n_ = sorted.size();
        if (n_ == 0) return;
        std::uint64_t max_val = sorted.back();
        std::uint64_t u = max_val + 1ULL;
        std::uint64_t avg_gap = (u / n_) > 0ULL ? (u / n_) : 1ULL;
        low_bits_ = static_cast<std::uint32_t>(63 - std::countl_zero(avg_gap));
        low_mask_ = (low_bits_ == 0U) ? 0ULL : ((1ULL << low_bits_) - 1ULL);

        std::size_t total_low_bits = n_ * low_bits_;
        lower_words_.assign((total_low_bits + 63U) / 64U + 1U, 0ULL);

        std::size_t max_high = static_cast<std::size_t>(max_val >> low_bits_);
        upper_words_.assign((n_ + max_high + 65U) / 64U, 0ULL);

        for (std::size_t r = 0; r < n_; ++r) {
            std::uint64_t val = sorted[r];
            if (low_bits_ > 0U) {
                std::uint64_t low = val & low_mask_;
                std::size_t bp = r * low_bits_;
                std::size_t w = bp >> 6;
                std::size_t off = bp & 63U;
                lower_words_[w] |= (low << off);
                if (off + low_bits_ > 64U) {
                    lower_words_[w + 1] |= (low >> (64U - off));
                }
            }
            std::size_t high = static_cast<std::size_t>(val >> low_bits_);
            std::size_t upos = r + high;
            upper_words_[upos >> 6] |= (1ULL << (upos & 63U));
        }
    }

    std::uint64_t get(std::size_t rank) const noexcept {
        std::size_t rem = rank;
        std::size_t unary_pos = 0;
        for (std::size_t w = 0; w < upper_words_.size(); ++w) {
            std::uint64_t word = upper_words_[w];
            std::size_t pop = static_cast<std::size_t>(std::popcount(word));
            if (rem < pop) {
                for (std::size_t i = 0; i < rem; ++i) {
                    word &= (word - 1ULL);
                }
                unary_pos = (w << 6) | static_cast<std::size_t>(std::countr_zero(word));
                break;
            }
            rem -= pop;
        }
        std::uint64_t high = static_cast<std::uint64_t>(unary_pos - rank);
        std::uint64_t low = 0ULL;
        if (low_bits_ > 0U) {
            std::size_t bp = rank * low_bits_;
            std::size_t w = bp >> 6;
            std::size_t off = bp & 63U;
            low = lower_words_[w] >> off;
            if (off + low_bits_ > 64U) {
                low |= (lower_words_[w + 1] << (64U - off));
            }
            low &= low_mask_;
        }
        return (high << low_bits_) | low;
    }

private:
    std::size_t n_ = 0;
    std::uint32_t low_bits_ = 0;
    std::uint64_t low_mask_ = 0;
    std::vector<std::uint64_t> lower_words_;
    std::vector<std::uint64_t> upper_words_;
};

inline constexpr std::uint32_t kVebNullTick = 0xFFFFFFFFU;

// Verified 3-level radix-64 (or 6+6 bit) and 4-level radix-4096 coordinate encode/decode helpers
// ported from generated_verified_kernels.rs and Rules.BitwiseVebWarpCodec.
constexpr std::uint32_t verified_veb3_4096_encode(std::uint32_t c1, std::uint32_t b0) noexcept {
    return ((c1 & 63U) << 6U) | (b0 & 63U);
}

constexpr std::uint64_t verified_veb3_4096_encode(std::uint64_t c0, std::uint64_t c1,
                                                  std::uint64_t c2, std::uint64_t c3) noexcept {
    return ((c0 & 0xFFFULL) << 36ULL) | ((c1 & 0xFFFULL) << 24ULL) |
           ((c2 & 0xFFFULL) << 12ULL) | (c3 & 0xFFFULL);
}

constexpr void verified_veb3_4096_decode(std::uint32_t tick, std::uint32_t& out_c1, std::uint32_t& out_b0) noexcept {
    out_c1 = (tick >> 6U) & 63U;
    out_b0 = tick & 63U;
}

constexpr std::uint32_t verified_veb4_16m_encode(std::uint32_t c3, std::uint32_t c2, std::uint32_t c1, std::uint32_t b0) noexcept {
    return ((c3 & 63U) << 18U) | ((c2 & 63U) << 12U) | ((c1 & 63U) << 6U) | (b0 & 63U);
}

// 8. Bijective 16-Byte Reversible vEB Delta Token (F_2 XOR Rollback)
struct ReversibleVebDelta {
    std::uint8_t cluster_idx = 0;
    std::uint64_t summary_xor_mask = 0ULL;
    std::uint64_t cluster_xor_mask = 0ULL;
};

// 9. L1D-Resident 3-Level / 4,096-Tick Bitwise van Emde Boas Tree (584 Payload Bytes, alignas(64))
struct alignas(64) BitwiseVeb4096 {
    static constexpr std::uint32_t kUniverse = 4096U;
    static constexpr std::uint32_t kNullTick = kVebNullTick;
    static constexpr std::size_t kPayloadBytes = 584U;

    union {
        std::uint64_t root = 0ULL;
        std::uint64_t summary;
    };
    union {
        std::uint64_t l2[64] = {0ULL};
        std::uint64_t l1[64];
        std::uint64_t clusters[64];
    };
    std::uint64_t summary_writes = 0ULL;
    std::uint64_t elided_summary_writes = 0ULL;
    std::uint64_t l1_group_summary[5] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL};

    constexpr BitwiseVeb4096() noexcept : root(0ULL), l2{0ULL} {}

    constexpr void clear() noexcept {
        root = 0ULL;
        for (std::size_t i = 0; i < 64; ++i) l2[i] = 0ULL;
        summary_writes = 0ULL;
        elided_summary_writes = 0ULL;
        for (std::size_t i = 0; i < 5; ++i) l1_group_summary[i] = 0ULL;
    }

    constexpr bool empty() const noexcept { return root == 0ULL; }

    constexpr void insert(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return;
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        const std::uint64_t old_w = l2[c];
        l2[c] = old_w | (1ULL << b);
        if (old_w == 0ULL) {
            root |= (1ULL << c);
            ++summary_writes;
        } else {
            ++elided_summary_writes;
        }
    }

    constexpr void remove(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return;
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        const std::uint64_t next_w = l2[c] & ~(1ULL << b);
        l2[c] = next_w;
        if (next_w == 0ULL) {
            root &= ~(1ULL << c);
            ++summary_writes;
        } else {
            ++elided_summary_writes;
        }
    }

    constexpr bool contains(std::uint32_t tick) const noexcept {
        if (tick >= kUniverse) return false;
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        return (l2[c] & (1ULL << b)) != 0ULL;
    }

    constexpr std::uint32_t min_tick() const noexcept {
        if (root == 0ULL) return kNullTick;
        const auto c = static_cast<std::uint32_t>(std::countr_zero(root));
        const auto b = static_cast<std::uint32_t>(std::countr_zero(l2[c]));
        return verified_veb3_4096_encode(c, b);
    }

    constexpr std::uint32_t max_tick() const noexcept {
        if (root == 0ULL) return kNullTick;
        const auto c = static_cast<std::uint32_t>(63 - std::countl_zero(root));
        const auto b = static_cast<std::uint32_t>(63 - std::countl_zero(l2[c]));
        return verified_veb3_4096_encode(c, b);
    }

    constexpr int find_min() const noexcept {
        const std::uint32_t m = min_tick();
        return (m == kNullTick) ? -1 : static_cast<int>(m);
    }

    constexpr int find_max() const noexcept {
        const std::uint32_t m = max_tick();
        return (m == kNullTick) ? -1 : static_cast<int>(m);
    }

    constexpr std::uint32_t next_higher(std::uint32_t tick) const noexcept {
        if (tick >= kUniverse - 1U) return kNullTick;
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        if (b < 63U) {
            const std::uint64_t local_rem = l2[c] & (~0ULL << (b + 1U));
            if (local_rem != 0ULL) {
                return verified_veb3_4096_encode(c, static_cast<std::uint32_t>(std::countr_zero(local_rem)));
            }
        }
        if (c >= 63U) return kNullTick;
        const std::uint64_t rem_summary = root & (~0ULL << (c + 1U));
        if (rem_summary == 0ULL) return kNullTick;
        const auto next_c = static_cast<std::uint32_t>(std::countr_zero(rem_summary));
        const auto next_b = static_cast<std::uint32_t>(std::countr_zero(l2[next_c]));
        return verified_veb3_4096_encode(next_c, next_b);
    }

    constexpr std::uint32_t prev_lower(std::uint32_t tick) const noexcept {
        if (tick == 0U) return kNullTick;
        if (tick >= kUniverse) return max_tick();
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        if (b > 0U) {
            const std::uint64_t local_rem = l2[c] & (~0ULL >> (64U - b));
            if (local_rem != 0ULL) {
                return verified_veb3_4096_encode(c, static_cast<std::uint32_t>(63 - std::countl_zero(local_rem)));
            }
        }
        if (c == 0U) return kNullTick;
        const std::uint64_t rem_summary = root & (~0ULL >> (64U - c));
        if (rem_summary == 0ULL) return kNullTick;
        const auto prev_c = static_cast<std::uint32_t>(63 - std::countl_zero(rem_summary));
        const auto prev_b = static_cast<std::uint32_t>(63 - std::countl_zero(l2[prev_c]));
        return verified_veb3_4096_encode(prev_c, prev_b);
    }

    constexpr int find_next(std::uint32_t tick) const noexcept {
        const std::uint32_t n = next_higher(tick);
        return (n == kNullTick) ? -1 : static_cast<int>(n);
    }

    constexpr int find_prev(std::uint32_t tick) const noexcept {
        const std::uint32_t p = prev_lower(tick);
        return (p == kNullTick) ? -1 : static_cast<int>(p);
    }

    constexpr ReversibleVebDelta insert_reversible(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return {};
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        const std::uint64_t old_cluster = l2[c];
        const std::uint64_t new_cluster = old_cluster | (1ULL << b);
        l2[c] = new_cluster;
        const std::uint64_t cluster_xor_mask = old_cluster ^ new_cluster;
        std::uint64_t summary_xor_mask = 0ULL;
        if (old_cluster == 0ULL && new_cluster != 0ULL) {
            summary_xor_mask = (1ULL << c);
            root ^= summary_xor_mask;
        }
        return ReversibleVebDelta{static_cast<std::uint8_t>(c), summary_xor_mask, cluster_xor_mask};
    }

    constexpr ReversibleVebDelta remove_reversible(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return {};
        std::uint32_t c = 0U, b = 0U;
        verified_veb3_4096_decode(tick, c, b);
        const std::uint64_t old_cluster = l2[c];
        const std::uint64_t new_cluster = old_cluster & ~(1ULL << b);
        l2[c] = new_cluster;
        const std::uint64_t cluster_xor_mask = old_cluster ^ new_cluster;
        std::uint64_t summary_xor_mask = 0ULL;
        if (old_cluster != 0ULL && new_cluster == 0ULL) {
            summary_xor_mask = (1ULL << c);
            root ^= summary_xor_mask;
        }
        return ReversibleVebDelta{static_cast<std::uint8_t>(c), summary_xor_mask, cluster_xor_mask};
    }

    constexpr void rollback_delta(const ReversibleVebDelta& delta) noexcept {
        l2[delta.cluster_idx & 63U] ^= delta.cluster_xor_mask;
        root ^= delta.summary_xor_mask;
    }

    constexpr std::uint32_t intersect_and_count(const BitwiseVeb4096& other, BitwiseVeb4096& out) const noexcept {
        out.clear();
        std::uint64_t cand = root & other.root;
        std::uint64_t act = 0ULL;
        std::uint32_t count = 0U;
        while (cand != 0ULL) {
            const auto c = static_cast<std::uint32_t>(std::countr_zero(cand));
            const std::uint64_t w = l2[c] & other.l2[c];
            if (w != 0ULL) {
                out.l2[c] = w;
                act |= (1ULL << c);
                count += static_cast<std::uint32_t>(std::popcount(w));
            }
            cand &= (cand - 1ULL);
        }
        out.root = act;
        return count;
    }
};

static_assert(alignof(BitwiseVeb4096) == 64, "BitwiseVeb4096 must be 64-byte cache-line aligned");
static_assert(sizeof(BitwiseVeb4096) == 576, "BitwiseVeb4096 must be 576 bytes (9 cache lines)");

// 10. 4-Level Hybrid Sparse Bitwise van Emde Boas Tree (U = 64^4 = 16,777,216)
class SparseBitwiseVeb16M {
public:
    static constexpr std::uint32_t kUniverse = 16777216U;
    static constexpr std::uint32_t kNullTick = kVebNullTick;

    SparseBitwiseVeb16M() noexcept = default;

    std::size_t allocated_pages() const noexcept { return allocated_pages_; }

    std::size_t active_footprint_bytes() const noexcept {
        return sizeof(BitwiseVeb4096) * (1U + allocated_pages_);
    }

    bool empty() const noexcept { return top_.empty(); }

    void clear() noexcept {
        top_.clear();
        for (auto& page : pages_) page.reset();
        allocated_pages_ = 0U;
    }

    void insert(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return;
        const std::uint32_t page_idx = (tick >> 12U) & 0xFFFU;
        const std::uint32_t low12 = tick & 0xFFFU;
        if (!pages_[page_idx]) {
            pages_[page_idx].reset(new (std::nothrow) BitwiseVeb4096());
            if (!pages_[page_idx]) return;
            ++allocated_pages_;
        }
        const bool was_empty = pages_[page_idx]->empty();
        pages_[page_idx]->insert(low12);
        if (was_empty) top_.insert(page_idx);
    }

    // Pre-populates the 4,096-tick leaf page for page_idx (or tick >> 12) so subsequent
    // insert/remove operations within that page execute with zero heap allocations.
    bool reserve_page(std::uint32_t page_idx) noexcept {
        if (page_idx >= 4096U) return false;
        if (!pages_[page_idx]) {
            pages_[page_idx].reset(new (std::nothrow) BitwiseVeb4096());
            if (!pages_[page_idx]) return false;
            ++allocated_pages_;
        }
        return true;
    }

    void remove(std::uint32_t tick) noexcept {
        if (tick >= kUniverse) return;
        const std::uint32_t page_idx = (tick >> 12U) & 0xFFFU;
        const std::uint32_t low12 = tick & 0xFFFU;
        auto* page = pages_[page_idx].get();
        if (!page) return;
        const bool was_non_empty = !page->empty();
        page->remove(low12);
        if (was_non_empty && page->empty()) top_.remove(page_idx);
    }

    bool contains(std::uint32_t tick) const noexcept {
        if (tick >= kUniverse) return false;
        const std::uint32_t page_idx = (tick >> 12U) & 0xFFFU;
        const std::uint32_t low12 = tick & 0xFFFU;
        const auto* page = pages_[page_idx].get();
        return page != nullptr && page->contains(low12);
    }

    std::uint32_t min_tick() const noexcept {
        const std::uint32_t min_page = top_.min_tick();
        if (min_page == kNullTick) return kNullTick;
        const auto* page = pages_[min_page].get();
        if (!page) return kNullTick;
        const std::uint32_t min_low = page->min_tick();
        if (min_low == kNullTick) return kNullTick;
        return (min_page << 12U) | min_low;
    }

    std::uint32_t max_tick() const noexcept {
        const std::uint32_t max_page = top_.max_tick();
        if (max_page == kNullTick) return kNullTick;
        const auto* page = pages_[max_page].get();
        if (!page) return kNullTick;
        const std::uint32_t max_low = page->max_tick();
        if (max_low == kNullTick) return kNullTick;
        return (max_page << 12U) | max_low;
    }

    int find_min() const noexcept {
        const std::uint32_t m = min_tick();
        return (m == kNullTick) ? -1 : static_cast<int>(m);
    }

    int find_max() const noexcept {
        const std::uint32_t m = max_tick();
        return (m == kNullTick) ? -1 : static_cast<int>(m);
    }

    std::uint32_t next_higher(std::uint32_t tick) const noexcept {
        if (tick >= kUniverse - 1U) return kNullTick;
        const std::uint32_t page_idx = (tick >> 12U) & 0xFFFU;
        const std::uint32_t low12 = tick & 0xFFFU;
        if (const auto* page = pages_[page_idx].get()) {
            const std::uint32_t b = page->next_higher(low12);
            if (b != kNullTick) return (page_idx << 12U) | b;
        }
        const std::uint32_t next_page = top_.next_higher(page_idx);
        if (next_page == kNullTick) return kNullTick;
        const auto* np = pages_[next_page].get();
        if (!np) return kNullTick;
        const std::uint32_t next_low = np->min_tick();
        if (next_low == kNullTick) return kNullTick;
        return (next_page << 12U) | next_low;
    }

    std::uint32_t prev_lower(std::uint32_t tick) const noexcept {
        if (tick == 0U) return kNullTick;
        if (tick >= kUniverse) return max_tick();
        const std::uint32_t page_idx = (tick >> 12U) & 0xFFFU;
        const std::uint32_t low12 = tick & 0xFFFU;
        if (const auto* page = pages_[page_idx].get()) {
            const std::uint32_t b = page->prev_lower(low12);
            if (b != kNullTick) return (page_idx << 12U) | b;
        }
        const std::uint32_t prev_page = top_.prev_lower(page_idx);
        if (prev_page == kNullTick) return kNullTick;
        const auto* pp = pages_[prev_page].get();
        if (!pp) return kNullTick;
        const std::uint32_t prev_low = pp->max_tick();
        if (prev_low == kNullTick) return kNullTick;
        return (prev_page << 12U) | prev_low;
    }

    int find_next(std::uint32_t tick) const noexcept {
        const std::uint32_t n = next_higher(tick);
        return (n == kNullTick) ? -1 : static_cast<int>(n);
    }

    int find_prev(std::uint32_t tick) const noexcept {
        const std::uint32_t p = prev_lower(tick);
        return (p == kNullTick) ? -1 : static_cast<int>(p);
    }

private:
    BitwiseVeb4096 top_{};
    std::array<std::unique_ptr<BitwiseVeb4096>, 4096> pages_{};
    std::size_t allocated_pages_ = 0U;
};

} // namespace sim
