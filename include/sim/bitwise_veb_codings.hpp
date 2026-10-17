#pragma once
// Lean-Aligned Bitwise van Emde Boas (vEB) Templates, Cross-ISA MSB-First Bitboard,
// Boustrophedon Reflection, Offset-Binary, Knuth Gray Subset Basket, and Elias-Fano Coding.
// Aligned with Lean 4 module: Rules.BitwiseVebWarpCodec
// Strict 7-bit ASCII compliance (INV-001).

#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
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

// 7. Elias-Fano Quasi-Succinct Monotonic Sequence Index (Tier 1.5 Bridge)
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

} // namespace sim
