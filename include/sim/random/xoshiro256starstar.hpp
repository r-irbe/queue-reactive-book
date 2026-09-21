#pragma once

// xoshiro256** reference algorithm by David Blackman and Sebastiano Vigna.
// Source family and jump constants are documented at:
//   https://prng.di.unimi.it/
// This is a local, header-only adaptation for deterministic simulation use.

#include "../types.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace sim {

class Xoshiro256StarStar {
public:
    using result_type = std::uint64_t;

    explicit Xoshiro256StarStar(std::uint64_t seed = rng_constants::kDefaultSeed) noexcept {
        seed_with(seed);
    }

    void seed_with(std::uint64_t seed) noexcept {
        SplitMix64 sm(seed);
        for (auto& word : state_) {
            word = sm.next();
        }
        if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0) {
            state_[0] = 0x9e3779b97f4a7c15ULL;
        }
    }

    result_type operator()() noexcept { return next_u64(); }

    result_type next_u64() noexcept {
        const std::uint64_t result = rotl(state_[1] * 5ULL, 7) * 9ULL;
        const std::uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);

        return result;
    }

    std::uint32_t next_u32() noexcept {
        return static_cast<std::uint32_t>(next_u64() >> 32);
    }

    static constexpr result_type min() noexcept {
        return std::numeric_limits<result_type>::min();
    }

    static constexpr result_type max() noexcept {
        return std::numeric_limits<result_type>::max();
    }

    void jump() noexcept {
        static constexpr std::array<std::uint64_t, 4> kJump = {
            0x180ec6d33cfd0abaULL,
            0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL,
            0x39abdc4529b1661cULL,
        };
        apply_jump_sequence(kJump);
    }

    void long_jump() noexcept {
        static constexpr std::array<std::uint64_t, 4> kLongJump = {
            0x76e15d3efefdcbbfULL,
            0xc5004e441c522fb3ULL,
            0x77710069854ee241ULL,
            0x39109bb02acbe635ULL,
        };
        apply_jump_sequence(kLongJump);
    }

private:
    void apply_jump_sequence(const std::array<std::uint64_t, 4>& jump_constants) noexcept {
        std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        
        for (std::uint64_t jc : jump_constants) {
            for (int bit = 0; bit < 64; ++bit) {
                if ((jc & (1ULL << bit)) != 0) {
                    s0 ^= state_[0];
                    s1 ^= state_[1];
                    s2 ^= state_[2];
                    s3 ^= state_[3];
                }
                (void)next_u64();
            }
        }
        
        state_[0] = s0;
        state_[1] = s1;
        state_[2] = s2;
        state_[3] = s3;
    }
    struct SplitMix64 {
        std::uint64_t state;

        explicit SplitMix64(std::uint64_t seed) noexcept : state(seed) {}

        std::uint64_t next() noexcept {
            std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        }
    };

    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    std::array<std::uint64_t, 4> state_{};
};

} // namespace sim
