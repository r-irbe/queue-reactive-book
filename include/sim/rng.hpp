#pragma once

#include "random/xoshiro256starstar.hpp"
#include "types.hpp"

#include <cstdint>
#include <limits>

namespace sim {

// Direct type alias: Rng IS Xoshiro256StarStar. No wrapper overhead.
// Use directly or wrap in RngState if stateful context needed.
using Rng = Xoshiro256StarStar;

// Distribution utilities (namespace free; compose as needed).
namespace distributions {

// Uniform double in [0, 1) with 53 bits of mantissa precision.
// Algorithm: (x >> kDoubleShift) / 2^53, where x is 64-bit uniform.
inline double next_unit_double(Rng& engine) noexcept {
    constexpr double denom = static_cast<double>(1ULL << rng_constants::kDoubleMantissa);
    const std::uint64_t mantissa = engine.next_u64() >> rng_constants::kDoubleShift;
    return static_cast<double>(mantissa) / denom;
}

// Uniform integer in [0, upper_exclusive) using rejection sampling.
// Rejects values >= (2^64 - (2^64 mod upper_exclusive)) to ensure uniform distribution.
inline std::uint64_t next_uniform_u64(Rng& engine, std::uint64_t upper_exclusive) noexcept {
    if (upper_exclusive == 0) return 0;
    
    const std::uint64_t rejection_threshold =
        (std::numeric_limits<std::uint64_t>::max() - upper_exclusive + 1ULL) % upper_exclusive;
    
    while (true) {
        const std::uint64_t candidate = engine.next_u64();
        if (candidate >= rejection_threshold) {
            return candidate % upper_exclusive;
        }
    }
}

} // namespace distributions

} // namespace sim
