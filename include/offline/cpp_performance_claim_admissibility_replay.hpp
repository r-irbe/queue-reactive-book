#pragma once

// =============================================================================
// Offline C++ performance-claim ADMISSIBILITY replay diagnostic.
//
// SCOPE / BOUNDARY (hard invariant):
//   This is OFFLINE tooling ONLY. It has ZERO dependency on the matching kernel,
//   any order book, the simulator, or any engine/replay state. It is a self-
//   contained, header-only, strictly READ-ONLY diagnostic. Std includes only;
//   <chrono> is used for an ILLUSTRATIVE wall-clock timing helper whose result
//   is NEVER part of any correctness assertion. The deterministic admissibility
//   verdict depends only on integer outputs, never on timing.
//
// WHAT IT PROVES (the discipline, NOT a benchmark result):
//   A C++ performance CLAIM (e.g. "this branch-free bit-twiddle is faster than
//   the naive routine") is ADMISSIBLE as project guidance only if it carries:
//     (a) a deterministic CORRECTNESS ORACLE -- the "optimized" path must match
//         a naive reference BIT-FOR-BIT across an exhaustive / representative
//         integer domain; and
//     (b) a REPLAY-NEUTRALITY guarantee -- substituting the optimized path for
//         the reference leaves the observable output stream byte-identical, so a
//         replay's deterministic hash is unchanged.
//   Speed is a SEPARATE, illustrative axis: a claim that is faster but fails the
//   oracle or perturbs the output stream is INADMISSIBLE. This fixture is the
//   code/replay falsifier for moc-cpp-performance-claim-admissibility: it shows
//   the admissibility checker is constructible and replay-neutral with no engine
//   coupling. It is IMPLEMENTATION EVIDENCE ONLY -- not a proof of theory and not
//   engine adoption.
//
// WORKED EXAMPLE:
//   Four integer transforms taken from the Hacker's Delight bit-twiddling
//   tradition (sign, abs, round-up-to-power-of-2, popcount), each expressed as a
//   naive reference and a branch-free "optimized" form computed in UNSIGNED
//   arithmetic (UB-clean: no signed overflow, no shift on negative operands).
//   The checker proves the optimized forms are bit-for-bit identical to the
//   references over a representative + boundary + exhaustive-16-bit domain, and
//   that their FNV-1a-64 output-stream hashes are equal (replay-neutral). A
//   deliberately broken "optimized" form is included as a negative control: it
//   is detected as INADMISSIBLE (oracle mismatch + hash divergence).
//
// SEMANTIC CONTRACT (determinism guarantee):
//   Every verdict is a pure function of the (immutable) domain and the two
//   transforms under comparison. The same inputs yield a byte-identical verdict
//   and a stable FNV-1a-64 hash, independent of platform, padding, or call
//   order. Nothing is ever written back into any book, engine, or replay state.
// =============================================================================

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace offline {

using i32 = std::int32_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// A transform under test: a pure u32 -> u32 map. Signed transforms reinterpret
// their u32 operand as i32 internally; the uniform signature lets the checker
// treat every claim the same way.
using Transform = u32 (*)(u32) noexcept;

// --- Worked-example transforms (naive reference vs branch-free "optimized") ---
// All optimized forms are computed in UNSIGNED arithmetic so they are UB-clean:
// no signed overflow (e.g. -INT_MIN) and no shifts on negative operands.

// sign(x) -> {-1, 0, +1} reinterpreted as u32. Reference uses comparisons.
constexpr u32 naive_sign(u32 bits) noexcept {
    const i32 x = static_cast<i32>(bits);
    const i32 s = static_cast<i32>(x > 0) - static_cast<i32>(x < 0);
    return static_cast<u32>(s);
}

// Branch-free sign (Hacker's Delight). hi = -1 if negative else 0; the second
// term is 1 iff x > 0. Both terms computed via logical shifts on unsigned.
constexpr u32 opt_sign(u32 bits) noexcept {
    const u32 hi = u32{0} - (bits >> 31);       // 0xFFFFFFFF if negative else 0
    const u32 pos = (u32{0} - bits) >> 31;      // 1 iff x > 0 else 0
    return hi | pos;
}

// abs(x). Reference negates in unsigned (well-defined) to avoid -INT_MIN UB.
constexpr u32 naive_abs(u32 bits) noexcept {
    const i32 x = static_cast<i32>(bits);
    return (x < 0) ? (u32{0} - bits) : bits;
}

// Branch-free abs (Hacker's Delight): mask = -sign_bit; (x ^ mask) - mask.
constexpr u32 opt_abs(u32 bits) noexcept {
    const u32 mask = u32{0} - (bits >> 31); // 0xFFFFFFFF if negative else 0
    return (bits ^ mask) - mask;
}

// Round up to the smallest power of two >= x (clp2). Returns 0 when x == 0 or
// when the result is not representable in 32 bits (x > 2^31).
constexpr u32 naive_clp2(u32 x) noexcept {
    if (x == 0) {
        return 0;
    }
    u64 p = 1;
    while (p < x) {
        p <<= 1;
    }
    return (p > 0xFFFFFFFFull) ? u32{0} : static_cast<u32>(p);
}

// Branch-free clp2 (Hacker's Delight): subtract 1, smear the high bit down,
// add 1. x == 0 smears to all-ones and wraps back to 0; x > 2^31 overflows to 0.
constexpr u32 opt_clp2(u32 x) noexcept {
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// popcount(x) -> number of set bits, as u32. Reference counts in a loop.
constexpr u32 naive_popcount(u32 x) noexcept {
    u32 n = 0;
    while (x != 0) {
        n += (x & 1u);
        x >>= 1;
    }
    return n;
}

// Branch-free popcount (Hacker's Delight parallel reduction).
constexpr u32 opt_popcount(u32 x) noexcept {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3Fu;
}

// Negative control: a deliberately BROKEN "optimized" sign that drops the
// positive case (returns only the sign-extension of the high bit). It MUST be
// detected as inadmissible by the checker.
constexpr u32 broken_opt_sign(u32 bits) noexcept {
    return u32{0} - (bits >> 31); // 0xFFFFFFFF for negative, 0 otherwise; misses +1
}

// --- Compile-time bit-for-bit oracle on representative constants --------------
// These fail to COMPILE if an optimized form ever diverges from its reference on
// a boundary value -- a deterministic, timing-independent correctness oracle.
static_assert(opt_sign(static_cast<u32>(5)) == naive_sign(static_cast<u32>(5)));
static_assert(opt_sign(static_cast<u32>(-5)) == naive_sign(static_cast<u32>(-5)));
static_assert(opt_sign(0) == naive_sign(0));
static_assert(opt_sign(0x80000000u) == naive_sign(0x80000000u)); // INT_MIN
static_assert(opt_sign(0x7FFFFFFFu) == naive_sign(0x7FFFFFFFu)); // INT_MAX
static_assert(opt_abs(0x80000000u) == naive_abs(0x80000000u));   // INT_MIN: UB-clean
static_assert(opt_abs(static_cast<u32>(-7)) == naive_abs(static_cast<u32>(-7)));
static_assert(opt_clp2(0) == naive_clp2(0));
static_assert(opt_clp2(1) == naive_clp2(1));
static_assert(opt_clp2(5) == naive_clp2(5));
static_assert(opt_clp2(0x80000000u) == naive_clp2(0x80000000u));
static_assert(opt_clp2(0x80000001u) == naive_clp2(0x80000001u)); // overflow -> 0
static_assert(opt_popcount(0xFFFFFFFFu) == naive_popcount(0xFFFFFFFFu));
static_assert(opt_popcount(0) == naive_popcount(0));
static_assert(opt_popcount(0xA5A5A5A5u) == naive_popcount(0xA5A5A5A5u));

// Verdict for a single performance claim. Pure copyable value: holds no handle
// to any external (engine/replay) state.
struct AdmissibilityVerdict {
    std::size_t samples_checked = 0;
    std::size_t mismatch_count = 0;
    u64 reference_hash = 0; // FNV-1a-64 over the reference output stream
    u64 optimized_hash = 0; // FNV-1a-64 over the optimized output stream
    bool oracle_matched = false;   // optimized == reference bit-for-bit, all samples
    bool replay_neutral = false;   // output-stream hashes are equal

    // A claim is admissible iff its optimized path matches the deterministic
    // oracle bit-for-bit AND is replay-neutral (output stream unchanged).
    constexpr bool admissible() const {
        return oracle_matched && replay_neutral && mismatch_count == 0;
    }
};

// Illustrative (NON-DETERMINISTIC) timing sample. NEVER asserted on; present
// only to document that timing is the separate, non-load-bearing axis.
struct IllustrativeTiming {
    double reference_ns = 0.0; // wall-clock, illustrative only
    double optimized_ns = 0.0; // wall-clock, illustrative only
};

// Offline admissibility checker. Build-once from an immutable integer domain;
// query-only. Every public method is const: the checker can never mutate engine
// or replay state, and never feeds its outputs back into any book.
class PerformanceClaimAdmissibilityReplay {
public:
    PerformanceClaimAdmissibilityReplay() = default;

    explicit PerformanceClaimAdmissibilityReplay(std::vector<u32> domain)
        : domain_(std::move(domain)) {}

    std::size_t size() const { return domain_.size(); }
    bool empty() const { return domain_.empty(); }
    const std::vector<u32>& domain() const { return domain_; }

    // FNV-1a-64 hash of one transform's output stream over the domain. Pure,
    // const, deterministic: identical (domain, transform) yields identical hash.
    u64 hash_transform(Transform fn) const {
        u64 h = 1469598103934665603ULL; // FNV offset basis
        for (const u32 in : domain_) {
            const u32 out = fn(in);
            mix64(h, static_cast<u64>(in));
            mix64(h, static_cast<u64>(out));
        }
        return h;
    }

    // Adjudicate one performance claim: does `optimized` match `reference`
    // bit-for-bit across the whole domain, and is it replay-neutral (identical
    // output-stream hash)? Pure, const, deterministic. Timing plays NO role.
    AdmissibilityVerdict check_claim(Transform reference, Transform optimized) const {
        AdmissibilityVerdict v;
        v.samples_checked = domain_.size();

        u64 ref_h = 1469598103934665603ULL;
        u64 opt_h = 1469598103934665603ULL;
        std::size_t mismatches = 0;
        for (const u32 in : domain_) {
            const u32 ro = reference(in);
            const u32 oo = optimized(in);
            if (ro != oo) {
                ++mismatches;
            }
            mix64(ref_h, static_cast<u64>(in));
            mix64(ref_h, static_cast<u64>(ro));
            mix64(opt_h, static_cast<u64>(in));
            mix64(opt_h, static_cast<u64>(oo));
        }

        v.mismatch_count = mismatches;
        v.reference_hash = ref_h;
        v.optimized_hash = opt_h;
        v.oracle_matched = (mismatches == 0);
        v.replay_neutral = (ref_h == opt_h);
        return v;
    }

    // ILLUSTRATIVE ONLY. Returns wall-clock nanoseconds for each path. The
    // result is non-deterministic and MUST NOT be asserted on; it exists only
    // to document that speed is the separate, non-load-bearing axis.
    IllustrativeTiming illustrative_timing(Transform reference,
                                           Transform optimized) const {
        using clock = std::chrono::steady_clock;
        IllustrativeTiming t;

        volatile u32 sink = 0;
        const auto t0 = clock::now();
        for (const u32 in : domain_) {
            sink ^= reference(in);
        }
        const auto t1 = clock::now();
        for (const u32 in : domain_) {
            sink ^= optimized(in);
        }
        const auto t2 = clock::now();
        (void)sink;

        t.reference_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count();
        t.optimized_ns =
            std::chrono::duration<double, std::nano>(t2 - t1).count();
        return t;
    }

private:
    static void mix64(u64& h, u64 v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v & 0xFFull);
            h *= 1099511628211ULL; // FNV prime
            v >>= 8;
        }
    }

    std::vector<u32> domain_;
};

// --- Compile-time read-only / non-feedback guarantees ------------------------
// These static_asserts are part of the falsifier: they fail to COMPILE if the
// checker ever gains a mutating path or stops being a pure value-producing
// projection (which is how engine/replay-state feedback would manifest).

// check_claim() must be invocable on a CONST checker (read-only diagnostic).
static_assert(
    std::is_invocable_r_v<AdmissibilityVerdict,
                          decltype(&PerformanceClaimAdmissibilityReplay::check_claim),
                          const PerformanceClaimAdmissibilityReplay&, Transform,
                          Transform>,
    "check_claim() must be a read-only (const) diagnostic");

// hash_transform() must be invocable on a CONST checker.
static_assert(
    std::is_invocable_r_v<u64,
                          decltype(&PerformanceClaimAdmissibilityReplay::hash_transform),
                          const PerformanceClaimAdmissibilityReplay&, Transform>,
    "hash_transform() must be a read-only (const) diagnostic");

// The verdict is a self-contained, copyable value: it holds no handle to any
// external (engine/replay) state that could be written back.
static_assert(std::is_copy_constructible_v<AdmissibilityVerdict>,
              "AdmissibilityVerdict must be a pure copyable value type");
static_assert(std::is_trivially_copyable_v<AdmissibilityVerdict>,
              "AdmissibilityVerdict must be a trivially copyable value type");

} // namespace offline
