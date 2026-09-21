#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sim {

// --- Offline accelerator PARITY + benchmark gate (Tier-3 item 16) --------------
//
// Roadmap item 16 (offline CUDA/cache accelerators): "parity + benchmark FIRST".
// This scaffold is the acceptance gate, not an accelerator: it compares a trusted
// reference output against a candidate accelerator output and decides
// acceptability. The HARD INVARIANT is that acceptance is gated on PARITY ALONE
// -- a faster-but-wrong accelerator is always rejected, no matter the speedup.
// Header-only, cold-path; never feeds the frozen matching kernel.
//
// Integer outputs must match EXACTLY. Double outputs match within a combined
// absolute + relative tolerance. Non-finite semantics are explicit: two NaNs
// agree, same-signed infinities agree, a finite-vs-non-finite pair is a
// mismatch. A size mismatch fails parity immediately (no out-of-bounds compare).

struct ParityTolerance {
    double abs_tol = 1e-9;
    double rel_tol = 1e-9;
};

inline bool parity_doubles_match(double ref, double cand,
                                 const ParityTolerance& tol) noexcept {
    const bool ref_nan = std::isnan(ref), cand_nan = std::isnan(cand);
    if (ref_nan || cand_nan) return ref_nan && cand_nan;        // both NaN agree
    const bool ref_inf = std::isinf(ref), cand_inf = std::isinf(cand);
    if (ref_inf || cand_inf) return ref == cand;                // same-signed inf
    const double diff = std::fabs(ref - cand);
    if (diff <= tol.abs_tol) return true;
    const double a = std::fabs(ref), b = std::fabs(cand);
    const double scale = a > b ? a : b;
    return diff <= tol.rel_tol * scale;
}

struct ParityResult {
    bool size_match = false;
    bool parity_ok = false;
    std::size_t compared = 0;
    std::size_t first_mismatch = 0;  // valid only when has_mismatch
    bool has_mismatch = false;
    bool accelerator_acceptable = false;  // == (size_match && parity_ok)
};

inline ParityResult check_parity_int(const std::int64_t* ref, std::size_t nref,
                                     const std::int64_t* cand, std::size_t ncand) noexcept {
    ParityResult r{};
    if (nref != ncand) return r;  // size_match stays false -> not acceptable
    r.size_match = true;
    for (std::size_t i = 0; i < nref; ++i) {
        ++r.compared;
        if (ref[i] != cand[i]) {
            r.has_mismatch = true;
            r.first_mismatch = i;
            break;
        }
    }
    r.parity_ok = !r.has_mismatch;
    r.accelerator_acceptable = r.size_match && r.parity_ok;
    return r;
}

inline ParityResult check_parity_double(const double* ref, std::size_t nref,
                                        const double* cand, std::size_t ncand,
                                        const ParityTolerance& tol = {}) noexcept {
    ParityResult r{};
    if (nref != ncand) return r;
    r.size_match = true;
    for (std::size_t i = 0; i < nref; ++i) {
        ++r.compared;
        if (!parity_doubles_match(ref[i], cand[i], tol)) {
            r.has_mismatch = true;
            r.first_mismatch = i;
            break;
        }
    }
    r.parity_ok = !r.has_mismatch;
    r.accelerator_acceptable = r.size_match && r.parity_ok;
    return r;
}

// Benchmark verdict: records timing/speedup but acceptance is parity-gated. A
// huge speedup NEVER promotes a parity failure to accepted.
struct AcceleratorVerdict {
    bool parity_acceptable = false;
    double ref_ns = 0.0;
    double cand_ns = 0.0;
    double speedup = 0.0;  // ref_ns / cand_ns (0 if cand timing invalid)
    bool accepted = false; // == parity_acceptable
};

inline AcceleratorVerdict accelerator_verdict(const ParityResult& parity,
                                              double ref_ns, double cand_ns) noexcept {
    AcceleratorVerdict v{};
    v.parity_acceptable = parity.accelerator_acceptable;
    v.ref_ns = ref_ns;
    v.cand_ns = cand_ns;
    v.speedup = (std::isfinite(cand_ns) && cand_ns > 0.0 && std::isfinite(ref_ns))
                    ? ref_ns / cand_ns
                    : 0.0;
    v.accepted = parity.accelerator_acceptable;  // parity gates acceptance
    return v;
}

} // namespace sim
