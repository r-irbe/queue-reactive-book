// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once
#include "gpu_kernels.hpp"
#include <cmath>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sim {

// Numerical parity and speedup acceptance checks comparing candidate accelerator
// outputs against a reference output. Integer arrays must match exactly;
// floating-point arrays match within combined absolute/relative tolerance with
// explicit NaN/Inf agreement. Acceptance requires parity regardless of speedup.

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


// ============================================================================
// Live CUDA Driver API (sm_89) Rollout & Ozaki Kernel Parity Harness
// ============================================================================

struct GpuKernelParityReport {
    bool gpu_available{false};
    int init_rc{-1};
    std::int32_t rollout_trajectories{0};
    std::uint32_t rollout_mismatches{0};
    std::uint32_t rollout_survived{0};
    std::int32_t ozaki_events{0};
    float ozaki_gpu_loglik{0.0f};
    float ozaki_cpu_loglik{0.0f};
    float ozaki_gpu_score{0.0f};
    float ozaki_cpu_score{0.0f};
    float ozaki_gpu_fisher{0.0f};
    float ozaki_cpu_fisher{0.0f};
    float ozaki_max_rel_err{0.0f};
    bool parity_ok{false};
};

[[nodiscard]] inline GpuKernelParityReport run_gpu_kernel_parity_suite(
    std::int32_t n_trajectories = 4096,
    std::int32_t n_events = 1024,
    std::uint64_t base_seed = 0x9E3779B97F4A7C15ULL) noexcept {
    GpuKernelParityReport rep{};
    rep.rollout_trajectories = n_trajectories;
    rep.ozaki_events = n_events;

    gpu::GpuContext ctx{};
    rep.init_rc = gpu::gpu_init(&ctx);
    if (rep.init_rc != 0) {
        gpu::gpu_shutdown(&ctx);
        return rep;
    }
    rep.gpu_available = true;

    std::vector<std::uint64_t> state_gpu(static_cast<std::size_t>(n_trajectories));
    std::vector<std::uint64_t> state_cpu(static_cast<std::size_t>(n_trajectories));
    std::vector<std::uint64_t> seeds_gpu(static_cast<std::size_t>(n_trajectories));
    std::vector<std::uint64_t> seeds_cpu(static_cast<std::size_t>(n_trajectories));
    std::vector<std::uint32_t> surv_gpu(static_cast<std::size_t>(n_trajectories), 0U);
    std::vector<float> inten_gpu(static_cast<std::size_t>(n_trajectories), 0.0f);

    for (std::int32_t i = 0; i < n_trajectories; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const std::uint64_t s = (static_cast<std::uint64_t>(i) * 40503ULL + 17ULL) & 0xFFFFULL;
        const std::uint64_t r = base_seed ^ (static_cast<std::uint64_t>(i + 1) * 0xBF58476D1CE4E5B9ULL);
        state_gpu[idx] = state_cpu[idx] = s;
        seeds_gpu[idx] = seeds_cpu[idx] = r;
    }

    const int rc_rollout = gpu::gpu_rollout(
        &ctx,
        state_gpu.data(),
        seeds_gpu.data(),
        surv_gpu.data(),
        inten_gpu.data(),
        n_trajectories);

    constexpr float kTwoPowMinus16 = 1.0f / 65536.0f;
    for (std::int32_t i = 0; i < n_trajectories; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const std::uint64_t decay = gpu::q16_horner_decay_step(state_cpu[idx]);
        const std::uint64_t rng_next = gpu::rng_mix_step(seeds_cpu[idx]);
        const float inten_ref = static_cast<float>(decay) * kTwoPowMinus16;
        const std::uint32_t surv_ref =
            ((static_cast<std::uint32_t>(decay) << 15U) >
             static_cast<std::uint32_t>(rng_next & 0x7FFFFFFFULL))
                ? 1U
                : 0U;
        if (state_gpu[idx] != decay ||
            seeds_gpu[idx] != rng_next ||
            surv_gpu[idx] != surv_ref ||
            inten_gpu[idx] != inten_ref) {
            ++rep.rollout_mismatches;
        }
        rep.rollout_survived += surv_gpu[idx];
    }

    std::vector<float> offsets(static_cast<std::size_t>(n_events));
    std::vector<std::uint32_t> channels(static_cast<std::size_t>(n_events));
    std::vector<float> depths(static_cast<std::size_t>(n_events));
    const float mu6[6]    = {1.20f, 1.15f, 0.85f, 0.80f, 0.45f, 0.40f};
    const float gamma6[6] = {0.18f, 0.16f, 0.22f, 0.20f, 0.30f, 0.28f};

    double cpu_ll = 0.0, cpu_sc = 0.0, cpu_fi = 0.0;
    for (std::int32_t i = 0; i < n_events; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        offsets[idx]  = 0.001f * static_cast<float>((i % 17) + 1);
        channels[idx] = static_cast<std::uint32_t>(i % 8);
        depths[idx]   = 0.25f * static_cast<float>((i % 5) + 1);

        std::uint32_t c = channels[idx] & 7U;
        if (c >= 6U) c = 0U;
        const float gq = gamma6[c] * depths[idx];
        const float phi = (c < 2U) ? (1.0f - gq) : (1.0f + gq);
        const float lam = mu6[c] * phi;
        cpu_ll += static_cast<double>(offsets[idx] * lam);
        cpu_sc += static_cast<double>(offsets[idx] * (gamma6[c] * lam));
        cpu_fi += static_cast<double>(offsets[idx] * (lam * lam));
    }

    const int rc_ozaki = gpu::gpu_ozaki(
        &ctx,
        offsets.data(),
        channels.data(),
        depths.data(),
        mu6,
        gamma6,
        &rep.ozaki_gpu_loglik,
        &rep.ozaki_gpu_score,
        &rep.ozaki_gpu_fisher,
        n_events);

    gpu::gpu_shutdown(&ctx);

    rep.ozaki_cpu_loglik = static_cast<float>(cpu_ll);
    rep.ozaki_cpu_score  = static_cast<float>(cpu_sc);
    rep.ozaki_cpu_fisher = static_cast<float>(cpu_fi);

    auto rel_err = [](float g, float c) noexcept -> float {
        const float denom = std::max(std::fabs(c), 1e-6f);
        return std::fabs(g - c) / denom;
    };
    rep.ozaki_max_rel_err = std::max({
        rel_err(rep.ozaki_gpu_loglik, rep.ozaki_cpu_loglik),
        rel_err(rep.ozaki_gpu_score,  rep.ozaki_cpu_score),
        rel_err(rep.ozaki_gpu_fisher, rep.ozaki_cpu_fisher),
    });

    rep.parity_ok = (rc_rollout == 0) &&
                    (rc_ozaki == 0) &&
                    (rep.rollout_mismatches == 0U) &&
                    (rep.ozaki_max_rel_err < 1e-4f);
    return rep;
}

} // namespace sim
