// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// CPU/GPU parity differential for the Lean-4-verified sm_89 QRLOB kernels
// (rust-mmap-engine lean_rules, Rules.QrlobGpu; see
// docs/39_QRLOB_LEAN_GPU_INTEGRATION_BLUEPRINT.md Area 1c/1d).
//
// Acceptance contract (sim::accelerator_parity.hpp):
//   - integer outputs (survival indicators): exact match required;
//   - float outputs (intensity): sim::ParityTolerance acceptance;
//   - statistical sums (Ozaki log-likelihood/score/fisher): relaxed tolerance,
//     because GPU accumulation order (warp butterfly + atomics) differs from a
//     serial host loop;
//   - libcuda.so.1 absent => kNoDriver => skip (CPU path is the reference and
//     remains authoritative; GPU absence is pass-through, not failure).
//
// Warp-uniformity note: n is chosen as a multiple of the 256-thread block so
// the kernels' full-mask butterflies never execute on partial warps.

#include "catch2/catch_test_macros.hpp"
#define QRLOB_NO_GLOBAL_NEW_OVERRIDE 1
#include "qr/alloc_guard.hpp"

#include "sim/accelerator_parity.hpp"
#include "sim/verified_lean_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// Deterministic host-side generator (xorshift64), so CPU and GPU consume the
// exact same inputs on every run.
struct Xorshift64 {
    std::uint64_t s;
    explicit Xorshift64(std::uint64_t seed) noexcept : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    std::uint64_t next() noexcept {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// Exact host mirror of the emitted rollout kernel step (Rules.QrlobGpu):
//   q      = state & 0xFFFF
//   decay  = 65536 - q + ((q*q) >>> 17)          (verified hawkesDecayQ16)
//   seed'  = seed ^ (seed >>> 7)
//   ipart  = decay >>> 16
//   surv   = ipart > (seed' & 0x7FFFFFFF) ? 1 : 0
//   intens = (float) ipart
void rollout_cpu_reference(const std::uint64_t* states, const std::uint64_t* seeds,
                           std::uint32_t* survival, float* intensity, std::int32_t n) noexcept {
    for (std::int32_t i = 0; i < n; ++i) {
        const std::uint64_t q = states[i] & 0xFFFFull;
        const std::uint64_t decay = 65536ull - q + ((q * q) >> 17);
        const std::uint64_t seed2 = seeds[i] ^ (seeds[i] >> 7);
        const std::uint32_t ipart = static_cast<std::uint32_t>(decay >> 16);
        const std::uint64_t thr = seed2 & 0x7FFFFFFFull;
        survival[i] = ipart > thr ? 1u : 0u;
        intensity[i] = static_cast<float>(ipart);
    }
}

// Host mirror of the emitted 6-channel Ozaki sufficient statistics:
//   c     = clamp(channel & 7, 6 -> 0)
//   phi   = c < 2 ? 1 - gamma[c]*q : 1 + gamma[c]*q
//   ll   += dt * mu[c] * phi
//   sc   += dt * gamma[c] * mu[c] * phi
//   fi   += dt * (mu[c] * phi)^2
void ozaki_cpu_reference(const float* offsets, const std::uint32_t* channels,
                         const float* depths, const float* mu, const float* gamma,
                         float* ll, float* sc, float* fi, std::int32_t n) noexcept {
    double acc_ll = 0.0, acc_sc = 0.0, acc_fi = 0.0;
    for (std::int32_t i = 0; i < n; ++i) {
        std::uint32_t c = channels[i] & 7u;
        if (c >= 6u) c = 0u;
        const float q = depths[i];
        const float dt = offsets[i];
        const float m = mu[c];
        const float g = gamma[c];
        const float phi = (c < 2u) ? 1.0f - g * q : 1.0f + g * q;
        const float mph = m * phi;
        acc_ll += static_cast<double>(dt * mph);
        acc_sc += static_cast<double>(dt * g * mph);
        acc_fi += static_cast<double>(dt * (mph * mph));
    }
    *ll = static_cast<float>(acc_ll);
    *sc = static_cast<float>(acc_sc);
    *fi = static_cast<float>(acc_fi);
}

}  // namespace

TEST_CASE("Lean-verified qrlob rollout kernel CPU/GPU parity", "[lean_gpu][qrlob]") {
    sim::lean::LeanGpu gpu;
    const sim::lean::GpuStatus init = sim::lean::lean_gpu_init(&gpu);
    if (init != sim::lean::GpuStatus::kOk) {
        std::printf("[skip] lean gpu parity: %s\n", sim::lean::gpu_status_string(init));
        return;  // pass-through: CPU path stays authoritative without a driver
    }

    constexpr std::int32_t kN = 8192;  // multiple of the 256-thread block
    std::vector<std::uint64_t> states(static_cast<std::size_t>(kN));
    std::vector<std::uint64_t> seeds(static_cast<std::size_t>(kN));
    Xorshift64 rng(0xDEADBEEF12345678ull);
    for (std::int32_t i = 0; i < kN; ++i) {
        // State: Q48.16 words with in-range integer and fractional parts.
        states[static_cast<std::size_t>(i)] = (rng.next() & 0x0000FFFFull) << 16 |
                                              (rng.next() & 0xFFFFull);
        seeds[static_cast<std::size_t>(i)] = rng.next();
    }

    std::vector<std::uint32_t> cpu_survival(static_cast<std::size_t>(kN), 0);
    std::vector<float> cpu_intensity(static_cast<std::size_t>(kN), 0.0f);
    rollout_cpu_reference(states.data(), seeds.data(), cpu_survival.data(),
                          cpu_intensity.data(), kN);

    std::vector<std::uint32_t> gpu_survival(static_cast<std::size_t>(kN), 0);
    std::vector<float> gpu_intensity(static_cast<std::size_t>(kN), 0.0f);
    const sim::lean::GpuStatus st = sim::lean::lean_gpu_rollout(
        &gpu, states.data(), seeds.data(), gpu_survival.data(), gpu_intensity.data(), kN);
    REQUIRE(st == sim::lean::GpuStatus::kOk);
    if (st != sim::lean::GpuStatus::kOk) {
        sim::lean::lean_gpu_shutdown(&gpu);
        return;
    }

    // Integer outputs: exact parity (survival indicators).
    std::vector<std::int64_t> ref_wide(static_cast<std::size_t>(kN));
    std::vector<std::int64_t> cand_wide(static_cast<std::size_t>(kN));
    for (std::int32_t i = 0; i < kN; ++i) {
        ref_wide[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>(cpu_survival[static_cast<std::size_t>(i)]);
        cand_wide[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>(gpu_survival[static_cast<std::size_t>(i)]);
    }
    const sim::ParityResult surv = sim::check_parity_int(
        ref_wide.data(), ref_wide.size(), cand_wide.data(), cand_wide.size());
    REQUIRE(surv.accelerator_acceptable);

    // Float outputs: intensity with the standard acceptance tolerance.
    std::vector<double> ref_d(static_cast<std::size_t>(kN));
    std::vector<double> cand_d(static_cast<std::size_t>(kN));
    for (std::int32_t i = 0; i < kN; ++i) {
        ref_d[static_cast<std::size_t>(i)] =
            static_cast<double>(cpu_intensity[static_cast<std::size_t>(i)]);
        cand_d[static_cast<std::size_t>(i)] =
            static_cast<double>(gpu_intensity[static_cast<std::size_t>(i)]);
    }
    const sim::ParityResult inten = sim::check_parity_double(
        ref_d.data(), ref_d.size(), cand_d.data(), cand_d.size());
    REQUIRE(inten.accelerator_acceptable);

    REQUIRE(sim::lean::lean_gpu_shutdown(&gpu) == sim::lean::GpuStatus::kOk);
}

TEST_CASE("Lean-verified qrlob ozaki score/fisher kernel CPU/GPU parity", "[lean_gpu][qrlob]") {
    sim::lean::LeanGpu gpu;
    const sim::lean::GpuStatus init = sim::lean::lean_gpu_init(&gpu);
    if (init != sim::lean::GpuStatus::kOk) {
        std::printf("[skip] lean gpu parity: %s\n", sim::lean::gpu_status_string(init));
        return;
    }

    constexpr std::int32_t kN = 8192;  // block multiple for warp uniformity
    std::vector<float> offsets(static_cast<std::size_t>(kN));
    std::vector<std::uint32_t> channels(static_cast<std::size_t>(kN));
    std::vector<float> depths(static_cast<std::size_t>(kN));
    Xorshift64 rng(0x0BADC0DECAFEBABEull);
    for (std::int32_t i = 0; i < kN; ++i) {
        offsets[static_cast<std::size_t>(i)] =
            0.001f + static_cast<float>(rng.next() % 1000ull) / 1000.0f;
        // Exercise all 8 raw codes: 6 valid channels plus the 2 clamped ones.
        channels[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(rng.next() % 8ull);
        depths[static_cast<std::size_t>(i)] =
            static_cast<float>(rng.next() % 65536ull) / 65536.0f;
    }
    const float mu[6] = {0.8f, 0.6f, 1.1f, 1.3f, 0.4f, 0.9f};
    const float gamma[6] = {0.05f, 0.10f, -0.02f, 0.15f, 0.00f, 0.07f};

    float cpu_ll = 0.0f, cpu_sc = 0.0f, cpu_fi = 0.0f;
    ozaki_cpu_reference(offsets.data(), channels.data(), depths.data(), mu, gamma,
                        &cpu_ll, &cpu_sc, &cpu_fi, kN);

    float gpu_ll = 0.0f, gpu_sc = 0.0f, gpu_fi = 0.0f;
    const sim::lean::GpuStatus st = sim::lean::lean_gpu_ozaki(
        &gpu, offsets.data(), channels.data(), depths.data(), mu, gamma,
        &gpu_ll, &gpu_sc, &gpu_fi, kN);
    REQUIRE(st == sim::lean::GpuStatus::kOk);
    if (st != sim::lean::GpuStatus::kOk) {
        sim::lean::lean_gpu_shutdown(&gpu);
        return;
    }

    // Statistical sums: relaxed tolerance for reordered float accumulation.
    const sim::ParityTolerance tol{1e-3, 1e-3};
    const double ref_ll = static_cast<double>(cpu_ll);
    const double ref_sc = static_cast<double>(cpu_sc);
    const double ref_fi = static_cast<double>(cpu_fi);
    const double c_ll = static_cast<double>(gpu_ll);
    const double c_sc = static_cast<double>(gpu_sc);
    const double c_fi = static_cast<double>(gpu_fi);
    const sim::ParityResult p_ll = sim::check_parity_double(&ref_ll, 1, &c_ll, 1, tol);
    const sim::ParityResult p_sc = sim::check_parity_double(&ref_sc, 1, &c_sc, 1, tol);
    const sim::ParityResult p_fi = sim::check_parity_double(&ref_fi, 1, &c_fi, 1, tol);
    REQUIRE(p_ll.accelerator_acceptable);
    REQUIRE(p_sc.accelerator_acceptable);
    REQUIRE(p_fi.accelerator_acceptable);

    REQUIRE(sim::lean::lean_gpu_shutdown(&gpu) == sim::lean::GpuStatus::kOk);
}
