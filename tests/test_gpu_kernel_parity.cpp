// Hardware parity tests for the CUDA Driver API PTX/cubin kernels in
// `sim/gpu_kernels.hpp`.
//
// Each test runs the actual sm_89 PTX/cubin kernel on the installed CUDA
// device and compares its output against a bit-accurate CPU reference
// implementation of the same specification. When no CUDA driver/device is
// available, tests skip ONLY if `QRLOB_REQUIRE_GPU` is unset; setting
// `QRLOB_REQUIRE_GPU=1` turns any missing-driver condition into a hard test
// failure.

#include <catch2/catch_test_macros.hpp>

#include "sim/gpu_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool gpu_required() noexcept {
    const char* v = std::getenv("QRLOB_REQUIRE_GPU");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

struct ScopedGpuContext {
    sim::gpu::GpuContext ctx{};
    int init_rc = -1;
    ScopedGpuContext() noexcept { init_rc = sim::gpu::gpu_init(&ctx); }
    ~ScopedGpuContext() noexcept { sim::gpu::gpu_shutdown(&ctx); }
};

// CPU reference matching qrlob_gpu_batch_qr_hawkes_rollout_kernel:
//   q          = state & 0xFFFF
//   decay      = 65536 - q + ((q * q) >> 17)
//   rng_next   = seed ^ (seed >> 7)
//   intensity  = float(decay) * 2^-16
//   survival   = ((uint32_t(decay) << 15) > uint32_t(rng_next & 0x7FFFFFFFULL)) ? 1u : 0u
void rollout_cpu_reference(
    std::uint64_t* state,
    std::uint64_t* seeds,
    std::uint32_t* out_survival,
    float* out_intensity,
    std::int32_t n) noexcept {
    constexpr float kTwoPowMinus16 = 1.0f / 65536.0f;
    for (std::int32_t i = 0; i < n; ++i) {
        const std::uint64_t s = state[i];
        const std::uint64_t q = s & 0xFFFFULL;
        const std::uint64_t decay = sim::gpu::q16_horner_decay_step(s);
        state[i] = decay;

        const std::uint64_t seed = seeds[i];
        const std::uint64_t rng_next = sim::gpu::rng_mix_step(seed);
        seeds[i] = rng_next;

        out_intensity[i] = static_cast<float>(decay) * kTwoPowMinus16;
        const std::uint32_t lhs = static_cast<std::uint32_t>(decay) << 15U;
        const std::uint32_t rhs = static_cast<std::uint32_t>(rng_next & 0x7FFFFFFFULL);
        out_survival[i] = (lhs > rhs) ? 1U : 0U;
        (void)q;
    }
}

// CPU reference matching qrlob_gpu_ozaki_score_fisher_kernel:
//   c      = (raw_c & 7) >= 6 ? 0 : (raw_c & 7)
//   phi    = (c < 2) ? (1.0f - gamma[c] * q_tilde) : (1.0f + gamma[c] * q_tilde)
//   lam    = mu[c] * phi
//   ll_i   = dt * lam
//   sc_i   = dt * (gamma[c] * lam)
//   fi_i   = dt * (lam * lam)
void ozaki_cpu_reference(
    const float* offsets,
    const std::uint32_t* channels,
    const float* depths,
    const float* mu6,
    const float* gamma6,
    double& out_ll,
    double& out_sc,
    double& out_fi,
    std::int32_t n) noexcept {
    out_ll = 0.0;
    out_sc = 0.0;
    out_fi = 0.0;
    for (std::int32_t i = 0; i < n; ++i) {
        std::uint32_t c = channels[i] & 7U;
        if (c >= 6U) c = 0U;
        const float q = depths[i];
        const float dt = offsets[i];
        const float mu = mu6[c];
        const float gam = gamma6[c];
        const float gq = gam * q;
        const float phi = (c < 2U) ? (1.0f - gq) : (1.0f + gq);
        const float f10 = mu * phi;
        const float f11 = dt * f10;
        const float f12 = gam * f10;
        const float f13 = dt * f12;
        const float f14 = f10 * f10;
        const float f15 = dt * f14;
        out_ll += static_cast<double>(f11);
        out_sc += static_cast<double>(f13);
        out_fi += static_cast<double>(f15);
    }
}

} // namespace

TEST_CASE("GPU kernel header embeds valid PTX and sm_89 cubin magic",
          "[gpu_parity][qrlob]") {
    REQUIRE(sim::gpu::kRolloutKernelPtxLen > 512U);
    REQUIRE(sim::gpu::kOzakiScoreFisherPtxLen > 512U);
    REQUIRE(std::strstr(sim::gpu::kRolloutKernelPtx,
                        ".visible .entry qrlob_gpu_batch_qr_hawkes_rollout_kernel") != nullptr);
    REQUIRE(std::strstr(sim::gpu::kOzakiScoreFisherPtx,
                        ".visible .entry qrlob_gpu_ozaki_score_fisher_kernel") != nullptr);

    // ELF magic 0x7F 'E' 'L' 'F' on both pre-compiled sm_89 cubin blobs.
    REQUIRE(sim::gpu::kMarketSubgameCubinLen > 1024U);
    REQUIRE(sim::gpu::kMarketSubgameCubin[0] == 0x7FU);
    REQUIRE(sim::gpu::kMarketSubgameCubin[1] == 'E');
    REQUIRE(sim::gpu::kMarketSubgameCubin[2] == 'L');
    REQUIRE(sim::gpu::kMarketSubgameCubin[3] == 'F');

    REQUIRE(sim::gpu::kOnlineRegretSubgameCubinLen > 1024U);
    REQUIRE(sim::gpu::kOnlineRegretSubgameCubin[0] == 0x7FU);
    REQUIRE(sim::gpu::kOnlineRegretSubgameCubin[1] == 'E');
    REQUIRE(sim::gpu::kOnlineRegretSubgameCubin[2] == 'L');
    REQUIRE(sim::gpu::kOnlineRegretSubgameCubin[3] == 'F');

    // Verify stateless counter RNG mix, SWAR nibble LUT, and GLFT/CJP quote helpers
    REQUIRE(sim::gpu::stateless_counter_rng_mix(0x12345678ULL, 0ULL) ==
            sim::gpu::rng_mix_step(0x12345678ULL));
    for (std::uint32_t r = 0; r < 16U; ++r) {
        REQUIRE(sim::gpu::extract_swar_nibble(sim::gpu::kSwarAddLutDefault, r) < 16U);
        REQUIRE(sim::gpu::extract_swar_nibble(sim::gpu::kSwarCancelLutDefault, r) < 16U);
        REQUIRE(sim::gpu::extract_swar_nibble(sim::gpu::kSwarExecLutDefault, r) < 16U);
    }
    const auto quotes = sim::gpu::evaluate_glft_cjp_quotes_q16(
        100LL << 16, 4LL << 16, 2048LL, 1LL << 16, 1024LL, 32768LL);
    REQUIRE(quotes.ask_q16 - quotes.bid_q16 == 65536LL);
}

TEST_CASE("qrlob_gpu_batch_qr_hawkes_rollout_kernel bit-exact GPU vs CPU parity",
          "[gpu_parity][qrlob]") {
    ScopedGpuContext gpu;
    if (gpu.init_rc != 0) {
        if (gpu_required()) {
            REQUIRE(gpu.init_rc == 0);
        }
        return;
    }

    constexpr std::int32_t kN = 4096;
    std::vector<std::uint64_t> state_gpu(kN), state_cpu(kN);
    std::vector<std::uint64_t> seeds_gpu(kN), seeds_cpu(kN);
    std::vector<std::uint32_t> surv_gpu(kN, 0U), surv_cpu(kN, 0U);
    std::vector<float> inten_gpu(kN, 0.0f), inten_cpu(kN, 0.0f);

    for (std::int32_t i = 0; i < kN; ++i) {
        const std::uint64_t s = static_cast<std::uint64_t>((i * 40503ULL + 17ULL) & 0xFFFFULL);
        const std::uint64_t r = 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(i + 1) * 0xBF58476D1CE4E5B9ULL);
        state_gpu[i] = state_cpu[i] = s;
        seeds_gpu[i] = seeds_cpu[i] = r;
    }

    REQUIRE(sim::gpu::gpu_rollout(
                &gpu.ctx,
                state_gpu.data(),
                seeds_gpu.data(),
                surv_gpu.data(),
                inten_gpu.data(),
                kN) == 0);

    rollout_cpu_reference(
        state_cpu.data(),
        seeds_cpu.data(),
        surv_cpu.data(),
        inten_cpu.data(),
        kN);

    std::uint32_t survived_total = 0;
    for (std::int32_t i = 0; i < kN; ++i) {
        REQUIRE(state_gpu[i] == state_cpu[i]);
        REQUIRE(state_gpu[i] >= 32768ULL);
        REQUIRE(state_gpu[i] <= 65536ULL);
        REQUIRE(seeds_gpu[i] == seeds_cpu[i]);
        REQUIRE(surv_gpu[i] == surv_cpu[i]);
        REQUIRE(inten_gpu[i] == inten_cpu[i]);
        REQUIRE(inten_gpu[i] >= 0.5f);
        REQUIRE(inten_gpu[i] <= 1.0f);
        survived_total += surv_gpu[i];
    }
    REQUIRE(survived_total > 1000U);
    REQUIRE(survived_total < static_cast<std::uint32_t>(kN));
}

TEST_CASE("qrlob_gpu_ozaki_score_fisher_kernel warp reduction GPU vs CPU parity",
          "[gpu_parity][qrlob]") {
    ScopedGpuContext gpu;
    if (gpu.init_rc != 0) {
        if (gpu_required()) {
            REQUIRE(gpu.init_rc == 0);
        }
        return;
    }

    for (std::int32_t kN : {1024, 1000, 131}) {
        std::vector<float> offsets(kN);
        std::vector<std::uint32_t> channels(kN);
        std::vector<float> depths(kN);
        const float mu6[6]    = {1.20f, 1.15f, 0.85f, 0.80f, 0.45f, 0.40f};
        const float gamma6[6] = {0.18f, 0.16f, 0.22f, 0.20f, 0.30f, 0.28f};

        for (std::int32_t i = 0; i < kN; ++i) {
            offsets[i]  = 0.001f * static_cast<float>((i % 17) + 1);
            channels[i] = static_cast<std::uint32_t>(i % 8);
            depths[i]   = 0.25f * static_cast<float>((i % 5) + 1);
        }

        float gpu_ll = 0.0f, gpu_sc = 0.0f, gpu_fi = 0.0f;
        REQUIRE(sim::gpu::gpu_ozaki(
                    &gpu.ctx,
                    offsets.data(),
                    channels.data(),
                    depths.data(),
                    mu6,
                    gamma6,
                    &gpu_ll,
                    &gpu_sc,
                    &gpu_fi,
                    kN) == 0);

        double cpu_ll = 0.0, cpu_sc = 0.0, cpu_fi = 0.0;
        ozaki_cpu_reference(
            offsets.data(),
            channels.data(),
            depths.data(),
            mu6,
            gamma6,
            cpu_ll,
            cpu_sc,
            cpu_fi,
            kN);

        REQUIRE(std::fabs(static_cast<double>(gpu_ll) - cpu_ll) <= 1e-4 * std::max(1.0, std::fabs(cpu_ll)));
        REQUIRE(std::fabs(static_cast<double>(gpu_sc) - cpu_sc) <= 1e-4 * std::max(1.0, std::fabs(cpu_sc)));
        REQUIRE(std::fabs(static_cast<double>(gpu_fi) - cpu_fi) <= 1e-4 * std::max(1.0, std::fabs(cpu_fi)));
    }
}
