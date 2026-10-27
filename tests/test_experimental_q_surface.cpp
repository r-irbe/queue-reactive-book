// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// Experimental surface tests: SPSC shared-memory ring, offline Hawkes/VPIN
// replay, and flat-ladder parity harness.

#if !defined(QRLOB_BUILD_EXPERIMENTAL) || QRLOB_BUILD_EXPERIMENTAL

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "flat_ladder_parity_harness.hpp"
#include "offline_hawkes_calibrator.hpp"
#include "spsc_ring_bridge.hpp"
#include "sim/qrlob_ffi.h"

#include <random>
#include <vector>

namespace {
struct KernelFeedbackControl {
    void apply_to_matching() {}
    static constexpr bool is_matching_kernel_input = true;
};
} // namespace

TEST_CASE("experimental SPSC ring bridge: layout, producer, consumer, backpressure",
          "[experimental][spsc_ring_bridge]") {
    static_assert(sizeof(sim::experimental::ShmIpcEvent) == 64, "Event size mismatch");
    static_assert(sizeof(sim::experimental::ShmIpcHeader) == 128, "Header size mismatch");
    static_assert(sizeof(sim::experimental::ShmRingChannel) ==
                      128 + 128 + sim::experimental::kRingCapacity * 64,
                  "Ring size mismatch");

    const char* test_path = "/dev/shm/qrlob_exp_spsc_test";
    sim::experimental::ShmSpscEndpoint endpoint;
    REQUIRE(endpoint.open(test_path, true));
    REQUIRE(endpoint.is_valid());

    sim::experimental::ShmIpcHeader* hdr = endpoint.header();
    REQUIRE(hdr->magic == sim::experimental::kShmMagic);
    REQUIRE(hdr->capacity == sim::experimental::kRingCapacity);
    REQUIRE(hdr->version == sim::experimental::kShmVersion);

    sim::experimental::ShmSpscProducer producer(endpoint.request_channel());
    sim::experimental::ShmSpscConsumer consumer(endpoint.request_channel());
    REQUIRE(producer.is_bound());
    REQUIRE(consumer.is_bound());

    // Empty ring underflow returns false
    sim::experimental::ShmIpcEvent out{};
    REQUIRE_FALSE(consumer.try_dequeue(out));

    // Round-trip 10,000 events (add, cancel, fill)
    constexpr std::size_t kN = 10000;
    for (std::size_t i = 0; i < kN; ++i) {
        if (i % 3 == 0) {
            REQUIRE(producer.publish_add(
                1000 + i, static_cast<std::uint32_t>(10050 + (i % 20)), 10,
                static_cast<std::uint8_t>(i & 1U), i * 50, 42));
        } else if (i % 3 == 1) {
            REQUIRE(producer.publish_cancel(
                1000 + i, static_cast<std::uint32_t>(10050 + (i % 20)), 10,
                static_cast<std::uint8_t>(i & 1U), i * 50));
        } else {
            REQUIRE(producer.publish_fill(
                1000 + i, 2000 + i, static_cast<std::uint32_t>(10050 + (i % 20)), 5,
                static_cast<std::uint8_t>(i & 1U), i * 50));
        }
    }
    REQUIRE(producer.current_seq() == kN);

    for (std::size_t i = 0; i < kN; ++i) {
        REQUIRE(consumer.try_dequeue(out));
        REQUIRE(out.seq == i + 1);
        REQUIRE(out.order_id == 1000 + i);
        REQUIRE(out.timestamp_ns == i * 50);
    }
    REQUIRE_FALSE(consumer.try_dequeue(out));

    endpoint.close();
    ::unlink(test_path);
}

TEST_CASE("experimental offline Hawkes calibrator & VPIN replay: determinism and non-feedback",
          "[experimental][offline_hawkes_calibrator]") {
    using namespace sim::experimental;

    // Positive controls for non-feedback / kernel-input traits
    static_assert(has_matching_feedback_v<KernelFeedbackControl>);
    static_assert(is_matching_kernel_input_v<KernelFeedbackControl>);
    static_assert(!has_matching_feedback_v<ArrivalReplayResult>);
    static_assert(!is_matching_kernel_input_v<ArrivalReplayResult>);

    std::vector<ArrivalEvent> stream = {
        {1, 10, 1, 50},
        {2, 11, 1, 60},
        {3, 12, 1, 40},
        {4, 20, 2, 100},
        {5, 21, 1, 80},
    };
    HawkesParams params{100, 40, 1, 0.0};
    ArrivalPointProcessReplay replay(stream, params);
    const auto r1 = replay.analyze();
    const auto r2 = replay.analyze();

    REQUIRE(r1.diagnostics.n == 5);
    REQUIRE(r1.diagnostics.total_span == 11);
    REQUIRE(r1.diagnostics.min_interarrival == 1);
    REQUIRE(r1.diagnostics.max_interarrival == 8);
    REQUIRE(r1.diagnostics.replay_hash == r2.diagnostics.replay_hash);
    REQUIRE(r1.events[0].intensity == Catch::Approx(100.0));
    // At t=11 (dt=1, decay=0.5): R_1 = 0.5 * (0 + 1) = 0.5, lambda_1 = 100 + 40*0.5 = 120
    REQUIRE(r1.events[1].intensity == Catch::Approx(120.0));

    // State-dependent empirical calibration
    const auto cal = StateDependentHawkesCalibrator::calibrate(stream, 1);
    REQUIRE(cal.empirical_rate > 0.0);
    REQUIRE(cal.branching_ratio >= 0.0);
    REQUIRE(cal.branching_ratio <= 0.95);
    REQUIRE(cal.calibration_hash != 0);

    // VPIN equal-volume bucketing with boundary trade split
    std::vector<SignedTrade> trades = {
        {1, 100, 1000, 60, +1},
        {2, 200, 1000, 80, -1},
        {3, 300, 1001, 60, +1},
        {4, 400, 1002, 25, +1},
    };
    VpinToxicityReplay vpin_replay(trades);
    const auto vres = vpin_replay.evaluate(100);
    REQUIRE(vres.n_buckets == 2);
    REQUIRE(vres.bucketed_volume == 200);
    REQUIRE(vres.residual_volume == 25);
    // Bucket 0: 60 buy + 40 sell -> imbalance +20, abs 20
    REQUIRE(vres.buckets[0].buy_vol == 60);
    REQUIRE(vres.buckets[0].sell_vol == 40);
    REQUIRE(vres.buckets[0].abs_imbalance == 20);
    // Bucket 1: remaining 40 sell + 60 buy -> imbalance +20, abs 20
    REQUIRE(vres.buckets[1].buy_vol == 60);
    REQUIRE(vres.buckets[1].sell_vol == 40);
    REQUIRE(vres.buckets[1].abs_imbalance == 20);
    REQUIRE(vres.vpin_scaled == 200000);
    REQUIRE(vpin_scaled_over_window(vres, 1) == 200000);
}

TEST_CASE("experimental flat_ladder_parity_harness: byte-identical book & vEB static parity",
          "[experimental][flat_ladder_parity]") {
    using namespace sim;
    using namespace sim::experimental;

    std::vector<TraceEvent> trace = {
        {.seq = 1, .ts = 10, .kind = TraceEventKind::SubmitLimit, .order_id = 1, .agent = 1, .side = Side::Buy, .price = 100, .qty = 50},
        {.seq = 2, .ts = 20, .kind = TraceEventKind::SubmitLimit, .order_id = 2, .agent = 2, .side = Side::Buy, .price = 101, .qty = 30},
        {.seq = 3, .ts = 30, .kind = TraceEventKind::SubmitLimit, .order_id = 3, .agent = 3, .side = Side::Sell, .price = 103, .qty = 40},
        {.seq = 4, .ts = 40, .kind = TraceEventKind::SubmitLimit, .order_id = 4, .agent = 4, .side = Side::Sell, .price = 101, .qty = 45},
        {.seq = 5, .ts = 50, .kind = TraceEventKind::Cancel, .order_id = 1},
        {.seq = 6, .ts = 60, .kind = TraceEventKind::SubmitMarket, .order_id = 5, .agent = 5, .side = Side::Buy, .price = 0, .qty = 20},
    };

    const auto rep = FlatLadderParityHarness::verify_trace(90, 110, trace);
    REQUIRE(rep.events_replayed == trace.size());
    REQUIRE(rep.trades_matched == 3);
    REQUIRE(rep.mismatches == 0);
    REQUIRE(rep.flat_digest == rep.ref_digest);
    REQUIRE(rep.parity_ok);

    // Cache-oblivious vEB static search parity against std::lower_bound
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_int_distribution<std::int64_t> dist(-5000, 5000);
    std::vector<std::int64_t> keys;
    keys.reserve(255);
    for (int i = 0; i < 255; ++i) keys.push_back(dist(rng));
    std::sort(keys.begin(), keys.end());

    auto idx = VebStaticSearch<std::int64_t>::from_sorted(keys);
    REQUIRE(idx.size() == keys.size());
    for (std::int64_t q = -5100; q <= 5100; q += 17) {
        const auto exp_rank = static_cast<std::size_t>(
            std::lower_bound(keys.begin(), keys.end(), q) - keys.begin());
        REQUIRE(idx.lower_bound_rank(q) == exp_rank);
        REQUIRE(idx.contains(q) == std::binary_search(keys.begin(), keys.end(), q));
    }
}


TEST_CASE("experimental Ozaki + Queue-Reactive Hawkes MLE: analytical score vs FD and LOBSTER AAPL OOS LLR",
          "[experimental][offline_hawkes_calibrator][mle]") {
    using namespace sim::experimental;

    std::vector<ArrivalEvent> arrivals;
    std::vector<SignedTrade> trades;
    REQUIRE(load_lobster_paired_csv("data/lobster_AAPL_2012-06-21_message_5.csv", arrivals, trades));
    REQUIRE(arrivals.size() == 2000);
    REQUIRE(trades.size() > 0);

    // Row 0 of AAPL has dir=+1, bid_q1=18, ask_q1=200, spread=(5859400-5853300)/100 = 61
    REQUIRE(arrivals[0].queue_depth == 18);
    REQUIRE(arrivals[0].opp_queue_depth == 200);
    REQUIRE(arrivals[0].spread_ticks == 61);
    // Verify pre-event resting L1 queue depth differs from raw order sizes across the stream
    bool found_resting_diff = false;
    for (const auto& ev : arrivals) {
        if (ev.queue_depth >= 100 && ev.spread_ticks >= 1) {
            found_resting_diff = true;
            break;
        }
    }
    REQUIRE(found_resting_diff);

    // 1. Verify analytical score vector against central finite differences to < 1e-5 relative error
    const std::span<const ArrivalEvent> sub_ev(arrivals.data(), 500);
    const double mu0 = 8.5;
    const double alpha0 = 420.0;
    const double beta0 = 950.0;
    const double gamma0 = 0.35;
    const double q_bar = 150.0;

    const auto eval0 = StateDependentHawkesCalibrator::evaluate_log_likelihood(
        sub_ev, mu0, alpha0, beta0, gamma0, q_bar);

    const double eps_mu = 1e-6 * mu0;
    const double eps_alpha = 1e-6 * alpha0;
    const double eps_beta = 1e-6 * beta0;
    const double eps_gamma = 1e-6;
    const double fd_mu = (StateDependentHawkesCalibrator::evaluate_log_likelihood(
                              sub_ev, mu0 + eps_mu, alpha0, beta0, gamma0, q_bar).log_likelihood -
                          StateDependentHawkesCalibrator::evaluate_log_likelihood(
                              sub_ev, mu0 - eps_mu, alpha0, beta0, gamma0, q_bar).log_likelihood) /
                         (2.0 * eps_mu);
    const double fd_alpha = (StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                 sub_ev, mu0, alpha0 + eps_alpha, beta0, gamma0, q_bar).log_likelihood -
                             StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                 sub_ev, mu0, alpha0 - eps_alpha, beta0, gamma0, q_bar).log_likelihood) /
                            (2.0 * eps_alpha);
    const double fd_beta = (StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                sub_ev, mu0, alpha0, beta0 + eps_beta, gamma0, q_bar).log_likelihood -
                            StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                sub_ev, mu0, alpha0, beta0 - eps_beta, gamma0, q_bar).log_likelihood) /
                           (2.0 * eps_beta);
    const double fd_gamma = (StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                 sub_ev, mu0, alpha0, beta0, gamma0 + eps_gamma, q_bar).log_likelihood -
                             StateDependentHawkesCalibrator::evaluate_log_likelihood(
                                 sub_ev, mu0, alpha0, beta0, gamma0 - eps_gamma, q_bar).log_likelihood) /
                            (2.0 * eps_gamma);

    const double rel_mu = std::abs(eval0.grad_mu - fd_mu) / std::max(1e-12, std::abs(fd_mu));
    const double rel_alpha = std::abs(eval0.grad_alpha - fd_alpha) / std::max(1e-12, std::abs(fd_alpha));
    const double rel_beta = std::abs(eval0.grad_beta - fd_beta) / std::max(1e-12, std::abs(fd_beta));
    const double rel_gamma = std::abs(eval0.grad_gamma - fd_gamma) / std::max(1e-12, std::abs(fd_gamma));

    REQUIRE(rel_mu < 1e-5);
    REQUIRE(rel_alpha < 1e-5);
    REQUIRE(rel_beta < 1e-5);
    REQUIRE(rel_gamma < 1e-5);

    // 2. Verify calibrate_mle strictly improves log-likelihood and achieves positive OOS LLR on AAPL
    const auto mle = StateDependentHawkesCalibrator::calibrate_mle(
        std::span<const ArrivalEvent>(arrivals.data(), arrivals.size()), 0.70);
    REQUIRE(mle.n_total == 2000);
    REQUIRE(mle.n_train == 1400);
    REQUIRE(mle.n_oos == 600);
    REQUIRE(mle.mu_hz > 0.0);
    REQUIRE(mle.alpha_hz > 0.0);
    REQUIRE(mle.beta_hz > 0.0);
    REQUIRE(mle.branching_ratio > 0.0);
    REQUIRE(mle.branching_ratio <= 0.98);
    REQUIRE(mle.log_likelihood_train > mle.log_likelihood_init);
    REQUIRE(mle.log_likelihood_train > mle.log_likelihood_poisson_train);
    REQUIRE(mle.oos_llr_per_event > 0.0);
    REQUIRE(mle.se_mu > 0.0);
    REQUIRE(mle.se_alpha > 0.0);
    REQUIRE(mle.se_beta > 0.0);
    REQUIRE(mle.oos_ogata_residuals.size() == mle.n_oos);
    REQUIRE(mle.ks_statistic > 0.0);
    REQUIRE(mle.ks_statistic < 1.0);

    // 3. Verify FFI entrypoint qrlob_ffi_calibrate_lobster_mle
    QrlobMleCalibrationReport ffi_rep{};
    std::vector<double> oos_buf(600, 0.0);
    std::size_t out_n_res = 0;
    const int rc = qrlob_ffi_calibrate_lobster_mle(
        "data/lobster_AAPL_2012-06-21_message_5.csv", 0.70,
        &ffi_rep, oos_buf.data(), oos_buf.size(), &out_n_res);
    REQUIRE(rc == 0);
    REQUIRE(ffi_rep.n_total == 2000);
    REQUIRE(ffi_rep.oos_llr_per_event == Catch::Approx(mle.oos_llr_per_event));
    REQUIRE(out_n_res == 600);
}

TEST_CASE("experimental SPSC unified leangrph ABI (0x434E49524E41454C) and 16B QrlobShmMicrostructureExt",
          "[experimental][spsc_ring_bridge][leangrph_abi]") {
    using namespace sim;
    using namespace sim::experimental;

    static_assert(sizeof(QrlobShmMicrostructureExt) == 16, "16B extension size mismatch");
    static_assert(sizeof(ShmIpcEvent) == 64, "64B event size mismatch");
    static_assert(kLeangrphShmMagic == 0x434E49524E41454CULL, "Magic constant mismatch");
    static_assert(kLeangrphRingCapacity == 1024, "Capacity constant mismatch");

    // 1. Pack / unpack round-trip on ShmIpcEvent::_reserved[16]
    ShmIpcEvent ev{};
    QrlobShmMicrostructureExt ext_in{
        .ofi_q16 = -131072,
        .hawkes_intensity_q16 = 983040U,
        .nr3d_vol_q8 = 5120U,
        .vpin_bps = 3125U,
        .spread_ticks = 3U,
        .level_and_regime = 0x21U,
        .reprice_shift = -2
    };
    QrlobShmMicrostructureExt ext_out{};
    REQUIRE_FALSE(unpack_microstructure_ext(ev, ext_out));
    pack_microstructure_ext(ev, ext_in);
    REQUIRE((ev.flags & kShmFlagQrlobTelemetry) != 0U);
    REQUIRE(unpack_microstructure_ext(ev, ext_out));
    REQUIRE(ext_out.ofi_q16 == ext_in.ofi_q16);
    REQUIRE(ext_out.hawkes_intensity_q16 == ext_in.hawkes_intensity_q16);
    REQUIRE(ext_out.nr3d_vol_q8 == ext_in.nr3d_vol_q8);
    REQUIRE(ext_out.vpin_bps == ext_in.vpin_bps);
    REQUIRE(ext_out.spread_ticks == ext_in.spread_ticks);
    REQUIRE(ext_out.level_and_regime == ext_in.level_and_regime);
    REQUIRE(ext_out.reprice_shift == ext_in.reprice_shift);

    // 2. Full FFI + ShmSpscFeatureExportSink + ShmSpscConsumer round-trip with leangrph ABI
    const char* shm_path = "/dev/shm/qrlob_leangrph_abi_test";
    std::uint64_t published = 0;
    std::uint64_t final_hash = 0;
    const int rc = qrlob_ffi_generate_shm_microstructure_stream(
        shm_path, 256, 20260621ULL, 1, &published, &final_hash);
    REQUIRE(rc == 0);
    REQUIRE(published == 256);
    REQUIRE(final_hash != 0);

    ShmSpscEndpoint reader_ep;
    REQUIRE(reader_ep.open(shm_path, false, true));
    REQUIRE(reader_ep.is_valid());
    REQUIRE(reader_ep.header()->magic == kLeangrphShmMagic);
    REQUIRE(reader_ep.header()->capacity == kLeangrphRingCapacity);

    ShmSpscConsumer consumer(reader_ep.request_channel(), reader_ep.ring_capacity());
    ShmIpcEvent read_ev{};
    std::size_t count = 0;
    while (consumer.try_dequeue(read_ev)) {
        ++count;
        REQUIRE(read_ev.seq == count);
        REQUIRE(read_ev.event_type == 4U);
        QrlobShmMicrostructureExt read_ext{};
        REQUIRE(unpack_microstructure_ext(read_ev, read_ext));
        REQUIRE(read_ext.hawkes_intensity_q16 > 0U);
        REQUIRE(read_ext.spread_ticks >= 1U);
    }
    REQUIRE(count == 256);
    reader_ep.close();
    ::unlink(shm_path);
}

#endif // QRLOB_BUILD_EXPERIMENTAL
