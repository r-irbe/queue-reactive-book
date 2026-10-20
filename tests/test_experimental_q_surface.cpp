// Experimental surface tests: SPSC shared-memory ring, offline Hawkes/VPIN
// replay, and flat-ladder parity harness.

#if !defined(QRLOB_BUILD_EXPERIMENTAL) || QRLOB_BUILD_EXPERIMENTAL

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "flat_ladder_parity_harness.hpp"
#include "offline_hawkes_calibrator.hpp"
#include "spsc_ring_bridge.hpp"

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

#endif // QRLOB_BUILD_EXPERIMENTAL
