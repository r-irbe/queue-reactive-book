// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "book_observables.hpp"  // BookObservables, TopKBookSnapshot, DepthLevel
#include "export_schema.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sim {

// Consecutive-snapshot L1 Order Flow Imbalance (Cont, Kukanov, Stoikov 2014)
// and populated-depth sparsity metrics across a fixed-K level grid:
//
//   bid term:  P_b up   -> +q_b(t)
//              P_b same -> +(q_b(t) - q_b(t-1))
//              P_b down -> -q_b(t-1)
//   ask term:  P_a up   -> +q_a(t-1)
//              P_a same -> -(q_a(t) - q_a(t-1))
//              P_a down -> -q_a(t)
//   OFI = bid term + ask term
//
// Bid pressure (bid lifting / ask pulled up) is positive; sell pressure
// (bid pulled down / ask lowered) is negative. All terms are exact integers.
// If either side is absent in either snapshot, that side's term is unknown and
// the sample is `censored` (the side's term is reported as 0, never fabricated).

struct OfiSample {
    EventSeq seq = 0;
    TimeNs ts = 0;
    bool has_bid = false;
    bool has_ask = false;
    bool censored = false;  // a quoted side was missing in prev or curr
    std::int64_t bid_term = 0;
    std::int64_t ask_term = 0;
    std::int64_t ofi = 0;
};

inline OfiSample make_ofi_sample(const BookObservables& prev,
                                 const BookObservables& curr) noexcept {
    OfiSample out{};
    out.seq = curr.seq;
    out.ts = curr.ts;
    out.has_bid = prev.has_bid & curr.has_bid;
    out.has_ask = prev.has_ask & curr.has_ask;
    out.censored = !(out.has_bid & out.has_ask);

    const std::int64_t qb = curr.best_bid_qty;
    const std::int64_t qb_prev = prev.best_bid_qty;
    const std::int64_t bid_val = qb * (curr.best_bid >= prev.best_bid) -
                                 qb_prev * (curr.best_bid <= prev.best_bid);
    out.bid_term = bid_val * out.has_bid;

    const std::int64_t qa = curr.best_ask_qty;
    const std::int64_t qa_prev = prev.best_ask_qty;
    const std::int64_t ask_val = qa_prev * (curr.best_ask >= prev.best_ask) -
                                 qa * (curr.best_ask <= prev.best_ask);
    out.ask_term = ask_val * out.has_ask;

    out.ofi = out.bid_term + out.ask_term;
    return out;
}

inline constexpr std::size_t kOfiExportCount = 7;
inline constexpr std::array<ExportColumn, kOfiExportCount> kOfiExportSchema = {
    {{"seq"}, {"ts"}, {"has_bid"}, {"has_ask"}, {"censored"}, {"ofi"}, {"ofi_complete"}}};

inline ExportSchemaMeta ofi_schema_meta() noexcept {
    ExportSchemaMeta meta{};
    meta.schema_name = "ofi_l1";
    meta.timestamp_basis = TimestampBasis::EventTime;
    meta.price_unit = QuantityUnit::Ticks;
    meta.quantity_unit = QuantityUnit::Lots;
    meta.label_availability = LabelAvailability::NotApplicable;
    // A censored sample is the sentinel-flag for an unavailable quoted side.
    meta.missing_policy = MissingPolicy::SentinelFlagged;
    return meta;
}

inline std::array<double, kOfiExportCount>
export_ofi_sample(const OfiSample& s) noexcept {
    return {static_cast<double>(s.seq),
            static_cast<double>(s.ts),
            s.has_bid ? 1.0 : 0.0,
            s.has_ask ? 1.0 : 0.0,
            s.censored ? 1.0 : 0.0,
            static_cast<double>(s.ofi),
            s.censored ? 0.0 : 1.0};  // ofi_complete: both sides contributed
}

// Actual-depth sparsity over a fixed-K snapshot. A fixed-K depth vector zero-
// fills levels that do not exist, so a consumer cannot tell a genuine zero from
// a structural hole. These fields make the populated-vs-grid ratio explicit so
// zero-filled cells are not read as real zero liquidity (MissingPolicy context).

struct DepthSparsity {
    EventSeq seq = 0;
    TimeNs ts = 0;
    std::uint16_t grid_levels = 0;       // K
    std::uint16_t bid_populated = 0;
    std::uint16_t ask_populated = 0;
    double bid_sparsity = 0.0;           // (K - populated) / K, in [0,1]
    double ask_sparsity = 0.0;
    bool bid_full = false;               // populated == K
    bool ask_full = false;
};

template <std::size_t K>
inline DepthSparsity make_depth_sparsity(const TopKBookSnapshot<K>& snap) noexcept {
    DepthSparsity out{};
    out.seq = snap.seq;
    out.ts = snap.ts;
    out.grid_levels = static_cast<std::uint16_t>(K);
    out.bid_populated = snap.bid_count;
    out.ask_populated = snap.ask_count;
    if constexpr (K > 0) {
        out.bid_sparsity =
            static_cast<double>(K - snap.bid_count) / static_cast<double>(K);
        out.ask_sparsity =
            static_cast<double>(K - snap.ask_count) / static_cast<double>(K);
    }
    out.bid_full = snap.bid_count == static_cast<std::uint8_t>(K);
    out.ask_full = snap.ask_count == static_cast<std::uint8_t>(K);
    return out;
}

inline constexpr std::size_t kDepthSparsityExportCount = 8;
inline constexpr std::array<ExportColumn, kDepthSparsityExportCount>
    kDepthSparsityExportSchema = {{{"seq"},
                                   {"ts"},
                                   {"grid_levels"},
                                   {"bid_populated"},
                                   {"ask_populated"},
                                   {"bid_sparsity"},
                                   {"ask_sparsity"},
                                   {"both_full"}}};

inline std::array<double, kDepthSparsityExportCount>
export_depth_sparsity(const DepthSparsity& s) noexcept {
    return {static_cast<double>(s.seq),
            static_cast<double>(s.ts),
            static_cast<double>(s.grid_levels),
            static_cast<double>(s.bid_populated),
            static_cast<double>(s.ask_populated),
            s.bid_sparsity,
            s.ask_sparsity,
            (s.bid_full && s.ask_full) ? 1.0 : 0.0};
}

// ----------------------------------------------------------------------------
// Multi-Level Order Flow Imbalance (OFI) across TopKBookSnapshot<K>
// Scalar Reference + AVX2 4-Way int64_t / double Branchless Acceleration
// ----------------------------------------------------------------------------

template <std::size_t K = 8>
struct MultiLevelOfiSample {
    EventSeq seq = 0;
    TimeNs ts = 0;
    std::uint8_t bid_levels_valid = 0;
    std::uint8_t ask_levels_valid = 0;
    std::array<std::int64_t, K> bid_terms{};
    std::array<std::int64_t, K> ask_terms{};
    std::array<std::int64_t, K> level_ofi{};
    std::int64_t l1_ofi = 0;
    std::int64_t total_ofi = 0;
    double integrated_ofi = 0.0;
    double weighted_ofi = 0.0;
    double normalized_ofi = 0.0;
};

inline constexpr std::array<double, 8> kDefaultOfiDecayWeights8 = {
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125
};

template <std::size_t K = 8>
constexpr std::array<double, K> default_ofi_decay_weights(double decay = 0.5) noexcept {
    std::array<double, K> w{};
    double cur = 1.0;
    for (std::size_t i = 0; i < K; ++i) {
        w[i] = cur;
        cur *= decay;
    }
    return w;
}

template <std::size_t K = 8>
inline MultiLevelOfiSample<K> compute_multilevel_ofi_scalar(
    const TopKBookSnapshot<K>& prev,
    const TopKBookSnapshot<K>& curr,
    const std::array<double, K>& weights = default_ofi_decay_weights<K>()) noexcept {
    MultiLevelOfiSample<K> out{};
    out.seq = curr.seq;
    out.ts = curr.ts;
    const std::uint8_t valid_b = std::min(prev.bid_count, curr.bid_count);
    const std::uint8_t valid_a = std::min(prev.ask_count, curr.ask_count);
    out.bid_levels_valid = valid_b;
    out.ask_levels_valid = valid_a;

    std::int64_t sum_ofi = 0;
    double weighted_sum = 0.0;

    for (std::size_t m = 0; m < K; ++m) {
        const std::int64_t has_b = static_cast<std::int64_t>(m < valid_b);
        const std::int64_t pb_curr = curr.bids[m].price;
        const std::int64_t pb_prev = prev.bids[m].price;
        const std::int64_t qb_curr = curr.bids[m].qty;
        const std::int64_t qb_prev = prev.bids[m].qty;
        const std::int64_t b_ge = static_cast<std::int64_t>(pb_curr >= pb_prev);
        const std::int64_t b_le = static_cast<std::int64_t>(pb_curr <= pb_prev);
        const std::int64_t b_term = (qb_curr * b_ge - qb_prev * b_le) * has_b;

        const std::int64_t has_a = static_cast<std::int64_t>(m < valid_a);
        const std::int64_t pa_curr = curr.asks[m].price;
        const std::int64_t pa_prev = prev.asks[m].price;
        const std::int64_t qa_curr = curr.asks[m].qty;
        const std::int64_t qa_prev = prev.asks[m].qty;
        const std::int64_t a_ge = static_cast<std::int64_t>(pa_curr >= pa_prev);
        const std::int64_t a_le = static_cast<std::int64_t>(pa_curr <= pa_prev);
        const std::int64_t a_term = (qa_prev * a_ge - qa_curr * a_le) * has_a;

        const std::int64_t l_ofi = b_term + a_term;
        out.bid_terms[m] = b_term;
        out.ask_terms[m] = a_term;
        out.level_ofi[m] = l_ofi;
        sum_ofi += l_ofi;
        weighted_sum += static_cast<double>(l_ofi) * weights[m];
    }

    out.l1_ofi = (K > 0) ? out.level_ofi[0] : 0;
    out.total_ofi = sum_ofi;
    out.integrated_ofi = weighted_sum;
    out.weighted_ofi = weighted_sum;
    double w_norm = 0.0;
    for (std::size_t i = 0; i < K; ++i) w_norm += weights[i];
    out.normalized_ofi = (w_norm > 0.0) ? (weighted_sum / w_norm) : 0.0;
    return out;
}

template <std::size_t K = 8>
[[gnu::hot]] inline MultiLevelOfiSample<K> compute_multilevel_ofi_avx2(
    const TopKBookSnapshot<K>& prev,
    const TopKBookSnapshot<K>& curr,
    const std::array<double, K>& weights = default_ofi_decay_weights<K>()) noexcept {
#if defined(__AVX2__)
    if constexpr (K == 8) {
        MultiLevelOfiSample<8> out{};
        out.seq = curr.seq;
        out.ts = curr.ts;
        const std::uint8_t valid_b = std::min(prev.bid_count, curr.bid_count);
        const std::uint8_t valid_a = std::min(prev.ask_count, curr.ask_count);
        out.bid_levels_valid = valid_b;
        out.ask_levels_valid = valid_a;

        for (std::size_t base = 0; base < 8; base += 4) {
            auto load_p_b = [](const std::array<DepthLevel, 8>& lvls, std::size_t b) noexcept {
                return _mm256_set_epi64x(lvls[b + 3].price, lvls[b + 2].price,
                                         lvls[b + 1].price, lvls[b + 0].price);
            };
            auto load_q_b = [](const std::array<DepthLevel, 8>& lvls, std::size_t b) noexcept {
                return _mm256_set_epi64x(lvls[b + 3].qty, lvls[b + 2].qty,
                                         lvls[b + 1].qty, lvls[b + 0].qty);
            };

            const __m256i pb_c = load_p_b(curr.bids, base);
            const __m256i pb_p = load_p_b(prev.bids, base);
            const __m256i qb_c = load_q_b(curr.bids, base);
            const __m256i qb_p = load_q_b(prev.bids, base);

            // mask_ge: !(pb_p > pb_c) => ~_mm256_cmpgt_epi64(pb_p, pb_c)
            // mask_le: !(pb_c > pb_p) => ~_mm256_cmpgt_epi64(pb_c, pb_p)
            const __m256i all_ones = _mm256_set1_epi64x(-1LL);
            const __m256i b_gt_pc = _mm256_cmpgt_epi64(pb_p, pb_c);
            const __m256i b_gt_cp = _mm256_cmpgt_epi64(pb_c, pb_p);
            const __m256i b_mask_ge = _mm256_andnot_si256(b_gt_pc, all_ones);
            const __m256i b_mask_le = _mm256_andnot_si256(b_gt_cp, all_ones);

            const __m256i valid_b_mask = _mm256_set_epi64x(
                (base + 3 < valid_b) ? -1LL : 0LL,
                (base + 2 < valid_b) ? -1LL : 0LL,
                (base + 1 < valid_b) ? -1LL : 0LL,
                (base + 0 < valid_b) ? -1LL : 0LL);

            const __m256i b_pos = _mm256_and_si256(qb_c, b_mask_ge);
            const __m256i b_neg = _mm256_and_si256(qb_p, b_mask_le);
            const __m256i b_term = _mm256_and_si256(_mm256_sub_epi64(b_pos, b_neg), valid_b_mask);

            const __m256i pa_c = load_p_b(curr.asks, base);
            const __m256i pa_p = load_p_b(prev.asks, base);
            const __m256i qa_c = load_q_b(curr.asks, base);
            const __m256i qa_p = load_q_b(prev.asks, base);

            const __m256i a_gt_pc = _mm256_cmpgt_epi64(pa_p, pa_c);
            const __m256i a_gt_cp = _mm256_cmpgt_epi64(pa_c, pa_p);
            const __m256i a_mask_ge = _mm256_andnot_si256(a_gt_pc, all_ones);
            const __m256i a_mask_le = _mm256_andnot_si256(a_gt_cp, all_ones);

            const __m256i valid_a_mask = _mm256_set_epi64x(
                (base + 3 < valid_a) ? -1LL : 0LL,
                (base + 2 < valid_a) ? -1LL : 0LL,
                (base + 1 < valid_a) ? -1LL : 0LL,
                (base + 0 < valid_a) ? -1LL : 0LL);

            const __m256i a_pos = _mm256_and_si256(qa_p, a_mask_ge);
            const __m256i a_neg = _mm256_and_si256(qa_c, a_mask_le);
            const __m256i a_term = _mm256_and_si256(_mm256_sub_epi64(a_pos, a_neg), valid_a_mask);

            const __m256i l_ofi = _mm256_add_epi64(b_term, a_term);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out.bid_terms[base]), b_term);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out.ask_terms[base]), a_term);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out.level_ofi[base]), l_ofi);
        }

        // AVX2 4-way double depth-decayed integration
        const __m256d ofi_d0 = _mm256_set_pd(
            static_cast<double>(out.level_ofi[3]),
            static_cast<double>(out.level_ofi[2]),
            static_cast<double>(out.level_ofi[1]),
            static_cast<double>(out.level_ofi[0]));
        const __m256d ofi_d1 = _mm256_set_pd(
            static_cast<double>(out.level_ofi[7]),
            static_cast<double>(out.level_ofi[6]),
            static_cast<double>(out.level_ofi[5]),
            static_cast<double>(out.level_ofi[4]));

        const __m256d w0 = _mm256_loadu_pd(&weights[0]);
        const __m256d w1 = _mm256_loadu_pd(&weights[4]);
        const __m256d prod0 = _mm256_mul_pd(ofi_d0, w0);
        const __m256d prod1 = _mm256_mul_pd(ofi_d1, w1);

        alignas(32) double p0[4]{};
        alignas(32) double p1[4]{};
        _mm256_store_pd(p0, prod0);
        _mm256_store_pd(p1, prod1);

        std::int64_t total = 0;
        double integrated = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            total += out.level_ofi[i];
            integrated += p0[i];
        }
        for (std::size_t i = 0; i < 4; ++i) {
            total += out.level_ofi[4 + i];
            integrated += p1[i];
        }
        out.l1_ofi = out.level_ofi[0];
        out.total_ofi = total;
        out.integrated_ofi = integrated;
        out.weighted_ofi = integrated;
        double w_norm = 0.0;
        for (std::size_t i = 0; i < 8; ++i) w_norm += weights[i];
        out.normalized_ofi = (w_norm > 0.0) ? (integrated / w_norm) : 0.0;
        return out;
    }
#endif
    return compute_multilevel_ofi_scalar<K>(prev, curr, weights);
}

template <std::size_t K = 8>
inline double compute_integrated_ofi_avx2(
    const TopKBookSnapshot<K>& prev,
    const TopKBookSnapshot<K>& curr,
    const std::array<double, K>& weights = default_ofi_decay_weights<K>()) noexcept {
    return compute_multilevel_ofi_avx2<K>(prev, curr, weights).integrated_ofi;
}

} // namespace sim
