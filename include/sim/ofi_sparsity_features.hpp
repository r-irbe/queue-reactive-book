#pragma once

#include "book_observables.hpp"  // BookObservables, TopKBookSnapshot, DepthLevel
#include "export_schema.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

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

} // namespace sim
