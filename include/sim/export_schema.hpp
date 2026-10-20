#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace sim {

struct ExportColumn {
    std::string_view name;
    std::string_view side{};
    std::size_t index = 0;
    std::size_t row = 0;
    std::size_t col = 0;
};

// Schema metadata descriptor for exported feature, label, and TCA frames.
// Records schema version, timestamp basis, price/quantity units, missing-value
// policy, and forward label horizon so consumers can verify compatibility.

enum class TimestampBasis : std::uint8_t {
    Unspecified = 0,
    EventTime,   // integer event sequence / event-time ns (deterministic)
    WallClock,   // wall-clock ns (non-deterministic; diagnostics only)
};

enum class QuantityUnit : std::uint8_t {
    Unspecified = 0,
    Ticks,       // integer price ticks
    PriceX2,     // doubled integer tick (mid_x2 style, exact half-ticks)
    Lots,        // integer share/contract lots
    Notional,    // integer price*qty notional
    Ratio,       // dimensionless double ratio (derived/cold-path diagnostic)
};

enum class LabelAvailability : std::uint8_t {
    NotApplicable = 0,  // frame carries no label columns
    Available,          // label horizon fully realized within the data
    Pending,            // label horizon extends beyond the exported prefix
    Censored,           // label cannot be realized (e.g. order never filled)
};

enum class MissingPolicy : std::uint8_t {
    Unspecified = 0,
    Forbidden,       // a missing/sparse cell is a contract violation
    ZeroFilled,      // sparse fixed-grid holes are written as 0
    SentinelFlagged, // missing cells carry an explicit sentinel + flag column
};

// Bump on any incompatible column-set or semantics change; consumers must check
// it before importing an exported frame.
inline constexpr std::uint32_t kExportSchemaVersion = 1;

struct ExportSchemaMeta {
    std::uint32_t schema_version = kExportSchemaVersion;
    std::string_view schema_name{};  // e.g. "book_observables", "order_flow"
    TimestampBasis timestamp_basis = TimestampBasis::Unspecified;
    QuantityUnit price_unit = QuantityUnit::Unspecified;
    QuantityUnit quantity_unit = QuantityUnit::Unspecified;
    // Forward markout / label horizon expressed in the timestamp_basis unit
    // (0 = no labels). Distinguishes feature-only frames from labelled frames.
    std::uint64_t label_horizon = 0;
    LabelAvailability label_availability = LabelAvailability::NotApplicable;
    MissingPolicy missing_policy = MissingPolicy::Unspecified;
    // Provenance: deterministic prefix replay hash of the event stream through
    // the frame this schema describes, plus an opaque source/run trace id and
    // the normalization/calibration window id (0 = raw / none).
    std::uint64_t replay_hash = 0;
    std::uint64_t source_trace_id = 0;
    std::uint64_t calibration_window_id = 0;
};

// Two metas are "interpretation-compatible" iff a consumer can compare or merge
// the underlying numeric frames without silently mixing schema versions, units,
// time bases, or label horizons. Provenance fields (replay_hash, source_trace_id,
// calibration_window_id) are deliberately excluded: they identify *which* stream
// produced the frame, not *how to read* its numbers.
inline constexpr bool interpretation_compatible(const ExportSchemaMeta& a,
                                                const ExportSchemaMeta& b) noexcept {
    return a.schema_version == b.schema_version &&
           a.schema_name == b.schema_name &&
           a.timestamp_basis == b.timestamp_basis &&
           a.price_unit == b.price_unit &&
           a.quantity_unit == b.quantity_unit &&
           a.label_horizon == b.label_horizon &&
           a.label_availability == b.label_availability &&
           a.missing_policy == b.missing_policy;
}

} // namespace sim
