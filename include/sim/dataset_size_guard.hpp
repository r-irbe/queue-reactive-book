// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

// Pre-flight memory and event-count budget check for in-RAM replay datasets.
// Classifies expected (events, bytes) into InRam, Warn, or Refuse against
// caller-configured limits (0 = unbounded on that axis).

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sim {

// A budget for an in-RAM replay. `0` for a limit means "unbounded" on that
// axis. Disabled by default so the guard is a no-op unless explicitly armed.
struct DatasetSizeBudget {
    std::uint64_t max_events = 0;   // 0 => unbounded
    std::uint64_t max_bytes = 0;    // 0 => unbounded
    double warn_fraction = 0.8;     // warn band starts at this fraction of a limit
    bool enabled = false;
};

enum class SizeRegime {
    InRam,  // comfortably within budget
    Warn,   // within budget but in the warn band
    Refuse  // exceeds a non-zero budget limit
};

inline SizeRegime classify_dataset_size(std::uint64_t expected_events,
                                        std::uint64_t expected_bytes,
                                        const DatasetSizeBudget& budget) noexcept {
    if (!budget.enabled) return SizeRegime::InRam;

    const bool over_events = budget.max_events != 0 && expected_events > budget.max_events;
    const bool over_bytes = budget.max_bytes != 0 && expected_bytes > budget.max_bytes;
    if (over_events || over_bytes) return SizeRegime::Refuse;

    // Reject invalid/NaN warn_fraction conservatively: fall back to 1.0 so the
    // warn band is just the limit itself rather than silently mis-classifying.
    double warn = budget.warn_fraction;
    if (!(warn >= 0.0) || warn > 1.0) warn = 1.0;

    const bool warn_events =
        budget.max_events != 0 &&
        static_cast<double>(expected_events) >= warn * static_cast<double>(budget.max_events);
    const bool warn_bytes =
        budget.max_bytes != 0 &&
        static_cast<double>(expected_bytes) >= warn * static_cast<double>(budget.max_bytes);
    if (warn_events || warn_bytes) return SizeRegime::Warn;

    return SizeRegime::InRam;
}

inline bool check_dataset_size(std::uint64_t expected_events,
                               std::uint64_t expected_bytes,
                               const DatasetSizeBudget& budget) noexcept {
    return classify_dataset_size(expected_events, expected_bytes, budget) != SizeRegime::Refuse;
}

// Throws std::runtime_error when the expected dataset exceeds a non-zero limit.
// A no-op when the budget is disabled or within bounds.
inline void check_dataset_size_or_throw(std::uint64_t expected_events,
                                        std::uint64_t expected_bytes,
                                        const DatasetSizeBudget& budget) {
    if (classify_dataset_size(expected_events, expected_bytes, budget) == SizeRegime::Refuse) {
#if defined(__cpp_exceptions)
        throw std::runtime_error(
            "dataset-size guard: refusing replay -- expected dataset (" +
            std::to_string(expected_events) + " events, " +
            std::to_string(expected_bytes) + " bytes) exceeds in-RAM budget (" +
            std::to_string(budget.max_events) + " events, " +
            std::to_string(budget.max_bytes) + " bytes); fail loud instead of thrashing");
#endif
    }
}

} // namespace sim
