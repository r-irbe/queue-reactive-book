// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

#pragma once

#include "export_schema.hpp"
#include "fill_attribution.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace sim {

struct MarkoutHorizon {
    EventSeq events = 0;
    TimeNs ns = 0;
};

inline bool markout_horizon_due(const FillRecord& fill,
                                const BookObservables& current,
                                const MarkoutHorizon& horizon) noexcept {
    const bool event_due =
        horizon.events == 0 ||
        current.seq >= static_cast<EventSeq>(fill.seq + horizon.events);
    const bool time_due =
        horizon.ns == 0 || current.ts >= static_cast<TimeNs>(fill.ts + horizon.ns);
    return event_due && time_due;
}

// Label-availability of a forward markout as of `current`, bridging to the
// export-schema LabelAvailability contract. A markout is Available only once its
// horizon is reached; before that it is Pending and MUST NOT be emitted with a
// current-book value (that would be look-ahead leakage). "Censored" (a horizon
// that can never be reached because the data ends first) is a higher-level
// judgement the observer cannot make from a single snapshot, so it is not
// returned here.
inline LabelAvailability markout_label_availability(const FillRecord& fill,
                                                    const BookObservables& current,
                                                    const MarkoutHorizon& horizon) noexcept {
    return markout_horizon_due(fill, current, horizon) ? LabelAvailability::Available
                                                       : LabelAvailability::Pending;
}

class MarkoutObserver {
public:
    explicit MarkoutObserver(std::vector<MarkoutHorizon> horizons)
        : horizons_(std::move(horizons)) {}

    void record_fill(const FillRecord& fill) {
        pending_.push_back(PendingFill{.fill = fill,
                                       .emitted = std::vector<bool>(horizons_.size(), false),
                                       .remaining = horizons_.size()});
    }

    std::vector<MarkoutRecord> evaluate(const BookObservables& current) {
        std::vector<MarkoutRecord> out{};
        for (auto& pending : pending_) {
            for (std::size_t idx = 0; idx < horizons_.size(); ++idx) {
                if (pending.emitted[idx]) continue;
                const MarkoutHorizon& horizon = horizons_[idx];
                if (!markout_horizon_due(pending.fill, current, horizon)) continue;

                out.push_back(make_markout_record(pending.fill, current, current.seq,
                                                  horizon.events, horizon.ns));
                pending.emitted[idx] = true;
                --pending.remaining;
            }
        }
        erase_completed();
        return out;
    }

    std::size_t pending_count() const noexcept { return pending_.size(); }
    const std::vector<MarkoutHorizon>& horizons() const noexcept { return horizons_; }

private:
    struct PendingFill {
        FillRecord fill{};
        std::vector<bool> emitted{};
        std::size_t remaining = 0;
    };

    std::vector<MarkoutHorizon> horizons_{};
    std::vector<PendingFill> pending_{};

    void erase_completed() {
        std::size_t write = 0;
        for (std::size_t read = 0; read < pending_.size(); ++read) {
            if (pending_[read].remaining == 0) continue;
            if (write != read) pending_[write] = std::move(pending_[read]);
            ++write;
        }
        pending_.resize(write);
    }
};

} // namespace sim
