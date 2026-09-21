#pragma once

#include "types.hpp"

namespace sim {
    //  Latency model
    //      md_delay_ns: from match-engine event to agent (market data hop)
    //      decision_delay_ns:  agent thinking time (depends on agent)
    //      oe_delay_ns:        from agent decision back to match engine (order entry hop)
    //  Constant + jitter, by default uniform distribution [0,jitter_ns]
    struct LatencyConfig {
        TimeNs md_delay_ns        = 50'000;    
        TimeNs decision_delay_ns  = 10'000;
        TimeNs oe_delay_ns        = 50'000;
        TimeNs jitter_ns          = 0;
    };

    struct LatencyProfile {
        LatencyConfig md;
        LatencyConfig dec;
        LatencyConfig oe;
    };
}