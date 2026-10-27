#pragma once

// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// ============================================================================
// Queue-Reactive Limit Order Book & Market Microstructure Simulation Toolkit
// Master Canonical Umbrella Header
// ============================================================================

#include "types.hpp"
#include "bitwise_veb_codings.hpp"
#include "veb_static_search.hpp"
#include "flat_ladder_book.hpp"
#include "lob.hpp"
#include "queue_reactive.hpp"
#include "rng.hpp"
#include "event_trace.hpp"
#include "features.hpp"
#include "analytics.hpp"
#include "replay.hpp"
#include "quote_adapters.hpp"
#include "risk_guard.hpp"
#include "alloc_probe.hpp"
#include "dataset_size_guard.hpp"
#include "ofi_sparsity_features.hpp"
#include "simulation.hpp"

#if !defined(QRLOB_BUILD_EXPERIMENTAL) || QRLOB_BUILD_EXPERIMENTAL
#include "flat_ladder_parity_harness.hpp"
#include "offline_hawkes_calibrator.hpp"
#include "spsc_ring_bridge.hpp"
#endif
