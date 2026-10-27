#pragma once

// PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0/)
// Required Notice: Copyright (c) 2026 Radu Irbe (GitHub: r-irbe).

// ============================================================================
// Consolidated Market Microstructure Features Module
//
// Unifies L1/L2 Book Observables, Multi-level Depth Vectors, Order Flow
// Imbalance (OFI), Sparse OFI, Moving Window Statistics, Queue-State
// Observers, Feature Export Schemas, and Causality/Embargo Join Manifests.
// ============================================================================

#include "book_observables.hpp"
#include "volume_depth_features.hpp"
#include "order_flow_features.hpp"
#include "ofi_sparsity_features.hpp"
#include "moving_window_features.hpp"
#include "queue_state_observer.hpp"
#include "feature_export.hpp"
#include "feature_export_contract.hpp"
#include "join_manifest.hpp"
