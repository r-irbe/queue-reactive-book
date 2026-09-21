#include <catch2/catch_test_macros.hpp>

#include "offline/feature_export_schema_contract_replay.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

// =============================================================================
// Deterministic MOC-level feature-export schema/horizon-contract replay fixture
//
// Proves the properties the MOC's load-bearing off-kernel export boundary needs:
//   (a) CONTRACT-HASH DETERMINISM + ORDER-SENSITIVITY -- same stream + schema ->
//       identical rows + FNV contract hash; reorder / tamper changes the hash.
//   (b) SELF-DESCRIBING SCHEMA ROUND-TRIP -- serialize()/parse() is the identity
//       on the export, and a MISMATCHED expected schema version is detected.
//   (c) OBSERVER NON-PERTURBATION -- a read-only observer (on/off) leaves the
//       contract hash invariant; a perturbing observer (positive control)
//       changes it.
//   (d) POINT-IN-TIME AVAILABILITY / CENSORING (no future leakage) -- a label is
//       available iff its future event t+h is present; a censored label flips to
//       available IFF that future event is appended to the input suffix.
//   (e) EXPORT CORRECTNESS -- features/labels match a hand-built series exactly.
//   (f) NON-FEEDBACK / READ-ONLY -- the export takes a const stream, cannot
//       mutate it, and the offline types carry no matching-engine coupling
//       (static_asserts + a positive-control discriminator).
//
// It touches NOTHING in the matching kernel: it depends only on the self-
// contained header `offline/feature_export_schema_contract_replay.hpp` (zero
// `sim::` symbols).
// =============================================================================

namespace {

using offline::ExportSchema;
using offline::FeatureEvent;
using offline::FeatureExport;
using offline::FeatureRow;
using offline::ObserverConfig;

FeatureEvent ev(std::uint64_t seq, std::uint64_t ts, std::int64_t price) {
    FeatureEvent e;
    e.seq = seq;
    e.ts_ns = ts;
    e.price = price;
    return e;
}

// A hand-built price series with mixed-sign moves.
//   index : 0    1    2    3    4    5
//   seq   : 1    2    3    4    5    6
//   price : 100  101  103  102  105  104
std::vector<FeatureEvent> sample_stream() {
    return {
        ev(1, 1000, 100),
        ev(2, 1100, 101),
        ev(3, 1200, 103),
        ev(4, 1300, 102),
        ev(5, 1400, 105),
        ev(6, 1500, 104),
    };
}

ExportSchema schema_v1() {
    ExportSchema s;
    s.schema_version = 1;
    s.ts_basis = 0;
    s.horizon = 1;
    s.lookback = 1;
    return s;
}

}  // namespace

TEST_CASE("feature-export contract hash is deterministic and order-sensitive") {
    const std::vector<FeatureEvent> a = sample_stream();
    const std::vector<FeatureEvent> b = sample_stream();
    const ExportSchema s = schema_v1();

    const FeatureExport ra = offline::export_features(a, s);
    const FeatureExport rb = offline::export_features(b, s);

    // Identical streams + schema -> identical rows and identical hash; re-run is
    // bit-stable.
    REQUIRE(ra.contract_hash == rb.contract_hash);
    REQUIRE(offline::export_features(a, s).contract_hash == ra.contract_hash);
    REQUIRE(ra.provenance_hash == rb.provenance_hash);
    REQUIRE(ra.rows.size() == rb.rows.size());

    // The read-only wrapper agrees with the free function.
    const offline::FeatureExportSchemaContractReplay replay(a);
    REQUIRE(replay.contract_hash(s) == ra.contract_hash);
    REQUIRE(replay.export_to(s).rows.size() == ra.rows.size());
    REQUIRE(replay.size() == a.size());

    // The empty stream is deterministic and distinct from any populated one.
    const std::vector<FeatureEvent> empty;
    REQUIRE(offline::export_features(empty, s).contract_hash ==
            offline::export_features(empty, s).contract_hash);
    REQUIRE(offline::export_features(empty, s).contract_hash != ra.contract_hash);

    SECTION("reordering two distinct events changes the hash") {
        std::vector<FeatureEvent> reordered = a;
        std::swap(reordered[1], reordered[3]);
        REQUIRE(offline::export_features(reordered, s).contract_hash !=
                ra.contract_hash);
    }

    SECTION("tampering with a price changes the hash") {
        std::vector<FeatureEvent> tampered = a;
        tampered[2].price += 1;
        REQUIRE(offline::export_features(tampered, s).contract_hash !=
                ra.contract_hash);
    }

    SECTION("a different declared schema version changes the contract hash") {
        ExportSchema s2 = s;
        s2.schema_version = 2;
        REQUIRE(offline::export_features(a, s2).contract_hash != ra.contract_hash);
    }

    SECTION("a different horizon changes the contract hash") {
        ExportSchema s3 = s;
        s3.horizon = 2;
        REQUIRE(offline::export_features(a, s3).contract_hash != ra.contract_hash);
    }
}

TEST_CASE("feature-export feature/label computation matches a hand series") {
    const std::vector<FeatureEvent> s = sample_stream();
    const FeatureExport r = offline::export_features(s, schema_v1());

    // lookback=1, horizon=1, n=6: rows for decision indices t=1..5.
    REQUIRE(r.n_rows == 5);
    REQUIRE(r.n_events == 6);

    // t=1: feature=101-100=+1 ; label=103-101=+2 (avail).
    REQUIRE(r.rows[0].event_seq == 2);
    REQUIRE(r.rows[0].feature == 1);
    REQUIRE(r.rows[0].label == 2);
    REQUIRE(r.rows[0].label_available);
    // t=2: feature=103-101=+2 ; label=102-103=-1.
    REQUIRE(r.rows[1].feature == 2);
    REQUIRE(r.rows[1].label == -1);
    // t=3: feature=102-103=-1 ; label=105-102=+3.
    REQUIRE(r.rows[2].feature == -1);
    REQUIRE(r.rows[2].label == 3);
    // t=4: feature=105-102=+3 ; label=104-105=-1.
    REQUIRE(r.rows[3].feature == 3);
    REQUIRE(r.rows[3].label == -1);
    // t=5: feature=104-105=-1 ; label CENSORED (t+1=6 not in stream).
    REQUIRE(r.rows[4].event_seq == 6);
    REQUIRE(r.rows[4].feature == -1);
    REQUIRE_FALSE(r.rows[4].label_available);
    REQUIRE(r.rows[4].label == offline::kCensoredLabel);

    REQUIRE(r.n_available == 4);
    REQUIRE(r.n_censored == 1);
    REQUIRE(r.is_admissible());
}

TEST_CASE("feature-export schema round-trips and detects version mismatch") {
    const std::vector<FeatureEvent> s = sample_stream();
    const FeatureExport ex = offline::export_features(s, schema_v1());

    const std::vector<std::uint8_t> bytes = offline::serialize_export(ex);

    // Round-trip under the matching expected version is the identity.
    const offline::ParseResult pr = offline::parse_export(bytes, /*expected=*/1);
    REQUIRE(pr.ok);
    REQUIRE(pr.schema_matched);
    REQUIRE(pr.value.contract_hash == ex.contract_hash);
    REQUIRE(pr.value.provenance_hash == ex.provenance_hash);
    REQUIRE(pr.value.rows.size() == ex.rows.size());
    REQUIRE(pr.value.n_available == ex.n_available);
    REQUIRE(pr.value.n_censored == ex.n_censored);
    for (std::size_t i = 0; i < ex.rows.size(); ++i) {
        REQUIRE(pr.value.rows[i].event_seq == ex.rows[i].event_seq);
        REQUIRE(pr.value.rows[i].feature == ex.rows[i].feature);
        REQUIRE(pr.value.rows[i].label == ex.rows[i].label);
        REQUIRE(pr.value.rows[i].label_available == ex.rows[i].label_available);
    }

    // The schema-drift falsifier: a reader expecting a DIFFERENT version still
    // parses the well-formed bytes but flags the mismatch.
    const offline::ParseResult mismatch = offline::parse_export(bytes, /*expected=*/2);
    REQUIRE(mismatch.ok);
    REQUIRE_FALSE(mismatch.schema_matched);

    // A tampered byte image (corrupted row payload) fails the recomputed-hash
    // integrity check and is rejected.
    std::vector<std::uint8_t> corrupt = bytes;
    corrupt[8 * 6 + 2] ^= 0xFFu;  // flip a byte inside the first row's payload
    const offline::ParseResult bad = offline::parse_export(corrupt, /*expected=*/1);
    REQUIRE_FALSE(bad.ok);
}

TEST_CASE("feature-export: a read-only observer does not perturb the contract") {
    const std::vector<FeatureEvent> s = sample_stream();
    const ExportSchema sc = schema_v1();

    ObserverConfig off;  // disabled
    ObserverConfig on;
    on.enabled = true;
    on.perturbs = false;
    ObserverConfig leaky;  // POSITIVE CONTROL: writes observer state back
    leaky.enabled = true;
    leaky.perturbs = true;

    const FeatureExport e_off = offline::export_features(s, sc, off);
    const FeatureExport e_on = offline::export_features(s, sc, on);
    const FeatureExport e_leaky = offline::export_features(s, sc, leaky);

    // Observer-on/off equality: read-only observation must NOT change the hash.
    REQUIRE(e_on.contract_hash == e_off.contract_hash);
    REQUIRE(e_on.rows.size() == e_off.rows.size());
    // The observer DID run (it counted rows) yet left the artifact unchanged.
    REQUIRE(e_on.observer_count == e_on.rows.size());
    REQUIRE(e_off.observer_count == 0);

    // Positive control: a perturbing observer that writes its state back DOES
    // change the contract hash -- proving the equality check above is non-vacuous.
    REQUIRE(e_leaky.contract_hash != e_off.contract_hash);
}

TEST_CASE("feature-export point-in-time label flips only when its future arrives") {
    const std::vector<FeatureEvent> s = sample_stream();  // n=6
    const ExportSchema sc = schema_v1();                  // horizon=1
    const FeatureExport before = offline::export_features(s, sc);

    // The last row (decision t=5, seq=6) is CENSORED: its t+1 event is absent.
    const FeatureRow& censored = before.rows.back();
    REQUIRE(censored.event_seq == 6);
    REQUIRE_FALSE(censored.label_available);
    REQUIRE(before.n_censored == 1);

    // Append the future event the censored label depends on.
    std::vector<FeatureEvent> grown = s;
    grown.push_back(ev(7, 1600, 107));  // price[6] = 107
    const FeatureExport after = offline::export_features(grown, sc);

    // The previously-censored row (seq=6) is now AVAILABLE with a real label:
    //   label = price[6] - price[5] = 107 - 104 = +3.
    bool found = false;
    for (const FeatureRow& r : after.rows) {
        if (r.event_seq == 6) {
            REQUIRE(r.label_available);
            REQUIRE(r.label == 3);
            found = true;
        }
    }
    REQUIRE(found);

    // No-leakage invariant: rows whose future was ALREADY present are byte-stable
    // across the append (t=1..4, seq=2..5 unchanged). Only the row whose future
    // newly arrived changed -- a value changes IFF a future event in the suffix
    // makes it definable.
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(after.rows[i].event_seq == before.rows[i].event_seq);
        REQUIRE(after.rows[i].feature == before.rows[i].feature);
        REQUIRE(after.rows[i].label == before.rows[i].label);
        REQUIRE(after.rows[i].label_available == before.rows[i].label_available);
    }

    // The new tail (decision t=6, seq=7) is itself censored (its t+1 is absent),
    // so growth never eliminates the point-in-time frontier.
    REQUIRE(after.rows.back().event_seq == 7);
    REQUIRE_FALSE(after.rows.back().label_available);

    // The contract hash necessarily changed because a label became defined.
    REQUIRE(after.contract_hash != before.contract_hash);
}

namespace {

// Positive control for the non-feedback discriminator: a type that DOES expose a
// matching-feedback member. If `has_matching_feedback` failed to detect this, the
// negative assertions on the offline types would be vacuous.
struct CoupledToMatching {
    void apply_to_matching() {}  // would feed an exported row back into the kernel
};

}  // namespace

TEST_CASE("feature-export diagnostic is read-only with no matching coupling") {
    std::vector<FeatureEvent> s = sample_stream();
    const std::vector<FeatureEvent> before = s;

    const FeatureExport r = offline::export_features(s, schema_v1());
    (void)r;

    // NO mutation of the input stream: export is offline/read-only and cannot
    // alter matching/fills because it never touches engine state.
    REQUIRE(s.size() == before.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        REQUIRE(s[i].seq == before[i].seq);
        REQUIRE(s[i].ts_ns == before[i].ts_ns);
        REQUIRE(s[i].price == before[i].price);
    }

    // The discriminator actually discriminates: a coupled type is detected...
    static_assert(offline::has_matching_feedback_v<CoupledToMatching>,
                  "positive control must be detected as matching-coupled");
    // ...while none of the offline types expose any matching-feedback member.
    static_assert(!offline::has_matching_feedback_v<offline::FeatureEvent>,
                  "FeatureEvent must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FeatureRow>,
                  "FeatureRow must not be a matching input");
    static_assert(!offline::has_matching_feedback_v<offline::FeatureExport>,
                  "FeatureExport must not be a matching input");
    static_assert(
        !offline::has_matching_feedback_v<
            offline::FeatureExportSchemaContractReplay>,
        "FeatureExportSchemaContractReplay must not be a matching input");

    // The export accepts only a const stream and returns a plain value: it has no
    // mutating engine API surface (compile-time evidence of the boundary).
    static_assert(
        std::is_same_v<decltype(offline::export_features(s, schema_v1())),
                       offline::FeatureExport>,
        "export must return a derived artifact, not engine state");
    static_assert(std::is_trivially_copyable_v<offline::FeatureRow>,
                  "FeatureRow must be pure data");
}
