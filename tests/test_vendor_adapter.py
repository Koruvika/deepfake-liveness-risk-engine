import pytest

from fdemo.fusion.vendor_adapter import VendorPayloadError, parse_vendor


def test_v10_session_liveness_persisted_from_liveness_step():
    session = [
        {"externalDatabaseRefID": "u1", "result": {"livenessProven": True}},
        {"externalDatabaseRefID": "u1", "idScanResultsSoFar": {"matchLevel": 7, "matchLevelNFCToFaceMap": 6}},
    ]
    r = parse_vendor(session)
    assert r.liveness_proven and r.match_level == 7 and r.match_level_nfc == 6
    assert r.version == "v10" and r.external_ref_id == "u1"


def test_v10_final_response_alone_is_not_proof_of_liveness():
    r = parse_vendor({"externalDatabaseRefID": "u1", "idScanResultsSoFar": {"matchLevel": 7}})
    assert not r.liveness_proven and r.match_level == 7


def test_v9_idscan_fields():
    r = parse_vendor({"scanResultBlob": "x", "matchLevel": 5, "matchLevelNFCToFaceMap": 6, "externalDatabaseRefID": "u2"})
    assert r.version == "v9" and r.match_level == 5 and r.match_level_nfc == 6 and not r.liveness_proven


def test_mixed_sessions_rejected():
    with pytest.raises(VendorPayloadError):
        parse_vendor([{"externalDatabaseRefID": "a", "result": {"livenessProven": True}},
                      {"externalDatabaseRefID": "b"}])


def test_malformed_payload_rejected():
    with pytest.raises(VendorPayloadError):
        parse_vendor(["not-a-dict"])


def test_session_binding_requires_ref_on_every_response():
    with pytest.raises(VendorPayloadError):
        parse_vendor([{"externalDatabaseRefID": "u1", "result": {"livenessProven": True}},
                      {"idScanResultsSoFar": {"matchLevel": 7}}])


def test_full_id_status_parsed():
    r = parse_vendor({"externalDatabaseRefID": "u1",
                       "idScanResultsSoFar": {"matchLevel": 7, "fullIDStatusEnumInt": 1}})
    assert r.full_id_status == 1


def test_full_id_status_defaults_to_none():
    r = parse_vendor({"externalDatabaseRefID": "u1", "idScanResultsSoFar": {"matchLevel": 7}})
    assert r.full_id_status is None


def test_bool_match_level_not_parsed_as_int():
    r = parse_vendor({"externalDatabaseRefID": "u1", "idScanResultsSoFar": {"matchLevel": True}})
    assert r.match_level is None


def test_v9_top_level_liveness_proven_not_trusted():
    r = parse_vendor({"scanResultBlob": "x", "matchLevel": 5, "livenessProven": True,
                       "externalDatabaseRefID": "u2"})
    assert not r.liveness_proven and r.version == "v9" and r.match_level == 5


def test_later_liveness_failure_overrides_earlier_success():
    session = [
        {"externalDatabaseRefID": "u1", "result": {"livenessProven": True}},
        {"externalDatabaseRefID": "u1", "result": {"livenessProven": False}},
    ]
    r = parse_vendor(session)
    assert not r.liveness_proven


def test_expected_ref_accepted_when_all_responses_match():
    session = [{"externalDatabaseRefID": "u1", "result": {"livenessProven": True}}]
    r = parse_vendor(session, expected_ref="u1")
    assert r.liveness_proven and r.external_ref_id == "u1"


def test_expected_ref_rejected_when_mismatched():
    session = [{"externalDatabaseRefID": "u1", "result": {"livenessProven": True}}]
    with pytest.raises(VendorPayloadError):
        parse_vendor(session, expected_ref="other")


def test_expected_ref_rejected_when_missing_on_a_response():
    session = [{"externalDatabaseRefID": "u1", "result": {"livenessProven": True}},
               {"idScanResultsSoFar": {"matchLevel": 7}}]
    with pytest.raises(VendorPayloadError):
        parse_vendor(session, expected_ref="u1")
