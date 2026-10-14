"""Normalise FaceTec Server responses into a VendorResult.

v10 (/process-request): `result.livenessProven` appears only in the liveness step's response, so the caller
passes the whole session (list of responses) and liveness is persisted across steps.
v9 (/match-3d-2d-idscan): `matchLevel`, `matchLevelNFCToFaceMap`, `externalDatabaseRefID`, `scanResultBlob`.
Flags we could not verify against FaceTec documentation (e.g. `wasProcessed`) are deliberately NOT trusted:
without `livenessProven` the session counts as not proven.

v9 support is match-level only: we could not verify where (if anywhere) v9 responses carry a liveness
verdict, so v9 payloads never set `liveness_proven`. Liveness can only come from a v10 `result.livenessProven`
field of a session step -- a top-level `livenessProven` key (as might appear on a v9-shaped payload) is
ignored on purpose.

Session binding: once any response in a session carries `externalDatabaseRefID`, every response in that
session must carry it (and the same value) -- a response without the ref id is rejected rather than silently
treated as belonging to the session. When the caller supplies `expected_ref` (e.g. the server-issued reference
for the transaction being authorised), every response must additionally carry that exact value -- this is what
lets the caller bind the session to a specific user/transaction instead of merely checking internal consistency.

Liveness proof is not permanent for the session: a response with `result.livenessProven is False` overrides any
earlier `True` in the same session (e.g. a retry after a failed liveness check) -- the session as a whole is
only proven if at least one response proved it and none later disproved it.
"""
from dataclasses import dataclass


class VendorPayloadError(ValueError):
    pass


@dataclass(frozen=True)
class VendorResult:
    liveness_proven: bool
    match_level: int | None
    match_level_nfc: int | None
    external_ref_id: str | None
    version: str  # "v10" | "v9" | "unknown" | "sim"
    full_id_status: int | None = None


def _find(d, key):
    if key in d:
        return d[key]
    for sub in ("result", "idScanResultsSoFar"):
        if isinstance(d.get(sub), dict) and key in d[sub]:
            return d[sub][key]
    return None


def _as_int(v):
    """Strict int parsing: reject bool (a subclass of int in Python) and anything else non-int."""
    return v if type(v) is int else None


def parse_vendor(payload, expected_ref=None) -> VendorResult:
    responses = payload if isinstance(payload, list) else [payload]
    if not responses or not all(isinstance(r, dict) for r in responses):
        raise VendorPayloadError("expected a JSON object or a list of objects (one per FaceTec session step)")
    raw_refs = [_find(r, "externalDatabaseRefID") for r in responses]
    present = [r for r in raw_refs if r]
    if present and len(present) != len(raw_refs):
        raise VendorPayloadError("response without externalDatabaseRefID in a bound session")
    if len(set(present)) > 1:
        raise VendorPayloadError(f"responses belong to different sessions: {sorted(set(present))}")
    if expected_ref is not None and any(r != expected_ref for r in raw_refs):
        raise VendorPayloadError(f"response does not carry the expected externalDatabaseRefID {expected_ref!r}")
    # liveness only ever comes from a v10 `result.livenessProven`; v9 has no verified liveness field. A later
    # explicit False overrides an earlier True (e.g. a retried liveness step that then failed).
    proven_true = any(isinstance(r.get("result"), dict) and r["result"].get("livenessProven") is True
                       for r in responses)
    proven_false = any(isinstance(r.get("result"), dict) and r["result"].get("livenessProven") is False
                        for r in responses)
    proven = proven_true and not proven_false
    levels = [lv for lv in (_as_int(_find(r, "matchLevel")) for r in responses) if lv is not None]
    nfc = [lv for lv in (_as_int(_find(r, "matchLevelNFCToFaceMap")) for r in responses) if lv is not None]
    full_id = [v for v in (_as_int(_find(r, "fullIDStatusEnumInt")) for r in responses) if v is not None]
    is_v10 = any(isinstance(r.get("result"), dict) or "idScanResultsSoFar" in r or "responseBlob" in r
                 for r in responses)
    is_v9 = any("scanResultBlob" in r for r in responses)
    return VendorResult(
        liveness_proven=proven,
        match_level=levels[-1] if levels else None,
        match_level_nfc=nfc[-1] if nfc else None,
        external_ref_id=present[0] if present else None,
        version="v9" if is_v9 else ("v10" if is_v10 else "unknown"),
        full_id_status=full_id[-1] if full_id else None,
    )
