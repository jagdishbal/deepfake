"""
Audit logging service to record model versions, thresholds, routing decisions,
and confidence metrics without altering the MySQL database schema.
"""

import json
from datetime import datetime
from pathlib import Path

from config import RUNTIME_DIR

AUDIT_LOG_PATH = Path(RUNTIME_DIR) / "prediction_audit.jsonl"
_audit_cache: dict[int, dict] = {}
_last_mtime: float = 0.0


def log_prediction_audit(
    prediction_id: int,
    user_id: int,
    filename: str,
    file_type: str,
    prediction: str,
    confidence: float,
    model_used: str,
    model_version: str,
    threshold: float,
    routing_reason: str,
    probability_real: float | None = None,
    probability_fake: float | None = None,
    is_inconclusive: bool = False,
) -> None:
    """Appends structured audit details for a prediction."""
    entry = {
        "prediction_id": prediction_id,
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "prediction": prediction,
        "confidence": confidence,
        "model_used": model_used,
        "model_version": model_version,
        "threshold": threshold,
        "routing_reason": routing_reason,
        "probability_real": probability_real,
        "probability_fake": probability_fake if probability_fake is not None else (round(1.0 - probability_real, 4) if probability_real is not None else None),
        "is_inconclusive": is_inconclusive,
        "timestamp": datetime.now().isoformat()
    }
    _audit_cache[prediction_id] = entry
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Never break production flow if audit log write fails
        pass


def get_prediction_audit(prediction_id: int) -> dict | None:
    """Retrieves audit details for a specific prediction ID by reading latest entries."""
    if prediction_id in _audit_cache:
        return _audit_cache[prediction_id]

    if not AUDIT_LOG_PATH.is_file():
        return None

    global _last_mtime
    try:
        mtime = AUDIT_LOG_PATH.stat().st_mtime
        if mtime != _last_mtime or prediction_id not in _audit_cache:
            with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pid = data.get("prediction_id")
                        if pid is not None:
                            _audit_cache[int(pid)] = data
                    except Exception:
                        continue
            _last_mtime = mtime
        return _audit_cache.get(prediction_id)
    except Exception:
        pass
    return None
