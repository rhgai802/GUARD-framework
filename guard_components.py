"""
guard_components.py  —  Five GUARD components
  PEL  : Policy Enforcement Layer
  AAL  : Agentic Action Ledger
  BAS  : Bounded Autonomy Scheduler
  AM   : Alignment Monitor
  CSCG : Cross-System Context Graph
"""
import hashlib, json, time, uuid
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ── 1. Policy Enforcement Layer ──────────────────────────────────────
class PolicyEnforcementLayer:
    """Real-time rule enforcement before agent action execution."""

    DEFAULT_POLICIES = {
        "max_confidence_threshold": 0.50,
        "forbidden_actions": ["delete_all", "override_safety"],
        "require_human_review_above_risk": 0.80,
        "max_batch_size": 512,
        "allowed_data_sources": ["cwru", "ai4i", "enterprise_db"],
    }

    def __init__(self, policies: Optional[Dict] = None):
        self.policies = policies or self.DEFAULT_POLICIES
        self.violation_log: List[Dict] = []

    def evaluate(self, action: str, context: Dict) -> Tuple[bool, str, float]:
        """Returns (allowed, reason, risk_score)."""
        risk = self._compute_risk(action, context)
        if action in self.policies["forbidden_actions"]:
            self._log_violation(action, "FORBIDDEN_ACTION", risk)
            return False, "Action explicitly forbidden by policy", risk
        if context.get("confidence", 1.0) < self.policies["max_confidence_threshold"]:
            self._log_violation(action, "LOW_CONFIDENCE", risk)
            return False, f"Confidence below threshold {self.policies['max_confidence_threshold']}", risk
        if risk > self.policies["require_human_review_above_risk"]:
            self._log_violation(action, "HIGH_RISK", risk)
            return False, f"Risk score {risk:.3f} requires human review", risk
        return True, "Policy check passed", risk

    def _compute_risk(self, action: str, context: Dict) -> float:
        base = 0.1
        if "production" in str(context.get("environment", "")):
            base += 0.3
        if context.get("anomaly_score", 0) > 0.7:
            base += 0.25
        if context.get("batch_size", 0) > self.policies["max_batch_size"]:
            base += 0.2
        return min(base, 1.0)

    def _log_violation(self, action, reason, risk):
        self.violation_log.append({"action": action, "reason": reason,
                                    "risk": risk, "ts": time.time()})

    @property
    def violation_rate(self):
        return len(self.violation_log)


# ── 2. Agentic Action Ledger ─────────────────────────────────────────
class AgenticActionLedger:
    """Tamper-evident, hash-chained audit log of every agent decision."""

    def __init__(self):
        self.chain: List[Dict] = []
        self._prev_hash = "GENESIS"

    def record(self, action: str, decision: Any, context: Dict,
               policy_result: Tuple, risk_score: float) -> str:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "action": action,
            "decision": str(decision),
            "context_hash": hashlib.sha256(json.dumps(context, sort_keys=True,
                                            default=str).encode()).hexdigest()[:16],
            "policy_allowed": policy_result[0],
            "policy_reason": policy_result[1],
            "risk_score": round(risk_score, 4),
            "prev_hash": self._prev_hash,
        }
        entry["hash"] = self._compute_hash(entry)
        self._prev_hash = entry["hash"]
        self.chain.append(entry)
        return entry["id"]

    def _compute_hash(self, entry: Dict) -> str:
        payload = json.dumps({k: v for k, v in entry.items() if k != "hash"},
                              sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:32]

    def verify_integrity(self) -> Tuple[bool, int]:
        """Verify chain integrity. Returns (valid, n_verified)."""
        prev = "GENESIS"
        for i, entry in enumerate(self.chain):
            if entry["prev_hash"] != prev:
                return False, i
            prev = entry["hash"]
        return True, len(self.chain)

    @property
    def completeness(self) -> float:
        if not self.chain:
            return 1.0
        valid, n = self.verify_integrity()
        return n / len(self.chain)


# ── 3. Bounded Autonomy Scheduler ───────────────────────────────────
class BoundedAutonomyScheduler:
    """Controls when agent acts autonomously vs escalates to human."""

    def __init__(self, auto_threshold=0.65, human_threshold=0.85,
                 max_autonomous_streak=50):
        self.auto_threshold = auto_threshold
        self.human_threshold = human_threshold
        self.max_streak = max_autonomous_streak
        self.autonomous_streak = 0
        self.escalation_count = 0
        self.total_decisions = 0

    def decide_mode(self, confidence: float, risk_score: float) -> str:
        """Returns 'autonomous', 'assisted', or 'human_required'."""
        self.total_decisions += 1
        combined = 0.6 * confidence + 0.4 * (1 - risk_score)
        if self.autonomous_streak >= self.max_streak:
            self.autonomous_streak = 0
            self.escalation_count += 1
            return "human_required"
        if combined >= self.auto_threshold and risk_score < self.human_threshold:
            self.autonomous_streak += 1
            return "autonomous"
        if risk_score >= self.human_threshold:
            self.autonomous_streak = 0
            self.escalation_count += 1
            return "human_required"
        self.autonomous_streak = 0
        return "assisted"

    @property
    def autonomy_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return 1.0 - (self.escalation_count / self.total_decisions)


# ── 4. Alignment Monitor ─────────────────────────────────────────────
class AlignmentMonitor:
    """Detects distributional drift between training and inference data."""

    def __init__(self, window_size=100, drift_threshold=0.15):
        self.window = deque(maxlen=window_size)
        self.reference_stats: Optional[Dict] = None
        self.drift_threshold = drift_threshold
        self.drift_events: List[Dict] = []

    def set_reference(self, X: np.ndarray):
        self.reference_stats = {
            "mean": X.mean(axis=0),
            "std":  X.std(axis=0) + 1e-9,
        }

    def observe(self, x: np.ndarray) -> Tuple[float, bool]:
        """Returns (drift_score, is_drifting)."""
        self.window.append(x)
        if self.reference_stats is None or len(self.window) < 20:
            return 0.0, False
        recent = np.vstack(self.window)
        z_score = np.abs((recent.mean(axis=0) - self.reference_stats["mean"])
                         / self.reference_stats["std"])
        drift_score = float(z_score.mean())
        is_drifting = drift_score > self.drift_threshold
        if is_drifting:
            self.drift_events.append({"score": drift_score, "ts": time.time()})
        return drift_score, is_drifting

    @property
    def drift_rate(self) -> float:
        if not self.window:
            return 0.0
        return len(self.drift_events) / max(len(self.window), 1)


# ── 5. Cross-System Context Graph ────────────────────────────────────
class CrossSystemContextGraph:
    """Lightweight provenance graph linking decisions to data sources."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, str]] = []

    def add_data_source(self, source_id: str, meta: Dict):
        self.nodes[source_id] = {"type": "data_source", **meta}

    def add_decision(self, decision_id: str, source_ids: List[str],
                     action: str, confidence: float):
        self.nodes[decision_id] = {
            "type": "decision", "action": action,
            "confidence": confidence, "ts": time.time()
        }
        for sid in source_ids:
            self.edges.append((sid, decision_id, "derived_from"))

    def get_provenance(self, decision_id: str) -> List[str]:
        return [src for src, dst, _ in self.edges if dst == decision_id]

    @property
    def provenance_coverage(self) -> float:
        decisions = [n for n, d in self.nodes.items() if d["type"] == "decision"]
        if not decisions:
            return 1.0
        traced = sum(1 for d in decisions if self.get_provenance(d))
        return traced / len(decisions)
