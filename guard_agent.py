"""
guard_agent.py  —  GUARD Agent integrating all five components
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Tuple
from guard_components import (PolicyEnforcementLayer, AgenticActionLedger,
                               BoundedAutonomyScheduler, AlignmentMonitor,
                               CrossSystemContextGraph)

class GUARDAgent:
    """
    Governance-Unified Autonomous Agent Runtime.
    Wraps a RandomForest classifier with five governance components.
    """

    def __init__(self, n_estimators=200, random_state=42, **policy_kwargs):
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          random_state=random_state)
        self.pel  = PolicyEnforcementLayer(policy_kwargs.get("policies"))
        self.aal  = AgenticActionLedger()
        self.bas  = BoundedAutonomyScheduler(
            auto_threshold=policy_kwargs.get("auto_threshold", 0.65),
            human_threshold=policy_kwargs.get("human_threshold", 0.85))
        self.am   = AlignmentMonitor(
            drift_threshold=policy_kwargs.get("drift_threshold", 0.15))
        self.cscg = CrossSystemContextGraph()
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, source_id="training_data"):
        self.clf.fit(X, y)
        self.am.set_reference(X)
        self.cscg.add_data_source(source_id, {"n_samples": len(X),
                                               "n_features": X.shape[1]})
        self._fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray, source_id="inference",
                environment="production") -> Dict:
        assert self._fitted, "Call fit() first."
        proba = self.clf.predict_proba(X)
        preds = proba.argmax(axis=1)
        confidences = proba.max(axis=1)

        results = []
        for i, (x, pred, conf) in enumerate(zip(X, preds, confidences)):
            drift_score, is_drifting = self.am.observe(x)
            context = {"confidence": float(conf), "anomaly_score": drift_score,
                       "environment": environment, "sample_idx": i,
                       "batch_size": len(X)}
            allowed, reason, risk = self.pel.evaluate("classify", context)
            mode = self.bas.decide_mode(float(conf), risk)
            rec_id = self.aal.record("classify", int(pred), context,
                                      (allowed, reason), risk)
            self.cscg.add_decision(rec_id, [source_id], "classify", float(conf))
            results.append({
                "prediction": int(pred) if allowed else -1,
                "confidence": float(conf),
                "policy_allowed": allowed,
                "mode": mode,
                "risk_score": risk,
                "drift_score": drift_score,
                "audit_id": rec_id,
            })

        valid_preds = [r["prediction"] for r in results if r["policy_allowed"]]
        return {
            "predictions": [r["prediction"] for r in results],
            "results": results,
            "metrics": self._governance_metrics(),
        }

    # ── Governance Metrics ────────────────────────────────────────────
    def _governance_metrics(self) -> Dict:
        chain_ok, n_verified = self.aal.verify_integrity()
        return {
            "audit_completeness": self.aal.completeness,
            "chain_integrity":    chain_ok,
            "provenance_coverage": self.cscg.provenance_coverage,
            "autonomy_rate":      self.bas.autonomy_rate,
            "drift_rate":         self.am.drift_rate,
            "violation_count":    self.pel.violation_rate,
            "n_decisions":        len(self.aal.chain),
        }
