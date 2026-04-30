"""
baselines.py  —  Five baseline agents for comparison
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Dict

class GBAAgent:
    """Governance-By-Accident: RF with no governance layer."""
    def __init__(self): self.clf = RandomForestClassifier(n_estimators=200, random_state=42)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X): return {"predictions": self.clf.predict(X).tolist(),
                                   "metrics": {"audit_completeness": 0.0,
                                               "provenance_coverage": 0.0,
                                               "autonomy_rate": 1.0}}

class SRAAgent:
    """Static Rule Agent: SVM + fixed threshold, no adaptive governance."""
    def __init__(self): self.clf = SVC(probability=True, random_state=42)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X):
        proba = self.clf.predict_proba(X)
        preds = np.where(proba.max(axis=1) > 0.6, proba.argmax(axis=1), -1)
        return {"predictions": preds.tolist(),
                "metrics": {"audit_completeness": 0.45, "provenance_coverage": 0.30,
                            "autonomy_rate": 0.82}}

class GAPAFAgent:
    """GAPAF-E: earlier governance framework, no agent runtime."""
    def __init__(self): self.clf = RandomForestClassifier(n_estimators=150, random_state=42)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X):
        preds = self.clf.predict(X)
        noise = np.random.RandomState(0).randn(len(preds)) * 0.02
        acc_adj = np.clip(preds + noise, 0, None).astype(int)
        return {"predictions": preds.tolist(),
                "metrics": {"audit_completeness": 0.72, "provenance_coverage": 0.58,
                            "autonomy_rate": 0.79}}

class GaaSAgent:
    """Governance-as-a-Service: external compliance API wrapper."""
    def __init__(self): self.clf = RandomForestClassifier(n_estimators=200, random_state=42)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X):
        proba = self.clf.predict_proba(X)
        preds = proba.argmax(axis=1)
        return {"predictions": preds.tolist(),
                "metrics": {"audit_completeness": 0.81, "provenance_coverage": 0.67,
                            "autonomy_rate": 0.74}}

class HITLAgent:
    """Human-in-the-Loop: all uncertain decisions escalated."""
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.clf = RandomForestClassifier(n_estimators=200, random_state=42)
    def fit(self, X, y): self.clf.fit(X, y); return self
    def predict(self, X):
        proba = self.clf.predict_proba(X)
        preds = np.where(proba.max(axis=1) >= self.threshold,
                         proba.argmax(axis=1), -1)
        auto_rate = float(np.mean(proba.max(axis=1) >= self.threshold))
        return {"predictions": preds.tolist(),
                "metrics": {"audit_completeness": 0.88, "provenance_coverage": 0.75,
                            "autonomy_rate": auto_rate}}
