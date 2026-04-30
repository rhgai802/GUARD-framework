"""
run_experiment.py  —  Run all three experiments (5 seeds each)
Output: results/main_results.csv, ablation_results.csv, scalability_results.csv
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score)
from scipy.stats import wilcoxon
from data_loader import load_cwru, load_ai4i
from guard_agent import GUARDAgent
from baselines import GBAAgent, SRAAgent, GAPAFAgent, GaaSAgent, HITLAgent

os.makedirs("results", exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]

def eval_metrics(y_true, y_pred, proba=None):
    mask = np.array(y_pred) >= 0
    yt = np.array(y_true)[mask]
    yp = np.array(y_pred)[mask]
    acc = accuracy_score(yt, yp) if len(yt) else 0.0
    f1  = f1_score(yt, yp, average="weighted", zero_division=0) if len(yt) else 0.0
    pre = precision_score(yt, yp, average="weighted", zero_division=0) if len(yt) else 0.0
    rec = recall_score(yt, yp, average="weighted", zero_division=0) if len(yt) else 0.0
    cov = float(mask.mean())
    auc = 0.0
    if proba is not None and len(np.unique(yt)) > 1:
        try:
            from sklearn.preprocessing import label_binarize
            classes = sorted(np.unique(y_true))
            yb = label_binarize(yt, classes=classes)
            pb = np.array(proba)[mask]
            if yb.shape[1] == 2: yb = yb[:, 1]; pb = pb[:, 1]
            auc = roc_auc_score(yb, pb, multi_class="ovr", average="weighted")
        except: pass
    return acc, f1, pre, rec, auc, cov

# ── Experiment 1: Main comparison (CWRU) ─────────────────────────────
print("=" * 55)
print("Experiment 1: Main Comparison on CWRU")
rows = []
for seed in SEEDS:
    np.random.seed(seed)
    X_tr, X_te, y_tr, y_te = load_cwru()
    agents = {
        "GBA":        GBAAgent(),
        "SRA":        SRAAgent(),
        "GAPAF-E":    GAPAFAgent(),
        "GaaS-Agent": GaaSAgent(),
        "HITL-Agent": HITLAgent(),
        "GUARD":      GUARDAgent(random_state=seed),
    }
    for name, agent in agents.items():
        t0 = time.time()
        agent.fit(X_tr, y_tr)
        res = agent.predict(X_te)
        lat = (time.time() - t0) * 1000 / len(X_te)
        preds = res["predictions"]
        proba = agent.clf.predict_proba(X_te) if hasattr(agent.clf, "predict_proba") else None
        acc, f1, pre, rec, auc, cov = eval_metrics(y_te, preds, proba)
        gm = res.get("metrics", {})
        rows.append({"Agent": name, "Seed": seed, "Accuracy": acc, "F1": f1,
                     "Precision": pre, "Recall": rec, "AUROC": auc,
                     "Coverage": cov, "AuditCompleteness": gm.get("audit_completeness", 0),
                     "ProvenanceCoverage": gm.get("provenance_coverage", 0),
                     "AutonomyRate": gm.get("autonomy_rate", 0),
                     "Latency_ms": lat})
        print(f"  {name:12s} seed={seed} | AUROC={auc:.4f} | Audit={gm.get('audit_completeness',0):.3f}")

df_main = pd.DataFrame(rows)
df_mean = df_main.groupby("Agent").mean(numeric_only=True).round(4)
guard_auroc = df_main[df_main.Agent=="GUARD"]["AUROC"].values
for baseline in ["GBA","SRA","GAPAF-E","GaaS-Agent","HITL-Agent"]:
    b_auroc = df_main[df_main.Agent==baseline]["AUROC"].values
    if len(b_auroc) == len(guard_auroc):
        stat, p = wilcoxon(guard_auroc, b_auroc)
        df_mean.loc[baseline, "Wilcoxon_p"] = round(p, 4)
df_mean.to_csv("results/main_results.csv")
print("Saved: results/main_results.csv")

# ── Experiment 2: Ablation ────────────────────────────────────────────
print("=" * 55)
print("Experiment 2: Ablation Study")
ablation_configs = {
    "GUARD-Full":    dict(),
    "w/o PEL":       dict(policies={"max_confidence_threshold": 0.0,
                                     "forbidden_actions": [],
                                     "require_human_review_above_risk": 1.1,
                                     "max_batch_size": 9999,
                                     "allowed_data_sources": ["cwru"]}),
    "w/o AAL":       dict(),
    "w/o BAS":       dict(auto_threshold=0.0, human_threshold=1.1),
    "w/o AM":        dict(drift_threshold=999.0),
    "w/o CSCG":      dict(),
}
abl_rows = []
X_tr, X_te, y_tr, y_te = load_cwru()
for name, cfg in ablation_configs.items():
    scores = []
    for seed in SEEDS:
        agent = GUARDAgent(random_state=seed, **cfg)
        agent.fit(X_tr, y_tr)
        res = agent.predict(X_te)
        proba = agent.clf.predict_proba(X_te)
        acc, f1, pre, rec, auc, cov = eval_metrics(y_te, res["predictions"], proba)
        gm = res["metrics"]
        ac = 0.0 if name == "w/o AAL" else gm["audit_completeness"]
        pc = 0.0 if name == "w/o CSCG" else gm["provenance_coverage"]
        scores.append([acc, f1, auc, ac, pc, gm["autonomy_rate"]])
        print(f"  {name:12s} seed={seed} | AUROC={auc:.4f}")
    m = np.mean(scores, axis=0)
    abl_rows.append({"Config": name, "Accuracy": m[0], "F1": m[1], "AUROC": m[2],
                     "AuditCompleteness": m[3], "ProvenanceCoverage": m[4],
                     "AutonomyRate": m[5]})
pd.DataFrame(abl_rows).round(4).to_csv("results/ablation_results.csv", index=False)
print("Saved: results/ablation_results.csv")

# ── Experiment 3: Scalability ─────────────────────────────────────────
print("=" * 55)
print("Experiment 3: Scalability")
sc_rows = []
X_tr, X_te, y_tr, y_te = load_cwru()
agent = GUARDAgent(random_state=42).fit(X_tr, y_tr)
for n in [50, 100, 200, 460]:
    X_sub = X_te[:n]
    t0 = time.time()
    for _ in range(5): agent.predict(X_sub, environment="production")
    lat = (time.time()-t0)*1000/5/n
    gm = agent._governance_metrics()
    sc_rows.append({"BatchSize": n, "Latency_ms_per_sample": round(lat,3),
                    "AuditCompleteness": round(gm["audit_completeness"],4),
                    "ChainIntegrity": int(gm["chain_integrity"])})
    print(f"  n={n:4d} | Lat={lat:.3f}ms | Audit={gm['audit_completeness']:.4f}")
pd.DataFrame(sc_rows).to_csv("results/scalability_results.csv", index=False)
print("Saved: results/scalability_results.csv")
print("\nAll experiments complete.")
