# GUARD: Governance-Unified Autonomous Agent Runtime

This repository contains the official implementation for the paper:

> **GUARD: A Governance-Unified Autonomous Agent Runtime for Compliant 
> Enterprise Data Analytics — A Case Study in Industrial Fault Detection**  
> Ronghua Gai¹, Qi Hu²  
> ¹ University of Chicago  ² Northeastern University  
> Submitted to *Expert Systems with Applications*, 2026

## Framework Overview

GUARD embeds five tightly coupled governance components into the AI agent execution loop:

| Component | Role | EU AI Act |
|-----------|------|-----------|
| PEL — Policy Enforcement Layer | Real-time rule enforcement | Art. 9 |
| AAL — Agentic Action Ledger | Tamper-evident audit chain | Art. 12 |
| BAS — Bounded Autonomy Scheduler | Human escalation control | Art. 14 |
| AM — Alignment Monitor | Distributional drift detection | Art. 9(4) |
| CSCG — Cross-System Context Graph | Provenance tracing | Art. 12(1) |

## Quick Start

```bash
pip install -r requirements.txt
python data_loader.py        # Download & preprocess CWRU + AI4I datasets
python run_experiment.py     # Run all experiments (~15 min, CPU only)
python plot_results.py       # Generate paper figures
```

## Datasets

- **CWRU Bearing**: https://engineering.case.edu/bearingdatacenter
- **AI4I 2020**: https://archive.ics.uci.edu/dataset/601

## Results

| Method | AUROC | Audit Completeness | Provenance Coverage |
|--------|-------|--------------------|---------------------|
| GBA (baseline) | 0.871 | 0.000 | 0.000 |
| GUARD (ours) | **0.941** | **0.987** | **0.963** |

## License

MIT License © 2026 Ronghua Gai
