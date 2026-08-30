# Portfolio Project Checklist — Clinical Text Simplifier

Every project must check every box before it counts as "portfolio-ready." Derived
from the ML-portfolio hiring guidelines discussed in
`Desktop/PORTFOLIO-PROJECT-PLANS/portfolio_rubric_and_roadmap.md`. Checked only
where actually verified, not aspirationally.

## 1. End-to-End Implementation
- [x] Data preprocessing pipeline present and documented — `src/data_prep.py` (deterministic, seed=13, topic-level split)
- [x] Feature engineering step present and documented — N/A in the classic tabular sense; the shared prompt builder (`src/prompting.py`) is the equivalent single-source-of-truth transform for this task
- [x] Training pipeline present — `src/train_lora.py`, LoRA via `peft`, real 800-example/59-minute CPU run
- [x] Evaluation step present, with real metrics — `src/evaluate_model.py`, SARI/sacreBLEU/ROUGE-L/FK grade on 200 held-out PLABA sentences, `reports/eval_results.json`
- [x] Deployed: runs somewhere beyond a notebook — `app/streamlit_app.py`, Dockerized, verified locally end-to-end (model load, adapter toggle, generation, live and cached-example paths)
- [x] Monitoring present, or at minimum a stated monitoring plan — README "Monitoring" section

## 2. Reproducibility
- [x] Clean, structured codebase (not notebooks-only)
- [x] `requirements.txt` included
- [x] A stranger can clone and run it without DMing the author — LoRA adapter committed directly to the repo (~20MB), base model auto-downloads from the HF Hub; no account/API key needed for the demo

## 3. Business Relevance
- [x] Tied to a practical/business outcome, not just a metric — README "Business relevance" section
- [x] That outcome framing is stated explicitly in the README

## 4. Scalability Awareness
- [x] Latency, cost, and infrastructure constraints are discussed — README "Scalability considerations", grounded in the real throughput benchmark that sized the training run
- [x] A stated plan for how it would scale beyond the demo

## 5. Documentation & Storytelling
- [x] README has a problem statement
- [x] README has an architecture diagram — Mermaid diagram, renders natively on GitHub
- [x] README has results — real numbers, both a clear win and two documented failure modes shown honestly
- [x] README has "how to run" instructions — demo (local + Docker) and full training reproduction, separated
- [ ] (Bonus) a blog post or demo video — not done, optional

## 6. Modern Stack
- [x] Docker — `Dockerfile`, builds the demo app with the committed adapter baked in
- [ ] Experiment tracking (e.g. MLflow) — `reports/eval_results.json` + `case_study.md` is the closest analogue for a single documented run; not MLflow-equivalent tracking over time, left unchecked rather than force-fit
- [x] Other tools where relevant — Hugging Face `transformers`/`peft`/`accelerate`/`evaluate`, `textstat`, real SARI/sacreBLEU/ROUGE scoring

## 7. Recruiter-Scan Signals
- [x] README is understandable in 30 seconds
- [x] Commit history reads as real work over time — training, eval, case study, and demo/deployment as separate documented stages
- [ ] Live deployment link — Docker + local run verified; not yet actually deployed to a public host (Streamlit Community Cloud) — needs that one manual step, see README
- [x] Architecture diagram a non-technical reviewer can follow

## 8. Elevation Pass
- [x] Data augmentation/preprocessing pipeline — `src/data_prep.py`
- [ ] MLOps practices: model versioning, experiment tracking — N/A at this scope; one adapter, one documented run, not force-checked
- [ ] Deployed as a REST API (Flask/FastAPI) — N/A by design; this project's interface is the interactive demo, not a service API
- [x] Demo app (Streamlit/Gradio) — `app/streamlit_app.py`, side-by-side baseline vs. fine-tuned, cached real examples plus live generation on user-entered text
- [x] Written explanation of a real-world application of the system — README "Business relevance"

## 9. Forward-Looking Signals (2025+)
- [x] Any LLM use is responsible and cost-aware, stated as such — README states the CPU-only, single-model-instance design explicitly; the app's own disclaimer states known failure modes rather than hiding them
- [x] Bias/explainability/compliance trade-offs considered and stated where relevant — the case study's invented-number and dropped-content failures are surfaced directly in the demo UI, not just buried in a report
- [ ] Multimodal awareness noted where relevant — N/A, text-only, not claiming otherwise
- [x] Project is treated as living — research phase (case study) and production phase (demo/Docker) are both present and dated as separate stages of work
