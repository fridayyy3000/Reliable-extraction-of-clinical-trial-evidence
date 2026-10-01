# Reliable Extraction of Clinical-Trial Evidence

**Paper:** “Reliable extraction of clinical-trial evidence with provenance, verification, and targeted review”

EviSearch builds structured clinical-trial evidence tables from research papers. Agent A extracts from the paper, Agent B searches pages, and reconciliation adjudicates their answers. An attribution verifier checks cited evidence; review flags route selected cells to human review.

## This Copy

This staging folder contains pipeline and analysis source, requirements, prompt/rubric source, the locked mHSPC schema, frozen knowledge notes, static GitHub Pages files, and an **exact copy of the repository's `dataset/` directory**.

The dataset copy has 30 files, including all 26 source PDFs. The source/staging files were verified identical. The repository owner has confirmed that redistribution rights for these PDFs are cleared. Structured gold annotations and extracted quotations may have separate terms and should be reviewed before release.

Not included: generated inference/results, final run manifests, scoring labels or queues, raw model responses, overnight/server logs, caches, embeddings, feedback, tests, the Flask web demo, and deployment configuration.

## Pipeline

- **B1:** single-pass extraction from parsed markdown
- **Agent A:** PDF Query Agent
- **Agent B:** Search Agent
- **Reconciliation:** independent reading of contested cells followed by adjudication
- **Attribution verification:** a separate reader checks claims against cited pages
- **Targeted review:** selected cells are flagged for human review

The journal benchmark covers 10 papers × 133 fields = 1,330 cells per system. Models:

- `Qwen/Qwen3.6-27B`
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `google/gemma-4-31B-it`
- `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`

Retrieval uses `Qwen/Qwen3-Embedding-8B`, top-k 5, with no reranker.

## Setup and Use

Requires Python 3.11 or compatible. The environment template contains placeholders only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Cloud inference requires provider credentials; Vertex uses Application Default Credentials. New PDF parsing requires `VISION_AGENT_API_KEY`. Local vLLM requires model checkpoints and the corresponding `EVISEARCH_MODEL_PATH_*` settings. Never commit a populated `.env` or credential file.

Inspect the actual pipeline interfaces before running model-backed commands:

```bash
python3 experiment-scripts/run_benchmark.py --help
python3 experiment-scripts/run_search_agent.py --help
```

The fixed scorer is `experiment-scripts/score_e1_vertex.py`; real scoring invokes Vertex. Its `--help` is safe to inspect. Saved result and label artifacts are not included in this staging folder.

## Analysis and Prompts

Deterministic paper analysis is in `experiment-analysis/`. The paper-clustered bootstrap script is `analysis/journal/run_clinical_bootstrap_ci.py`; it requires the validated canonical CSV, which is a generated artifact and is not included here.

Prompt definitions are in `src/evisearch/services/`; the scoring rubric is `experiment-scripts/scoring/RUBRIC.md`. Run IDs and model settings for the completed journal experiments are documented in the parent repository's `analysis/journal/` records; generated manifests are intentionally not copied here.

## Citation and License

**Citation:** “Reliable extraction of clinical-trial evidence with provenance, verification, and targeted review.” Author list, venue, DOI, and publication metadata are pending maintainer input.

**License:** No license is included. A maintainer must select and add an appropriate code/data license before redistribution. PDF redistribution has been confirmed by the repository owner; structured data terms still require review.
