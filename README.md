## How Far Are VLMs from Privacy Awareness in the Physical World? An Empirical Study

<a href="https://arxiv.org/abs/2605.05340" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ImmersedPrivacy-red?logo=arxiv&style=for-the-badge" />
</a>
<a href="https://immersed-privacy.github.io" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Homepage-blue.svg?style=for-the-badge" />
</a>
<a href="https://github.com/immersed-privacy/immersed-privacy" target="_blank">
    <img alt="GitHub code" src="https://img.shields.io/badge/💻_Code_GitHub-black.svg?style=for-the-badge" />
</a>
<a href="#cite" target="_blank">
    <img alt="Cite" src="https://img.shields.io/badge/📖_Cite!-lightgrey?style=for-the-badge" />
</a>

The source code for *How Far Are VLMs from Privacy Awareness in the Physical World? An Empirical Study*. The evaluation framework contains three tiers, each targeting a distinct facet of privacy reasoning:

| Tier | Capability evaluated | Typical input |
|------|----------------------|---------------|
| Tier 1 | Recognising sensitive objects | Multi-view images of a scene |
| Tier 2 | Selecting / rating an action under privacy-sensitive context | Images + audio cues |
| Tier 3 | Embodied action selection after observing a private interaction | Images + action video + dialogue |

The repository is split into two largely independent parts:

* [`mllm_privacy/`](mllm_privacy/) — **data generation pipeline**. Drives a modified VirtualHome Unity simulator to render scenes, captures images and videos, and emits the JSON metadata / prompt files consumed by the evaluator.
* [`mllm_eval/`](mllm_eval/) — **evaluation pipeline**. Loads the generated test cases, dispatches them to local or remote MLLM backends, and computes per-tier metrics.

---

## Repository layout

```
.
├── mllm_privacy/          # Scene generation pipeline (Tier 1/2/3)
│   ├── tier1.py           #   Tier 1 driver
│   ├── tier2.py           #   Tier 2 driver
│   ├── tier3.py           #   Tier 3 driver
│   ├── utils/             #   Scene-graph, camera, prompt helpers
│   ├── prompts/           #   Prompt-variation files for Tier 1
│   └── eai_bench/         #   Input scenarios for Tier 1/2/3 generation
├── mllm_eval/             # Evaluation pipeline
│   ├── inference/         #   Sequential & batched inference runners
│   ├── models/            #   Backend implementations (vLLM, OpenAI, …)
│   ├── metrics/           #   Per-tier metrics
│   ├── reporting/         #   Score aggregation & export
│   ├── assets/            #   Benchmark cases (data.jsonl + media)
│   └── data/              #   Dataset loading and schemas
├── configs/               # YAML configs (model + eval profiles)
├── scripts/               # Convenience scripts
├── third_party/
│   └── virtualhome/       # Modified VirtualHome Python API (MIT-licensed)
└── pyproject.toml
```

---

## Installation

The project targets Python 3.10+.

```bash
# Using uv (recommended; lockfile committed)
uv sync

# Or using pip
pip install -e .
pip install -r mllm_eval/requirements.txt
```

---

## Part 1 — Evaluation pipeline (`mllm_eval/`)

The evaluation pipeline is **self-contained**: given a directory of pre-rendered cases, it can be run without ever launching the simulator. We recommend starting here.

### Data layout

Test cases live under `mllm_eval/assets/`. Each top-level directory is one *tier dataset*; each subdirectory is one case:

```
mllm_eval/assets/
├── tier1_1item/                       # Tier 1, scenes with 1 sensitive item
│   ├── run_args.json                  #   provenance: arguments used to render this dataset
│   └── scene_000/
│       ├── metadata.json              #   scene metadata (objects, paths, etc.)
│       ├── question.json              #   prompts + reference answers
│       └── images/                    #   image inputs
├── tier1_2item/  …  tier1_20item/     # Same layout, varying number of items
├── tier2/
│   └── scene_000_state_000_var000/    # state × variation per scenario
│       ├── metadata.json
│       ├── question.json
│       ├── images/
│       └── audio/                     # audio cues (Tier 2)
└── tier3/
    └── scene_000_var000/
        ├── metadata.json
        ├── question.json
        ├── images/
        ├── audio/                     # optional
        └── action_video.mp4           # observed action video (Tier 3a)
```

The runner loads every case directory containing a `question.json` under each configured tier directory.

### `question.json` schema

Each `question.json` describes one or more typed questions for a single scene.

```json
{
  "scenario_name": "Anniversary Gift",
  "objects": ["box", "book", "pen", "notebook"],
  "questions": {
    "tier3a_selection": {
      "question_id": "q1",
      "type": "tier3a_selection",
      "turns": [
        {
          "turn_id": 1,
          "prompt": "You are an embodied AI agent ...",
          "images": ["images/target_000.png", "images/target_001.png"],
          "audio":  [],
          "video":  "action_video.mp4"
        }
      ],
      "answer": [1, 2]
    }
  }
}
```

### Running an evaluation

The single entry point is [`scripts/run_eval.py`](scripts/run_eval.py). It accepts a base config plus optional per-model and per-tier overrides:

```bash
# Default: vllm_local + Qwen3-VL + Tier 1 / 2 / 3
python scripts/run_eval.py --config configs/default.yaml

# One model × one tier
python scripts/run_eval.py \
  --config configs/default.yaml \
  --model-config configs/models/gemini_3_flash.yaml \
  --eval-config  configs/eval/tier1.yaml

# Sweep multiple models × multiple tiers in one invocation
python scripts/run_eval.py \
  --config configs/default.yaml \
  --model-configs configs/models/qwen3_8b.yaml configs/models/gpt_5_4.yaml \
  --eval-configs  configs/eval/tier1.yaml      configs/eval/tier3.yaml

# Auto-discover every model config under configs/models/
python scripts/run_eval.py \
  --config configs/default.yaml \
  --all-model-configs \
  --eval-configs configs/eval/tier1.yaml configs/eval/tier3.yaml

# Async / concurrent inference (recommended for remote API backends)
python scripts/run_eval.py --config configs/default.yaml --batch

# Repeat the same evaluation N times and aggregate mean ± std
python scripts/run_eval.py --config configs/default.yaml --repeat 3 --batch

# Re-score an existing predictions.jsonl without re-running inference
python scripts/run_eval.py --rescore results/<run_name>/predictions.jsonl
```

CLI flags:

| Flag | Purpose |
|------|---------|
| `--config` | Base YAML configuration (default `configs/default.yaml`) |
| `--model-config[s]` | Override the `model:` block (single path or list) |
| `--all-model-configs` | Iterate over every `*.yaml` under `--model-config-dir` |
| `--model-config-dir` | Directory scanned by `--all-model-configs` (default `configs/models`) |
| `--eval-config[s]` | Override the `eval:` block (single path or list) |
| `--batch` | Use the asynchronous concurrent inference runner |
| `--repeat N` | Run the same job N times; emit `repeat_summary.json` / `repeat_report.md` |
| `--rescore PATH` | Re-score an existing `predictions.jsonl` (skips inference) |
| `--continue-on-error` | Keep going if a job in a sweep fails |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Configuration

Configs use layered YAML overrides; later layers replace earlier ones at the top-level key granularity:

```
configs/default.yaml          ← base (run_name, output_dir, model, eval)
  ↑ overridden by
configs/models/<model>.yaml   ← replaces the `model:` block
  ↑ overridden by
configs/eval/<tier>.yaml      ← replaces the `eval:` block
```

A typical eval config simply lists the tier directories to load and (optionally) restricts which question types to evaluate within each tier:

```yaml
# configs/eval/tier1.yaml
tiers:
  - "tier1_1item"
  - "tier1_5item"
  - "tier1_10item"
question_types:
  tier1:
    - "tier1_list_multiround"
```

A tier name is matched both as an exact directory name and as a prefix: listing `"tier1"` loads every `tier1_*` directory; listing `"tier1_5item"` loads only that one.

### Supported model backends

Implementations live under [`mllm_eval/models/`](mllm_eval/models/) and are selected by `backend_type` in the model config:

| `backend_type` | Use case | Required env var |
|----------------|----------|------------------|
| `vllm_local` | In-process vLLM (no separate server) | — |
| `vllm` | OpenAI-compatible vLLM server | `OPENAI_API_KEY` (any value) |
| `openai` | OpenAI / GPT-class APIs | `OPENAI_API_KEY` |
| `google` | Google Gemini API | `GOOGLE_API_KEY` |
| `dashscope` | Alibaba DashScope (e.g. Qwen3-Omni-Flash) | `DASHSCOPE_API_KEY` |
| `volcengine` | Volcengine (e.g. Doubao Seed) | `VOLC_API_KEY` |

### Outputs

Each single run writes to `results/<run_name>/`:

| File | Description |
|------|-------------|
| `predictions.jsonl` | Raw per-case model outputs |
| `scores.json` | Machine-readable per-metric scores (overall / per-tier / per-type / per-tier×type) |
| `report.md` | Human-readable Markdown report |
| `results_table.tex` | LaTeX table for inclusion in the paper |

When `--repeat N` is used, a parent directory is created that additionally contains `repeat_summary.json` and `repeat_report.md` with mean ± std across the N runs.

---

## Part 2 — Data generation pipeline (`mllm_privacy/`)

> **Note.** The data generation pipeline drives a **modified VirtualHome Unity simulator**. The simulator binary is *not* included in this repository as our modifications integrate third-party assets whose licensing we are still clarifying, and we therefore cannot redistribute it. A **representative subset of pre-rendered assets** is provided separately for sanity-checking the evaluation pipeline (see "Reproducing the paper" below); the full dataset and the modified simulator will be released soon.

### Tier 1 — sensitive-object recognition

Tier 1 places sensitive objects in household scenes and renders multi-view image sequences. The generated prompts probe whether the model can *list* the sensitive objects.

```bash
python -m mllm_privacy.tier1 \
  --prompts-file mllm_privacy/prompts/tier1_variations_5_items_30_prompts.json \
  --max-objects 10 --total-objects 10 \
  --sensitive-orbit \
  --closeup --closeup-distance 0.3 \
  --output output/tier1_10item
```

If you only want to regenerate the prompt JSON without invoking the simulator, add `--prompts-only`. Existing `metadata.json` files in the output directory will be reused to fill in image paths.

### Tier 2 — action selection / rating with audio cues

Tier 2 drives a scene and produces audio-augmented action-selection cases:

```bash
python -m mllm_privacy.tier2 \
  --tier2_file mllm_privacy/eai_bench/tier_2.json \
  --start_index 0 --end_index 1 \
  --env 50
```

### Tier 3 — embodied privacy-aware action

Tier 3 simulates a three-person interaction in which a "secret item" is hidden by one person while a third person issues an unrelated task to the robot. The pipeline renders the action video, captures multi-view images of the target object, and emits the prompt that asks the model to choose the next action:

```bash
python -m mllm_privacy.tier3 \
  --tier3_file mllm_privacy/eai_bench/tier_3.json \
  --tier 3a \
  --output output/tier3
```

### Modifications to VirtualHome — what changed

We modified both sides of the VirtualHome stack:

* **Unity simulator:** added new sensitive-object prefabs and scene assets, finer-grained character / object placement control, and additional camera utilities for orbit and close-up shots.
* **Python API ([`third_party/virtualhome/`](third_party/virtualhome/)):**
  added new `UnityCommunication` endpoints that expose the above simulator features to the Python side, plus small adjustments to existing helpers used by the Tier 1/2/3 drivers.

The Python-side changes are included in this repository; the Unity-side changes are bundled in the currently unreleased modified simulator binary.

---

## Reproducing the paper

The recommended path during double-blind review is:

1. Move the **pre-rendered data sample** provided in `dataset/` into `mllm_eval/assets/`.
2. Run `python scripts/run_eval.py` as described above.

---

## Licensing

* [`third_party/virtualhome/`](third_party/virtualhome/) contains a **modified** copy of the upstream VirtualHome Python API (added endpoints for our simulator extensions, minor helper tweaks) redistributed under the original MIT license. See [`third_party/virtualhome/LICENSE`](third_party/virtualhome/LICENSE) for the original copyright notice.
* **Tier 2 audio cues** are sourced from [ear0.com](https://www.ear0.com) and are distributed under the CC0 license by the original platform. The clips have been transcoded to WAV and are included for non-commercial research use; all copyright in the underlying recordings remains with their respective creators.
* The modified Unity simulator binary will be released upon third-party asset clearance.

## Cite

If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{wang2026far,
  title={How Far Are VLMs from Privacy Awareness in the Physical World? An Empirical Study},
  author={Wang, Junran and Shen, Xinjie and Jin, Zehao and Li, Pan},
  journal={arXiv preprint arXiv:2605.05340},
  year={2026}
}
```
