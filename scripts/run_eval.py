#!/usr/bin/env python3
"""
run_eval.py — Main entry point for the mllm_privacy evaluation benchmark.

Supports sequential and batch inference, repeat runs with mean/std
aggregation, and rescoring of existing predictions.

Usage:
    python scripts/run_eval.py --config configs/default.yaml
    python scripts/run_eval.py --config configs/default.yaml --batch
    python scripts/run_eval.py --config configs/default.yaml --repeat 3
    python scripts/run_eval.py --rescore results/run_01/predictions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: dict) -> dict:
    result = {}
    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return value.strip("_") or "run"


def resolve_model_config_paths(args: argparse.Namespace) -> list[Path | None]:
    if args.all_model_configs:
        model_dir = Path(args.model_config_dir)
        paths = sorted(model_dir.glob("*.yaml"))
        if not paths:
            raise ValueError(f"No model configs found under: {model_dir}")
        return paths
    if args.model_configs:
        return [Path(p) for p in args.model_configs]
    if args.model_config:
        return [Path(args.model_config)]
    return [None]


def resolve_eval_config_paths(args: argparse.Namespace) -> list[Path | None]:
    if args.eval_configs:
        return [Path(p) for p in args.eval_configs]
    if args.eval_config:
        return [Path(args.eval_config)]
    return [None]


def build_run_name(
    base_name: str,
    model_cfg: Path | None,
    eval_cfg: Path | None,
    timestamp: str,
    force_suffix: bool,
) -> str:
    if not force_suffix and model_cfg is None and eval_cfg is None:
        return base_name
    model_tag = sanitize_name(model_cfg.stem if model_cfg else "default_model")
    eval_tag = sanitize_name(eval_cfg.stem if eval_cfg else "default_eval")
    return f"{sanitize_name(base_name)}__{model_tag}__{eval_tag}__{timestamp}"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

async def run_single(config: dict, use_batch: bool = False) -> dict[str, Any]:
    """Run inference + scoring for a single config. Returns scores dict."""
    from mllm_eval.reporting.aggregator import Aggregator
    from mllm_eval.reporting.export import ResultExporter

    if use_batch:
        from mllm_eval.inference.batch import run_batch_inference
        logging.info("Running batch inference...")
        predictions_path = await run_batch_inference(config)
    else:
        from mllm_eval.inference.runner import run_inference
        logging.info("Running sequential inference...")
        predictions_path = await run_inference(config)

    logging.info("Scoring predictions...")
    aggregator = Aggregator()
    scores = aggregator.score_predictions(predictions_path)

    output_dir = predictions_path.parent
    run_name = config.get("run_name", "")
    logging.info("Exporting results...")
    paths = ResultExporter.export_all(scores, output_dir, run_name)

    overall = scores.get("overall", {})
    logging.info("=" * 60)
    logging.info("EVALUATION COMPLETE")
    for k, v in sorted(overall.items()):
        if isinstance(v, float):
            logging.info("  %s: %.4f", k, v)
        else:
            logging.info("  %s: %s", k, v)
    logging.info("Results directory: %s", output_dir)
    logging.info("=" * 60)

    return scores


# ---------------------------------------------------------------------------
# Rescore
# ---------------------------------------------------------------------------

async def run_rescore(predictions_path: Path, run_name: str = "") -> dict[str, Any]:
    """Re-score existing predictions without running inference."""
    from mllm_eval.reporting.aggregator import Aggregator
    from mllm_eval.reporting.export import ResultExporter

    logging.info("Rescoring predictions from %s", predictions_path)
    aggregator = Aggregator()
    scores = aggregator.score_predictions(predictions_path)

    output_dir = predictions_path.parent
    paths = ResultExporter.export_all(scores, output_dir, run_name)

    overall = scores.get("overall", {})
    logging.info("=" * 60)
    logging.info("RESCORE COMPLETE")
    for k, v in sorted(overall.items()):
        if isinstance(v, float):
            logging.info("  %s: %.4f", k, v)
        else:
            logging.info("  %s: %s", k, v)
    logging.info("Results directory: %s", output_dir)
    logging.info("=" * 60)

    return scores


# ---------------------------------------------------------------------------
# Repeat runs
# ---------------------------------------------------------------------------

def _compute_repeat_summary(all_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std/values across N runs for every metric."""
    n = len(all_scores)
    summary: dict[str, Any] = {"num_runs": n}

    def _stat(values: list[float]) -> dict[str, Any]:
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        return {"mean": mean, "std": math.sqrt(variance), "values": values}

    def _aggregate_level(key: str) -> dict[str, Any]:
        """Aggregate a top-level key (overall, per_tier, per_type, per_tier_type)."""
        if key == "overall":
            all_metrics: dict[str, list[float]] = {}
            for scores in all_scores:
                data = scores.get(key, {})
                for mk, mv in data.items():
                    if isinstance(mv, (int, float)):
                        all_metrics.setdefault(mk, []).append(float(mv))
            return {mk: _stat(vals) for mk, vals in all_metrics.items()}
        else:
            sub_keys: set[str] = set()
            for scores in all_scores:
                sub_keys.update(scores.get(key, {}).keys())

            result: dict[str, dict[str, Any]] = {}
            for sk in sorted(sub_keys):
                all_metrics = {}
                for scores in all_scores:
                    data = scores.get(key, {}).get(sk, {})
                    for mk, mv in data.items():
                        if isinstance(mv, (int, float)):
                            all_metrics.setdefault(mk, []).append(float(mv))
                result[sk] = {mk: _stat(vals) for mk, vals in all_metrics.items()}
            return result

    for level in ("overall", "per_tier", "per_type", "per_tier_type"):
        summary[level] = _aggregate_level(level)

    return summary


def _write_repeat_report(summary: dict[str, Any], output_dir: Path, run_name: str) -> None:
    """Write repeat_report.md and repeat_summary.json."""
    with open(output_dir / "repeat_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines: list[str] = []
    display_name = run_name or "Experiment"
    lines.append(f"# Repeat Evaluation Report: {display_name}")
    lines.append("")
    lines.append(f"**Runs:** {summary['num_runs']}")
    lines.append("")

    def _format_section(title: str, data: dict[str, Any], is_flat: bool = False) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if is_flat:
            for mk in sorted(data.keys()):
                s = data[mk]
                lines.append(f"- **{mk}**: {s['mean']:.4f} ± {s['std']:.4f}")
            lines.append("")
        else:
            for sk in sorted(data.keys()):
                lines.append(f"### {sk}")
                lines.append("")
                metrics = data[sk]
                for mk in sorted(metrics.keys()):
                    s = metrics[mk]
                    lines.append(f"- **{mk}**: {s['mean']:.4f} ± {s['std']:.4f}")
                lines.append("")

    overall = summary.get("overall", {})
    _format_section("Overall Results (mean ± std)", overall, is_flat=True)

    per_tier = summary.get("per_tier", {})
    if per_tier:
        lines.append("## Per-Tier Results")
        lines.append("")
        lines.append("| Tier | Metric | mean ± std |")
        lines.append("|------|--------|------------|")
        for tier in sorted(per_tier.keys()):
            metrics = per_tier[tier]
            first = True
            for mk in sorted(metrics.keys()):
                s = metrics[mk]
                tier_label = tier if first else ""
                first = False
                lines.append(f"| {tier_label} | {mk} | {s['mean']:.4f} ± {s['std']:.4f} |")
        lines.append("")

    per_type = summary.get("per_type", {})
    if per_type:
        _format_section("Per-Type Breakdown", per_type)

    per_tier_type = summary.get("per_tier_type", {})
    if per_tier_type:
        lines.append("## Per Tier×Type Breakdown")
        lines.append("")
        current_tier = ""
        for key in sorted(per_tier_type.keys()):
            parts = key.split("/", 1)
            tier = parts[0] if len(parts) > 1 else ""
            type_tag = parts[1] if len(parts) > 1 else key
            if tier != current_tier:
                lines.append(f"### {tier}")
                lines.append("")
                current_tier = tier
            lines.append(f"#### {type_tag}")
            lines.append("")
            metrics = per_tier_type[key]
            for mk in sorted(metrics.keys()):
                s = metrics[mk]
                lines.append(f"- **{mk}**: {s['mean']:.4f} ± {s['std']:.4f}")
            lines.append("")

    with open(output_dir / "repeat_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info("Repeat summary written to %s", output_dir)


async def run_repeat(
    config: dict, n: int, use_batch: bool = False
) -> dict[str, Any]:
    """Run N repeat evaluations and aggregate."""
    base_run_name = config.get("run_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_name = f"{base_run_name}_repeat{n}_{timestamp}"
    output_base = Path(config.get("output_dir", "results"))
    parent_dir = output_base / parent_name
    parent_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        run_name = f"{base_run_name}_run{i:02d}"
        run_config = merge_configs({}, config)
        run_config["run_name"] = str(parent_dir / run_name)
        run_config["output_dir"] = ""

        logging.info("-" * 80)
        logging.info("Repeat run [%d/%d]: %s", i, n, run_name)

        scores = await run_single(run_config, use_batch=use_batch)
        all_scores.append(scores)

    summary = _compute_repeat_summary(all_scores)
    _write_repeat_report(summary, parent_dir, base_run_name)

    logging.info("=" * 60)
    logging.info("ALL %d REPEAT RUNS COMPLETE", n)
    logging.info("Summary: %s", parent_dir / "repeat_summary.json")
    logging.info("Report:  %s", parent_dir / "repeat_report.md")
    logging.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="mllm_privacy evaluation benchmark runner"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the main experiment config",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to a model-specific config to override model settings",
    )
    parser.add_argument(
        "--model-configs", type=str, nargs="+", default=None,
        help="Multiple model config paths to run sequentially",
    )
    parser.add_argument(
        "--all-model-configs", action="store_true",
        help="Automatically run all YAML files under --model-config-dir",
    )
    parser.add_argument(
        "--model-config-dir", type=str, default="configs/models",
        help="Directory used by --all-model-configs",
    )
    parser.add_argument(
        "--eval-config", type=str, default=None,
        help="Path to a tier-specific eval config to override eval settings",
    )
    parser.add_argument(
        "--eval-configs", type=str, nargs="+", default=None,
        help="Multiple eval config paths to run sequentially",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use async batch inference with concurrency control",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of repeat runs (default: 1, no repeat)",
    )
    parser.add_argument(
        "--rescore", type=str, default=None,
        help="Path to existing predictions.jsonl to re-score (skips inference)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue remaining jobs if one fails",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.model_config and args.model_configs:
        parser.error("Use either --model-config or --model-configs, not both.")
    if args.eval_config and args.eval_configs:
        parser.error("Use either --eval-config or --eval-configs, not both.")

    if args.rescore:
        predictions_path = Path(args.rescore)
        if not predictions_path.exists():
            parser.error(f"Predictions file not found: {predictions_path}")
        asyncio.run(run_rescore(predictions_path))
        return

    base_config = load_yaml_config(args.config)

    model_paths = resolve_model_config_paths(args)
    eval_paths = resolve_eval_config_paths(args)

    jobs: list[tuple[dict, Path | None, Path | None]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_job = len(model_paths) * len(eval_paths) > 1

    for model_path in model_paths:
        for eval_path in eval_paths:
            config = merge_configs({}, base_config)

            if model_path:
                model_override = load_yaml_config(model_path)
                config = merge_configs(config, {"model": model_override})

            if eval_path:
                eval_override = load_yaml_config(eval_path)
                config = merge_configs(config, {"eval": eval_override})

            base_run_name = config.get("run_name", "experiment")
            config["run_name"] = build_run_name(
                base_name=base_run_name,
                model_cfg=model_path,
                eval_cfg=eval_path,
                timestamp=timestamp,
                force_suffix=multi_job,
            )

            jobs.append((config, model_path, eval_path))

    logging.info("Loaded base configuration: %s", args.config)
    logging.info("Prepared %d evaluation job(s)", len(jobs))

    async def run_jobs() -> None:
        for index, (config, model_path, eval_path) in enumerate(jobs, start=1):
            logging.info("-" * 80)
            logging.info("Starting job [%d/%d]", index, len(jobs))
            logging.info("Model config: %s", model_path if model_path else "<default>")
            logging.info("Eval config : %s", eval_path if eval_path else "<default>")
            logging.info("Run name    : %s", config.get("run_name"))
            logging.info(
                "Backend     : %s",
                config.get("model", {}).get("backend_type", "unknown"),
            )
            logging.info(
                "Model       : %s",
                config.get("model", {}).get("model_name", "unknown"),
            )

            try:
                if args.repeat > 1:
                    await run_repeat(config, args.repeat, use_batch=args.batch)
                else:
                    await run_single(config, use_batch=args.batch)
            except Exception:
                logging.exception("Job [%d/%d] failed", index, len(jobs))
                if not args.continue_on_error:
                    raise
                logging.warning("Continuing due to --continue-on-error")

    asyncio.run(run_jobs())


if __name__ == "__main__":
    main()
