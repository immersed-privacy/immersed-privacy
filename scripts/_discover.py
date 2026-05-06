"""Shared discovery of task-level prediction files across two directory layouts.

Layout A (repeat3, multi-run):
    results/<tier>/<repeat_dir>/<run_dir>/tasks/<task>/predictions.jsonl

Layout B (single-run):
    results/<tier>/<experiment_dir>/tasks/<task>/predictions.jsonl

Returns unified tuples: (tier, experiment_dir, run_name, task, pred_path)
  - For layout A: experiment_dir = repeat_dir, run_name = run_dir name
  - For layout B: experiment_dir = experiment_dir, run_name = "run01" (synthetic)
"""
from __future__ import annotations
from pathlib import Path


def discover_task_predictions(
    results_dir: Path,
    tier_dirs: list[str] | None = None,
    task_filter: str | None = None,
) -> list[tuple[str, str, str, str, Path]]:
    """Yield (tier, experiment_dir, run_name, task_name, pred_path)."""
    if tier_dirs is None:
        tier_dirs = ["tier2", "tier3"]

    found: list[tuple[str, str, str, str, Path]] = []

    for tier in tier_dirs:
        tier_path = results_dir / tier
        if not tier_path.is_dir():
            continue
        for exp_dir in sorted(tier_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            tasks_dir = exp_dir / "tasks"
            if tasks_dir.is_dir():
                # Layout B: single-run  <tier>/<exp>/tasks/<task>/predictions.jsonl
                for task_dir in sorted(tasks_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    if task_filter and task_filter not in task_dir.name:
                        continue
                    pf = task_dir / "predictions.jsonl"
                    if pf.exists():
                        found.append((tier, exp_dir.name, "run01", task_dir.name, pf))
            else:
                # Layout A: repeat3  <tier>/<repeat>/<run>/tasks/<task>/predictions.jsonl
                for run_dir in sorted(exp_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    run_tasks = run_dir / "tasks"
                    if not run_tasks.is_dir():
                        continue
                    for task_dir in sorted(run_tasks.iterdir()):
                        if not task_dir.is_dir():
                            continue
                        if task_filter and task_filter not in task_dir.name:
                            continue
                        pf = task_dir / "predictions.jsonl"
                        if pf.exists():
                            found.append((tier, exp_dir.name, run_dir.name, task_dir.name, pf))

    return found
