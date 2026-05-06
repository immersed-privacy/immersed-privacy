"""
export.py — Export results to various formats.

Generates machine-readable (JSON), human-readable (Markdown), and
publication-ready (LaTeX) outputs from aggregated scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultExporter:
    """Exports evaluation results to multiple formats."""

    @staticmethod
    def export_all(
        scores: dict[str, Any],
        output_dir: Path,
        run_name: str = "",
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        paths["json"] = ResultExporter.export_json(scores, output_dir)
        paths["markdown"] = ResultExporter.export_markdown(scores, output_dir, run_name)
        paths["latex"] = ResultExporter.export_latex(scores, output_dir)
        return paths

    @staticmethod
    def export_json(scores: dict[str, Any], output_dir: Path) -> Path:
        path = output_dir / "scores.json"

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False, default=_serialize)

        logger.info("Exported JSON scores to %s", path)
        return path

    @staticmethod
    def export_markdown(
        scores: dict[str, Any], output_dir: Path, run_name: str = ""
    ) -> Path:
        path = output_dir / "report.md"
        lines: list[str] = []

        title = f"Evaluation Report: {run_name}" if run_name else "Evaluation Report"
        lines.append(f"# {title}")
        lines.append("")

        overall = scores.get("overall", {})
        lines.append("## Overall Results")
        lines.append("")
        for k in sorted(overall.keys()):
            v = overall[k]
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        per_tier = scores.get("per_tier", {})
        if per_tier:
            lines.append("## Per-Tier Results")
            lines.append("")
            lines.append("| Tier | Metric | Value |")
            lines.append("|------|--------|-------|")
            for tier in sorted(per_tier.keys()):
                info = per_tier[tier]
                first = True
                for k in sorted(info.keys()):
                    v = info[k]
                    tier_label = tier if first else ""
                    first = False
                    if isinstance(v, float):
                        lines.append(f"| {tier_label} | {k} | {v:.4f} |")
                    else:
                        lines.append(f"| {tier_label} | {k} | {v} |")
            lines.append("")

        per_type = scores.get("per_type", {})
        if per_type:
            lines.append("## Per-Type Breakdown")
            lines.append("")
            for type_tag in sorted(per_type.keys()):
                metrics = per_type[type_tag]
                lines.append(f"### {type_tag}")
                lines.append("")
                for k in sorted(metrics.keys()):
                    v = metrics[k]
                    if isinstance(v, float):
                        lines.append(f"- **{k}**: {v:.4f}")
                    else:
                        lines.append(f"- **{k}**: {v}")
                lines.append("")

        per_tier_type = scores.get("per_tier_type", {})
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
                for k in sorted(metrics.keys()):
                    v = metrics[k]
                    if isinstance(v, float):
                        lines.append(f"- **{k}**: {v:.4f}")
                    else:
                        lines.append(f"- **{k}**: {v}")
                lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Exported Markdown report to %s", path)
        return path

    @staticmethod
    def export_latex(scores: dict[str, Any], output_dir: Path) -> Path:
        path = output_dir / "results_table.tex"
        lines: list[str] = []

        per_tier = scores.get("per_tier", {})
        tiers = sorted(per_tier.keys())

        all_metrics = set()
        for info in per_tier.values():
            for k, v in info.items():
                if isinstance(v, float):
                    all_metrics.add(k)
        metric_cols = sorted(all_metrics - {"num_records"})

        n_cols = 1 + len(metric_cols) + 1
        col_spec = "l" + "c" * (len(metric_cols) + 1)

        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Evaluation Results}")
        lines.append(r"\label{tab:eval_results}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        header = "Tier & " + " & ".join(metric_cols) + r" & Records \\"
        lines.append(header)
        lines.append(r"\midrule")

        for tier in tiers:
            info = per_tier[tier]
            vals = [f"{info.get(m, 0.0):.4f}" for m in metric_cols]
            records = info.get("num_records", 0)
            lines.append(f"{tier} & " + " & ".join(vals) + f" & {records} \\\\")

        overall = scores.get("overall", {})
        lines.append(r"\midrule")
        vals = [f"{overall.get(m, 0.0):.4f}" for m in metric_cols]
        total = overall.get("total_records", 0)
        lines.append(f"Overall & " + " & ".join(vals) + f" & {total} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Exported LaTeX table to %s", path)
        return path
