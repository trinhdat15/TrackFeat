#!/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python
"""Extract paper-only ACCIDENT ablations from sealed Prompt-1/Prompt-2 ledgers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/ssd1/team_cam_ai/ttdat")
OUT = ROOT / "final_draft/trackfeat_aisi_submission_03"
BASE = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "accident_interaction_type_classification"
)
P1 = BASE / "accident_three_task_handoff_01"
P2 = BASE / "accident_three_task_phase2_temporal_01"
COMMAND = (
    "/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python "
    "final_draft/trackfeat_aisi_submission_03/control/"
    "build_accident_ablation_registry.py"
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_temporal(rows: list[dict], row: pd.Series, variant: str, status: str, source: Path) -> None:
    rows.append(
        {
            "task": "temporal",
            "variant": variant,
            "condition": "grouped-OOF development",
            "metric": "T",
            "T_at_0_5": row.T_sigma_0_5,
            "T_at_1": row.T_sigma_1,
            "T_at_2": row.T_sigma_2,
            "median_absolute_error_s": row.median_absolute_error_s,
            "fallback_count": row.fallback_count,
            "S": pd.NA,
            "C": pd.NA,
            "macro_f1": pd.NA,
            "balanced_accuracy": pd.NA,
            "status": status,
            "finding": "",
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": sha(source),
        }
    )


def add_spatial(rows: list[dict], row: pd.Series, variant: str, status: str, source: Path) -> None:
    rows.append(
        {
            "task": "spatial",
            "variant": variant,
            "condition": row.condition,
            "metric": "S@1",
            "T_at_0_5": pd.NA,
            "T_at_1": pd.NA,
            "T_at_2": pd.NA,
            "median_absolute_error_s": pd.NA,
            "fallback_count": row.fallback_count,
            "S": row.S_sigma_1,
            "C": pd.NA,
            "macro_f1": pd.NA,
            "balanced_accuracy": pd.NA,
            "status": status,
            "finding": "",
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": sha(source),
        }
    )


def add_type(rows: list[dict], row: pd.Series, variant: str, status: str, source: Path) -> None:
    rows.append(
        {
            "task": "collision type",
            "variant": variant,
            "condition": row.condition,
            "metric": "C",
            "T_at_0_5": pd.NA,
            "T_at_1": pd.NA,
            "T_at_2": pd.NA,
            "median_absolute_error_s": pd.NA,
            "fallback_count": pd.NA,
            "S": pd.NA,
            "C": row.top1_accuracy_C,
            "macro_f1": row.macro_f1,
            "balanced_accuracy": row.balanced_accuracy,
            "status": status,
            "finding": "",
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": sha(source),
        }
    )


def main() -> None:
    p1_temporal_path = P1 / "temporal_results.csv"
    p2_temporal_path = P2 / "temporal_candidate_results.csv"
    spatial_pred_path = P1 / "spatial_predicted_time_results.csv"
    spatial_oracle_path = P1 / "spatial_oracle_results.csv"
    p1_type_path = P1 / "collision_type_results.csv"
    p2_type_path = P2 / "type_candidate_results.csv"
    unified_path = P1 / "unified_accident_results.csv"
    bootstrap_path = P1 / "bootstrap_summary.csv"
    phase2_chain_path = P2 / "combined_chain_report_only.json"
    phase2_verdict_path = P2 / "phase2_scientific_verdict.json"

    t1 = pd.read_csv(p1_temporal_path).set_index("method")
    t2 = pd.read_csv(p2_temporal_path).set_index("candidate_id")
    sp = pd.read_csv(spatial_pred_path)
    so = pd.read_csv(spatial_oracle_path)
    ty1 = pd.read_csv(p1_type_path)
    ty2 = pd.read_csv(p2_type_path).set_index("candidate_id")

    rows: list[dict] = []
    add_temporal(rows, t1.loc["clip_midpoint"], "Clip midpoint", "control", p1_temporal_path)
    add_temporal(rows, t1.loc["learned_pair_lgbm"], "Learned pair LightGBM", "Prompt-1 learned diagnostic", p1_temporal_path)
    add_temporal(rows, t1.loc["frozen_full132_score_peak"], "Full-132 score peak", "frozen-score control", p1_temporal_path)
    add_temporal(rows, t1.loc["pair_interaction_peak"], "Raw pair-interaction peak", "interaction control", p1_temporal_path)
    add_temporal(rows, t2.loc["T2-A"], "T2-A Gaussian regressor", "Phase-2 diagnostic; not supported", p2_temporal_path)

    add_spatial(
        rows,
        sp[sp.method.eq("image_center")].iloc[0],
        "Image center",
        "control",
        spatial_pred_path,
    )
    add_spatial(
        rows,
        sp[sp.method.eq("closest_pair_midpoint")].iloc[0],
        "Closest-pair midpoint",
        "control",
        spatial_pred_path,
    )
    add_spatial(
        rows,
        sp[sp.method.eq("highest_risk_pair_midpoint")].iloc[0],
        "Highest-risk-pair midpoint",
        "Prompt-1 primary spatial component",
        spatial_pred_path,
    )
    add_spatial(
        rows,
        so[so.method.eq("highest_risk_pair_midpoint")].iloc[0],
        "Highest-risk pair with oracle time",
        "oracle diagnostic",
        spatial_oracle_path,
    )

    majority = ty1[
        ty1.system_id.eq("official_majority")
        & ty1.condition.eq("oracle_time_full_scene")
        & ty1.delay_s.eq(4.0)
    ].iloc[0]
    hard = ty1[
        ty1.system_id.eq("exact_reduced_52__extra_trees")
        & ty1.condition.eq("predicted_time_and_pair")
        & ty1.delay_s.eq(4.0)
    ].iloc[0]
    full = ty1[
        ty1.system_id.eq("exact_reduced_52__extra_trees")
        & ty1.condition.eq("predicted_time_full_scene")
        & ty1.delay_s.eq(4.0)
    ].iloc[0]
    oracle = ty1[
        ty1.system_id.eq("exact_reduced_52__extra_trees")
        & ty1.condition.eq("oracle_time_full_scene")
        & ty1.delay_s.eq(4.0)
    ].iloc[0]
    c2c = ty2.loc["C2-C"]
    if isinstance(c2c, pd.DataFrame):
        c2c = c2c[c2c.condition.eq("prompt1_predicted_time_4s")].iloc[0]
    add_type(rows, majority, "Fold-majority", "control", p1_type_path)
    add_type(rows, hard, "Predicted-time hard selected pair", "Prompt-1 primary chained type component", p1_type_path)
    add_type(rows, full, "Predicted-time full scene", "Prompt-1 diagnostic", p1_type_path)
    add_type(rows, oracle, "Oracle-time full scene", "oracle diagnostic", p1_type_path)
    add_type(rows, c2c, "Hierarchical C2-C", "Phase-2 diagnostic; modest", p2_type_path)

    registry = pd.DataFrame(rows)
    registry.loc[registry.variant.eq("Clip midpoint"), "finding"] = "strongest selected temporal result"
    registry.loc[registry.variant.eq("T2-A Gaussian regressor"), "finding"] = "did not improve T@1"
    registry.loc[registry.variant.eq("Highest-risk-pair midpoint"), "finding"] = "best predicted-time spatial component"
    registry.loc[registry.variant.eq("Predicted-time hard selected pair"), "finding"] = "hard pair restriction loses type context"
    registry.loc[registry.variant.eq("Hierarchical C2-C"), "finding"] = "higher C, but paired interval crosses zero"
    registry.to_csv(OUT / "accident_ablation_registry.csv", index=False)

    evidence = registry[
        ["task", "variant", "condition", "metric", "status", "source_path", "source_sha256"]
    ].copy()
    evidence.insert(0, "ablation_id", [f"ACC-ABL-{i:02d}" for i in range(1, len(evidence) + 1)])
    evidence["generation_command"] = COMMAND
    evidence.to_csv(OUT / "accident_ablation_evidence_map.csv", index=False)

    primary = pd.read_csv(unified_path).iloc[0]
    boot = pd.read_csv(bootstrap_path).set_index("metric")
    phase2_chain = json.loads(phase2_chain_path.read_text())
    phase2_verdict = json.loads(phase2_verdict_path.read_text())
    summary_rows = [
        {
            "result_id": "ACCIDENT_PROMPT1_PRIMARY",
            "status": "primary grouped-OOF development",
            "T": primary.T_sigma_1,
            "T_ci_low": boot.loc["T", "ci_2_5"],
            "T_ci_high": boot.loc["T", "ci_97_5"],
            "S": primary.S_sigma_1,
            "S_ci_low": boot.loc["S", "ci_2_5"],
            "S_ci_high": boot.loc["S", "ci_97_5"],
            "C": primary.C_top1,
            "C_ci_low": boot.loc["C", "ci_2_5"],
            "C_ci_high": boot.loc["C", "ci_97_5"],
            "ACC_S": primary.ACC_S,
            "ACC_S_ci_low": boot.loc["ACC_S", "ci_2_5"],
            "ACC_S_ci_high": boot.loc["ACC_S", "ci_97_5"],
            "wrong_pair_count": 140,
            "verdict": "ACCIDENT_INTERACTION_THREE_TASK_BASELINE_MIXED",
            "source_path": str(unified_path.relative_to(ROOT)),
        },
        {
            "result_id": "ACCIDENT_PHASE2_REPORT_ONLY",
            "status": "report-only diagnostic",
            "T": phase2_chain["T_sigma_1"],
            "T_ci_low": pd.NA,
            "T_ci_high": pd.NA,
            "S": phase2_chain["predicted_time_S_sigma_1"],
            "S_ci_low": pd.NA,
            "S_ci_high": pd.NA,
            "C": phase2_chain["selected_type_C"],
            "C_ci_low": pd.NA,
            "C_ci_high": pd.NA,
            "ACC_S": phase2_chain["ACC_S"],
            "ACC_S_ci_low": pd.NA,
            "ACC_S_ci_high": pd.NA,
            "wrong_pair_count": phase2_chain["wrong_pair_count"],
            "verdict": phase2_verdict["overall_verdict"],
            "source_path": str(phase2_chain_path.relative_to(ROOT)),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(OUT / "accident_primary_and_diagnostic_results.csv", index=False)

    inputs = [
        p1_temporal_path,
        p2_temporal_path,
        spatial_pred_path,
        spatial_oracle_path,
        p1_type_path,
        p2_type_path,
        unified_path,
        bootstrap_path,
        phase2_chain_path,
        phase2_verdict_path,
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "paper-only extraction of completed registered ablations",
        "scientific_experiment_run": False,
        "training_or_tuning": False,
        "official_test_accessed": False,
        "population": "507 official IID-train videos; five source-grouped OOF folds",
        "primary_result": "Prompt-1 chain",
        "phase2_status": "report-only diagnostics; overall not supported",
        "command": COMMAND,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in inputs],
        "outputs": {
            "accident_ablation_registry.csv": sha(OUT / "accident_ablation_registry.csv"),
            "accident_ablation_evidence_map.csv": sha(OUT / "accident_ablation_evidence_map.csv"),
            "accident_primary_and_diagnostic_results.csv": sha(
                OUT / "accident_primary_and_diagnostic_results.csv"
            ),
        },
    }
    (OUT / "accident_ablation_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"ablation_rows": len(registry), "primary": summary_rows[0]}, indent=2))


if __name__ == "__main__":
    main()
