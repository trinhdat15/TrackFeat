#!/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python
"""Build the frozen-evidence package for the TrackFeat AAAI-27 manuscript.

This script is report-only.  It does not train, tune, rescore validation
features, or alter any predecessor artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/ssd1/team_cam_ai/ttdat")
OUT = ROOT / "final_draft/trackfeat_aisi_submission_01"
FIG = OUT / "figures"
V13 = ROOT / "outputs_sotad_phase12_old4c_motor_lowconf_t012/v13_interpretable_132_feature_study"
VAL = ROOT / "outputs_sotad_phase12_old4c_motor_lowconf_t012/v13_final_prevalidation_evidence_closure/validation_frozen_cache_recovery_01"
UP = ROOT / "outputs_sotad_phase12_old4c_motor_lowconf_t012/v12_59a_aaai_causal_upstream_pilot"
TIMESTAMP = ROOT / "outputs_sotad_phase12_old4c_motor_lowconf_t012/aisi_event_timestamp_closure_01"
NOW = datetime.now(timezone.utc).isoformat()
SEED = 130057
N_BOOT = 2000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def fhash(path: str | Path) -> str:
    p = Path(path)
    return sha(p) if p.exists() and p.is_file() else "NOT_A_SINGLE_FILE"


FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# A1. Upstream perception contract
# ---------------------------------------------------------------------------
frozen_lineage_path = ROOT / "outputs_sotad_phase12_old4c_motor_lowconf_t012/v12_57a_upstream_architecture_audit/frozen_lineage.json"
model_lineage_path = UP / "model_lineage.json"
tracker_lineage_path = UP / "tracker_lineage.json"
reid_lineage_path = UP / "reid_lineage.json"
frozen = json.loads(frozen_lineage_path.read_text())
ml = json.loads(model_lineage_path.read_text())
tl = json.loads(tracker_lineage_path.read_text())
rl = json.loads(reid_lineage_path.read_text())

det_ckpt = ROOT / "yolo/VHD.pt"
reid_ckpt = ROOT / "reid_noleak_v9_bs192_noblur/reid_best_topology.pth"
upstream_rows = [
    ("detector_family_variant", "task-adapted Ultralytics YOLOv8s-derived DetectionModel", "VERIFIED",
     "checkpoint train_args.model is yolov8s.pt; five-class DetectionModel width/depth multipliers match YOLOv8s", str(det_ckpt), sha(det_ckpt)),
    ("detector_checkpoint", "yolo/VHD.pt", "VERIFIED", "exact frozen runtime path", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("detector_checkpoint_sha256", sha(det_ckpt), "VERIFIED", "recomputed", str(det_ckpt), sha(det_ckpt)),
    ("detector_training_dataset", "checkpoint records relative manifest my_dataset.yaml; matching local package indicates VHD_VLM_NIGHT_REPLAY_4C, but the original runtime-relative manifest is not uniquely recoverable", "PARTIALLY_VERIFIED",
     "dataset basename is checkpoint metadata; expanded dataset identity remains ambiguous", str(det_ckpt), sha(det_ckpt)),
    ("checkpoint_training_image_size", "640 px", "VERIFIED", "checkpoint train_args.imgsz", str(det_ckpt), sha(det_ckpt)),
    ("inference_image_size", "960 px", "VERIFIED", "frozen runtime", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("detector_classes", "motorbike, car, bus, truck, bicycle", "VERIFIED", "checkpoint class mapping", str(det_ckpt), sha(det_ckpt)),
    ("deployable_classes", "motorbike, car, bus, truck (class IDs 0--3); bicycle excluded from downstream admission", "VERIFIED", "frozen vehicle_classes", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("detector_thresholds", "raw diagnostic confidence 0.001; admitted low stage 0.05; operational confidence 0.15; tracker high 0.18; new-track 0.22", "VERIFIED",
     "frozen detector/tracker lineage", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("nms", "Ultralytics NMS active; class-agnostic; IoU 0.55", "VERIFIED", "agnostic_nms=true and iou=0.55", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("tracker", "BoxMOT BoT-SORT", "VERIFIED", "frozen implementation", str(frozen_lineage_path), sha(frozen_lineage_path)),
    ("tracker_configuration", "track_high=0.18, track_low=0.05, new_track=0.22, buffer=30; match=0.8, proximity=0.5, appearance=0.25", "VERIFIED",
     "frozen wrapper plus constructor lineage", str(tracker_lineage_path), sha(tracker_lineage_path)),
    ("camera_motion_compensation", "ECC", "VERIFIED", "cmc_method=ecc", str(tracker_lineage_path), sha(tracker_lineage_path)),
    ("reid_enabled", "true", "VERIFIED", "BoT-SORT with current-detection-crop ReID", str(reid_lineage_path), sha(reid_lineage_path)),
    ("reid_architecture", "ResNet50-IBN-a backbone, GeM pooling, 2048-to-512 projection, batch normalization, L2 normalization", "VERIFIED",
     "frozen executable and state-dict topology", str(model_lineage_path), sha(model_lineage_path)),
    ("reid_checkpoint", str(reid_ckpt.relative_to(ROOT)), "VERIFIED", "frozen runtime path", str(reid_lineage_path), sha(reid_lineage_path)),
    ("reid_checkpoint_sha256", sha(reid_ckpt), "VERIFIED", "recomputed", str(reid_ckpt), sha(reid_ckpt)),
    ("reid_embedding_dimension", "512", "VERIFIED", "projection and exported embedding schema", str(model_lineage_path), sha(model_lineage_path)),
    ("reid_crop", "92 x 92 current detection crop; BGR-to-RGB; ImageNet normalization", "VERIFIED", "frozen ReID lineage", str(reid_lineage_path), sha(reid_lineage_path)),
    ("reid_training_datasets", "not encoded in the checkpoint or frozen executable contract", "NOT_FOUND", "no recoverable training-data metadata", str(reid_ckpt), sha(reid_ckpt)),
    ("analysis_rate", "5 Hz causal grid", "VERIFIED", "frozen upstream protocol", str(UP / "experiment_spec.json"), sha(UP / "experiment_spec.json")),
    ("low_fps_behavior", "SO-TAD replay fails closed when source FPS is below 5 Hz; it does not interpolate future frames", "VERIFIED",
     "frozen replay contract", str(UP / "causal_upstream_protocol.json"), sha(UP / "causal_upstream_protocol.json")),
]
pd.DataFrame(upstream_rows, columns=["field", "value", "status", "evidence", "source_path", "source_sha256"]).to_csv(
    OUT / "upstream_perception_contract.csv", index=False
)
(OUT / "upstream_selection_evidence.md").write_text(
    "# Upstream detector selection evidence\n\n"
    "The frozen checkpoint is a task-adapted **YOLOv8s-derived** model, not YOLO11m. "
    "This follows executable checkpoint metadata (`train_args.model=yolov8s.pt`) and "
    "the serialized five-class detection graph. No complete, protocol-compatible "
    "generic-versus-fine-tuned detector comparison was found in the frozen evidence, "
    "so the paper does not claim categorical superiority over another detector family.\n\n"
    "The task-adapted checkpoint was the established upstream detector in the project "
    "pipeline and was fixed before the experiments reported here. It covers the required "
    "traffic classes, while downstream motion, direction, contact, and persistence "
    "features depend on adequate vehicle coverage and stable boxes. The study therefore "
    "evaluates TrackFeat conditional on this perception stack and does not claim an "
    "exhaustive detector comparison.\n"
)

# ---------------------------------------------------------------------------
# A2. Exact feature contracts
# ---------------------------------------------------------------------------
sem_path = V13 / "v13_a_lineage_feature_semantics/corrections/v13_a_semantics_correction_01/feature_semantics.csv"
proposal_path = V13 / "v13_d_feature_ablation_reduction/completion_01/reduced_feature_proposal.json"
sem = pd.read_csv(sem_path).sort_values("position").reset_index(drop=True)
proposal = json.loads(proposal_path.read_text())
reduced = set(proposal["features"])

temporal_tokens = ("_lag_", "_diff_", "_roll_", "_ewm", "_streak", "F1_F2_interaction")
reliability_raw = {
    "F13", "F14", "track_age", "spawn_burst", "is_small_box", "is_edge_box",
    "far_edge_unreliable", "horizon_small_unreliable", "track_fragment_risk",
    "valid_physics_mask", "frame_gap",
}
pair_bases = {"F0", "F5", "F6", "F7", "F18", "F19"}


def base_of(name: str) -> str:
    for b in ["F19", "F18", "F17", "F16", "F15", "F14", "F13", "F11", "F10", "F7", "F6", "F5", "F2", "F1", "F0"]:
        if name == b or name.startswith(b + "_"):
            return b
    return name


def paper_family(name: str) -> str:
    if any(t in name for t in temporal_tokens):
        return "temporal summaries"
    if name in reliability_raw:
        return "reliability and missingness"
    b = base_of(name)
    if b in {"F0", "F5", "F18", "F19"}:
        return "pair interaction"
    if b in {"F7", "F10"}:
        return "direction"
    if b in {"F11"} or name in {"bbox_scale_jitter", "bbox_aspect_jitter"}:
        return "shape variation"
    if b in {"F1", "F2", "F6", "F16"} or name == "bbox_center_jitter":
        return "motion"
    return "geometry"


def history_text(row: pd.Series) -> str:
    s = str(row["causal_temporal_support_samples"])
    return "current sample" if s == "0" else f"current plus up to {s} prior 5-Hz samples"


contract = sem.copy()
contract.insert(2, "paper_feature_family", contract["feature"].map(paper_family))
contract.insert(3, "unary_or_pairwise", contract["feature"].map(
    lambda x: "pair-context" if base_of(x) in pair_bases else "unary"
))
contract.insert(4, "raw_or_temporal_summary", contract["feature"].map(
    lambda x: "temporal_summary" if any(t in x for t in temporal_tokens) else "raw/current"
))
contract.insert(5, "causal_status", "causal/trailing-only")
contract.insert(6, "required_track_history", contract.apply(history_text, axis=1))
contract.insert(7, "validity_or_missingness_field", contract["contractual_missingness"])
contract["included_full_132"] = True
contract["included_reduced_52"] = contract["feature"].isin(reduced)
contract["selection_rationale"] = contract["feature"].map(
    lambda x: "retained by frozen train-OOF utility, stability, redundancy, and family-coverage rule"
    if x in reduced else "excluded by frozen train-OOF reduction rule; remains in full contract"
)
contract["implementation_source"] = contract.apply(
    lambda r: f"{r.source_code}:{r.source_symbol}:{r.source_line}", axis=1
)
contract["implementation_source_sha256"] = contract["source_code"].map(
    lambda x: fhash(ROOT / str(x)) if (ROOT / str(x)).exists() else "source hash recorded by lineage contract"
)

full_cols = [
    "position", "feature", "paper_feature_family", "unary_or_pairwise",
    "raw_or_temporal_summary", "causal_status", "required_track_history",
    "validity_or_missingness_field", "included_full_132", "included_reduced_52",
    "selection_rationale", "meaning", "normalization", "expected_range",
    "status", "implementation_source", "implementation_source_sha256",
]
contract[full_cols].to_csv(OUT / "full_132_feature_contract.csv", index=False)
contract.loc[contract.included_reduced_52, full_cols].to_csv(OUT / "reduced_52_feature_contract.csv", index=False)
contract[["position", "feature", "paper_feature_family", "included_full_132", "included_reduced_52",
          "selection_rationale"]].to_csv(OUT / "feature_contract_comparison.csv", index=False)
(OUT / "feature_selection_lineage.md").write_text(
    "# Frozen feature-selection lineage\n\n"
    "**Classification: mixed.** The 52-feature contract was selected before sealed "
    "validation from training-OOF evidence only. The deterministic rule removed "
    "high-missingness and constant inputs, estimated family/feature utility across "
    "three grouped folds, required positive direction in at least two folds, clustered "
    "features at absolute Spearman correlation 0.95, retained simple representatives "
    "covering 95% of stable gain, and preserved one representative from each favorable "
    "family. It is therefore an importance-guided development reduction combined with "
    "redundancy reduction and preregistered family coverage; it is not a manual "
    "validation-driven selection. Calibration ablation results were not used.\n\n"
    f"Authoritative rule: `{json.dumps(proposal['selection_rule'], sort_keys=True)}`\n"
)

# ---------------------------------------------------------------------------
# A3. Paired video bootstrap on frozen validation decisions
# ---------------------------------------------------------------------------
full_dec_path = VAL / "04_event_evaluation/validation_video_decisions_132.parquet"
red_dec_path = VAL / "04_event_evaluation/validation_video_decisions_52.parquet"
fd = pd.read_parquet(full_dec_path).sort_values("canonical_video_id").reset_index(drop=True)
rd = pd.read_parquet(red_dec_path).sort_values("canonical_video_id").reset_index(drop=True)
assert list(fd.canonical_video_id) == list(rd.canonical_video_id)
assert len(fd) == 332 and (fd.category == "accident").sum() == 46 and (fd.category == "normal").sum() == 286


def event_first(row: pd.Series) -> float:
    ev = json.loads(row.events_json)
    return min((float(x["time_s"]) for x in ev), default=np.nan)


for d in (fd, rd):
    d["first_event_s"] = d.apply(event_first, axis=1)
    d["first_alert_relative_s"] = d["first_event_s"] - d["time_of_accident_s"]


def metrics(d: pd.DataFrame) -> dict[str, float]:
    tp = float(d.matched.sum())
    fn = float(d.missed.sum())
    normal = d[d.category == "normal"]
    nfp = float(normal.normal_false_events.sum())
    wrong = float(d.wrong_window_alerts.sum())
    dup = float(d.duplicate_alerts.sum())
    fp = nfp + wrong + dup
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hours = float(normal.duration_s.sum()) / 3600.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fph": nfp / hours, "normal_false_events": nfp,
        "wrong_window_alerts": wrong,
    }


rng = np.random.default_rng(SEED)
acc_idx = np.flatnonzero(fd.category.to_numpy() == "accident")
norm_idx = np.flatnonzero(fd.category.to_numpy() == "normal")
replicates: dict[str, list[float]] = {}
for _ in range(N_BOOT):
    ia = rng.choice(acc_idx, len(acc_idx), replace=True)
    inn = rng.choice(norm_idx, len(norm_idx), replace=True)
    idx = np.concatenate([ia, inn])
    a, b = fd.iloc[idx], rd.iloc[idx]
    ma, mb = metrics(a), metrics(b)
    for k in ma:
        replicates.setdefault(f"full_132:{k}", []).append(ma[k])
        replicates.setdefault(f"reduced_52:{k}", []).append(mb[k])
        replicates.setdefault(f"reduced_minus_full:{k}", []).append(mb[k] - ma[k])
    common = a.first_alert_relative_s.notna().to_numpy() & b.first_alert_relative_s.notna().to_numpy()
    td = float(np.mean(b.first_alert_relative_s.to_numpy()[common] - a.first_alert_relative_s.to_numpy()[common])) if common.any() else np.nan
    replicates.setdefault("reduced_minus_full:first_alert_timing_s", []).append(td)

mf, mr = metrics(fd), metrics(rd)
point: dict[str, float] = {}
for k in mf:
    point[f"full_132:{k}"] = mf[k]
    point[f"reduced_52:{k}"] = mr[k]
    point[f"reduced_minus_full:{k}"] = mr[k] - mf[k]
common = fd.first_alert_relative_s.notna() & rd.first_alert_relative_s.notna()
point["reduced_minus_full:first_alert_timing_s"] = float(
    (rd.loc[common, "first_alert_relative_s"] - fd.loc[common, "first_alert_relative_s"]).mean()
)
boot_rows = []
for key, vals in replicates.items():
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    comparison, metric_name = key.split(":", 1)
    boot_rows.append({
        "comparison": comparison, "metric": metric_name,
        "point_estimate": point[key], "bootstrap_median": float(np.median(arr)),
        "ci_2_5": float(np.quantile(arr, .025)), "ci_97_5": float(np.quantile(arr, .975)),
        "replicates": len(arr), "seed": SEED,
        "sampling_unit": "paired video; accident and normal strata resampled separately",
        "population": "sealed SO-TAD validation: 46 accident, 286 normal",
        "status": "primary paired uncertainty analysis",
    })
pd.DataFrame(boot_rows).to_csv(OUT / "paired_validation_bootstrap.csv", index=False)

# ---------------------------------------------------------------------------
# A4. Frozen calibration registry and Pareto frontier
# ---------------------------------------------------------------------------
cal_path = V13 / "v13_c_event_calibration/all_calibration_cells.csv"
sel_path = V13 / "v13_c_event_calibration/selected_event_policy.json"
cal = pd.read_csv(cal_path)
selected = json.loads(sel_path.read_text())
cal["threshold"] = cal.tau_up
cal["persistence_window_s"] = cal.k_up / 5.0
cal["required_active_observations"] = cal.k_up
cal["merge_gap_s"] = cal.cooldown_s
cal["feasible_under_fph5"] = cal.fph <= 5.0 + 1e-12
cal["selected_calibration_policy"] = cal.policy_id.eq(selected["policy_id"])


def pareto_flags(group: pd.DataFrame) -> pd.Series:
    flags = []
    arr = group[["fph", "recall", "f1"]].to_numpy()
    for i, (x, y, z) in enumerate(arr):
        dominated = np.any(
            (arr[:, 0] <= x) & (arr[:, 1] >= y) & (arr[:, 2] >= z) &
            ((arr[:, 0] < x) | (arr[:, 1] > y) | (arr[:, 2] > z))
        )
        flags.append(not bool(dominated))
    return pd.Series(flags, index=group.index)


cal["pareto_frontier"] = cal.groupby("model_id", group_keys=False).apply(
    pareto_flags, include_groups=False
).sort_index()
cal["dominated"] = ~cal.pareto_frontier
cal["selection_rule"] = selected["selection_rule"]
cal["fixed_policy_validation_tp"] = np.where(cal.selected_calibration_policy, 25, np.nan)
cal["fixed_policy_validation_normal_fp"] = np.where(cal.selected_calibration_policy, 8, np.nan)
cal["fixed_policy_validation_fn"] = np.where(cal.selected_calibration_policy, 21, np.nan)
cal["fixed_policy_validation_recall"] = np.where(cal.selected_calibration_policy, 0.5434782608695652, np.nan)
cal["fixed_policy_validation_precision"] = np.where(cal.selected_calibration_policy, 0.5102040816326531, np.nan)
cal["fixed_policy_validation_f1"] = np.where(cal.selected_calibration_policy, 0.5263157894736842, np.nan)
cal["fixed_policy_validation_fph"] = np.where(cal.selected_calibration_policy, 5.0310, np.nan)
cal.to_csv(OUT / "calibration_policy_registry.csv", index=False)
cal[cal.pareto_frontier].to_csv(OUT / "calibration_pareto_frontier.csv", index=False)

# ---------------------------------------------------------------------------
# A5. Training-OOF TreeSHAP.  One maximum-OOF-score row per training video
# gives equal video weight.  Outcomes are row-label outcomes at p>=0.5.
# ---------------------------------------------------------------------------
oof_path = V13 / "v13_b_clean_132_training/training/model_runs/lgbm_05/oof_predictions.parquet"
train_feat_dir = V13 / "v13_b_clean_132_training/manifold/features/train"
oof = pd.read_parquet(oof_path)
selected_rows = oof.sort_values(["canonical_video_id", "probability"], ascending=[True, False]).drop_duplicates("canonical_video_id")
features = list(sem.feature)
rows = []
for rec in selected_rows.itertuples(index=False):
    vp = train_feat_dir / f"{rec.canonical_video_id}.parquet"
    d = pd.read_parquet(vp, columns=["proposal_id"] + features)
    hit = d[d.proposal_id == rec.proposal_id]
    if len(hit) != 1:
        continue
    x = hit.iloc[0][features].to_dict()
    x.update({
        "canonical_video_id": rec.canonical_video_id, "fold": int(rec.fold),
        "label": int(rec.label), "full_oof_probability": float(rec.probability),
    })
    rows.append(x)
shap_frame = pd.DataFrame(rows)
assert shap_frame.canonical_video_id.nunique() == len(selected_rows)
for c in features:
    shap_frame[c] = pd.to_numeric(shap_frame[c], errors="raise")

full_model_dir = V13 / "v13_b_clean_132_training/training/model_runs/lgbm_05"
red_model_dir = V13 / "v13_d_feature_ablation_reduction/completion_01/reduced_model"
models = {
    "full_132": ([str(x) for x in features], [full_model_dir / f"model_fold_{k}.joblib" for k in range(3)]),
    "reduced_52": (proposal["features"], [red_model_dir / f"lightgbm_reduced_fold_{k}.joblib" for k in range(3)]),
}
family_rows, feature_rows = [], []
for model_id, (model_features, fold_paths) in models.items():
    all_contrib = []
    all_probs = []
    for k, mp in enumerate(fold_paths):
        idx = shap_frame.fold.eq(k)
        model = joblib.load(mp)
        X = shap_frame.loc[idx, model_features]
        contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]
        prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
        cf = pd.DataFrame(contrib, columns=model_features, index=shap_frame.index[idx])
        all_contrib.append(cf)
        all_probs.append(pd.Series(prob, index=shap_frame.index[idx]))
    contrib = pd.concat(all_contrib).sort_index()
    probs = pd.concat(all_probs).sort_index()
    outcome = np.select(
        [(shap_frame.label.eq(1) & probs.ge(.5)), (shap_frame.label.eq(0) & probs.ge(.5)),
         (shap_frame.label.eq(1) & probs.lt(.5))],
        ["TP", "FP", "FN"], default="TN"
    )
    fam_map = {f: paper_family(f) for f in model_features}
    for k in range(3):
        idx = shap_frame.fold.eq(k)
        raw_imp = {g: float(contrib.loc[idx, [f for f in model_features if fam_map[f] == g]].abs().mean().sum())
                   for g in sorted(set(fam_map.values()))}
        denom = sum(raw_imp.values()) or 1.0
        for g, val in raw_imp.items():
            family_rows.append({
                "model_id": model_id, "fold": k, "feature_family": g,
                "feature_count": sum(fam_map[f] == g for f in model_features),
                "importance_raw": val, "importance_normalized": val / denom,
            })
    for f in model_features:
        feature_rows.append({
            "model_id": model_id, "feature": f, "position": int(sem.set_index("feature").loc[f, "position"]),
            "feature_family": fam_map[f], "mean_abs_shap": float(contrib[f].abs().mean()),
        })
    temp = shap_frame[["canonical_video_id", "fold"]].copy()
    temp["outcome"] = outcome
    for g in sorted(set(fam_map.values())):
        cols = [f for f in model_features if fam_map[f] == g]
        mag = contrib[cols].abs().sum(axis=1)
        for oc in ["TP", "FP", "FN", "TN"]:
            mask = outcome == oc
            family_rows.append({
                "model_id": model_id, "fold": "all", "feature_family": g,
                "feature_count": len(cols), "outcome": oc,
                "outcome_mean_abs_shap": float(mag[mask].mean()) if mask.any() else np.nan,
                "outcome_support": int(mask.sum()),
            })
fam_df = pd.DataFrame(family_rows)
feat_df = pd.DataFrame(feature_rows)
fam_df.to_csv(OUT / "feature_family_shap_summary.csv", index=False)
feat_df.to_csv(OUT / "feature_level_shap_summary.csv", index=False)

fold_only = fam_df[fam_df.fold.astype(str).isin(["0", "1", "2"])].copy()
fold_only["fold"] = fold_only.fold.astype(int)
agg = fold_only.groupby(["model_id", "feature_family"]).agg(
    normalized_importance_mean=("importance_normalized", "mean"),
    normalized_importance_std=("importance_normalized", "std"),
    feature_count=("feature_count", "first"),
).reset_index()
agg["coefficient_of_variation"] = agg.normalized_importance_std / agg.normalized_importance_mean.replace(0, np.nan)
outcomes = fam_df[fam_df.fold.eq("all")].pivot_table(
    index=["model_id", "feature_family"], columns="outcome", values="outcome_mean_abs_shap"
).reset_index()
family_paper = agg.merge(outcomes, on=["model_id", "feature_family"], how="left")
family_paper["tp_minus_fp_signed_contrast"] = family_paper.get("TP", np.nan) - family_paper.get("FP", np.nan)
family_paper.to_csv(OUT / "feature_family_paper_statistics.csv", index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

# Figure 1: pipeline
fig, ax = plt.subplots(figsize=(7.0, 1.58))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
boxes = [
    (.01, .22, .12, .55, "Fixed-camera\nvideo", "#e8eef7"),
    (.16, .22, .14, .55, "YOLOv8s\nvehicles", "#dbeafe"),
    (.33, .22, .16, .55, "BoT-SORT + ReID\npersistent tracks", "#dbeafe"),
    (.52, .22, .17, .55, "Causal track and\ninteraction features", "#dcfce7"),
    (.72, .22, .11, .55, "LightGBM\ntrack scores", "#fef3c7"),
    (.86, .22, .13, .55, "Mean aggregation +\nhysteretic events", "#fee2e2"),
]
for x, y, w, h, label, color in boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=color, ec="#334155", lw=.8))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7.4)
for a, b in zip(boxes[:-1], boxes[1:]):
    ax.annotate("", xy=(b[0]-.006, .495), xytext=(a[0]+a[2]+.006, .495),
                arrowprops=dict(arrowstyle="->", lw=.9, color="#475569"))
ax.text(.325, .08, "fixed upstream perception", ha="center", fontsize=7, color="#475569")
ax.text(.605, .08, "explicit 132- or 52-feature contract", ha="center", fontsize=7, color="#475569")
ax.text(.855, .08, "calibration-frozen decision layer", ha="center", fontsize=7, color="#475569")
fig.tight_layout(pad=.15)
fig.savefig(FIG / "figure_pipeline.pdf", bbox_inches="tight")
fig.savefig(FIG / "figure_pipeline.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: family inclusion and attribution heatmap
families = ["geometry", "motion", "direction", "shape variation", "temporal summaries",
            "reliability and missingness", "pair interaction"]
matrix = []
for g in families:
    row = []
    for m in ["reduced_52", "full_132"]:
        r = family_paper[(family_paper.model_id == m) & (family_paper.feature_family == g)].iloc[0]
        row += [r.feature_count / (52 if m == "reduced_52" else 132), r.normalized_importance_mean,
                r.tp_minus_fp_signed_contrast]
    matrix.append(row)
mat = np.asarray(matrix)
col_labels = ["52 count\nshare", "52 norm.\n|SHAP|", "52 TP-FP\ncontrast",
              "132 count\nshare", "132 norm.\n|SHAP|", "132 TP-FP\ncontrast"]
scaled = mat.copy()
for j in [2, 5]:
    mx = np.nanmax(np.abs(scaled[:, j]))
    scaled[:, j] = scaled[:, j] / mx if mx else scaled[:, j]
fig, ax = plt.subplots(figsize=(6.9, 2.45))
sns.heatmap(scaled, cmap="vlag", center=0, annot=mat, fmt=".2f", linewidths=.5,
            yticklabels=families, xticklabels=col_labels, cbar_kws={"label": "column-scaled value"}, ax=ax,
            annot_kws={"fontsize": 6.5})
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_title("Training-OOF feature-family inclusion and model attribution")
fig.tight_layout()
fig.savefig(FIG / "figure_feature_family_heatmap.pdf", bbox_inches="tight")
fig.savefig(FIG / "figure_feature_family_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Supplement: ordered feature-level fold attribution
full_pivot = feat_df.pivot(index="feature", columns="model_id", values="mean_abs_shap")
full_pivot = full_pivot.reindex(features).fillna(0)
arr = full_pivot[["reduced_52", "full_132"]].to_numpy()
arr = arr / np.maximum(arr.sum(axis=0, keepdims=True), 1e-12)
fig, ax = plt.subplots(figsize=(5.4, 20))
sns.heatmap(arr, cmap="mako", yticklabels=features, xticklabels=["Reduced 52", "Full 132"],
            cbar_kws={"label": "normalized mean |SHAP|"}, ax=ax)
ax.tick_params(axis="y", labelsize=5)
ax.set_title("Complete ordered training-OOF feature attribution")
fig.tight_layout()
fig.savefig(FIG / "figure_feature_level_heatmap_supplement.pdf", bbox_inches="tight")
fig.savefig(FIG / "figure_feature_level_heatmap_supplement.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Calibration transfer: one validation point for the fixed selected policy.
plot_cal = cal[(cal.model_id == "trackfeat132") & (cal.phase == "detail")].copy()
plot_front = plot_cal[plot_cal.pareto_frontier].sort_values("fph")
sel = plot_cal[plot_cal.selected_calibration_policy].iloc[0]
fig, ax = plt.subplots(figsize=(3.35, 2.45))
ax.scatter(plot_cal.fph, plot_cal.recall, s=13, alpha=.28, color="#64748b", label="calibration policies")
ax.plot(plot_front.fph, plot_front.recall, "-o", lw=1.2, ms=3, color="#2563eb", label="calibration Pareto")
ax.scatter([sel.fph], [sel.recall], marker="*", s=95, color="#f59e0b", edgecolor="black", linewidth=.5, label="selected on calibration")
ax.scatter([5.0310], [0.5434782609], marker="D", s=40, color="#dc2626", label="same policy on validation")
ax.annotate("fixed transfer", xy=(5.0310, .54348), xytext=(6.4, .49),
            arrowprops=dict(arrowstyle="->", lw=.7), fontsize=7)
ax.set_xlim(left=-.3); ax.set_ylim(0, 1)
ax.set_xlabel("Normal false events per camera-hour")
ax.set_ylabel("Event recall")
ax.legend(fontsize=6.2, frameon=True)
fig.tight_layout()
fig.savefig(FIG / "figure_calibration_transfer.pdf", bbox_inches="tight")
fig.savefig(FIG / "figure_calibration_transfer.png", dpi=300, bbox_inches="tight")
plt.close(fig)

figure_inputs = {
    "figure_pipeline": [frozen_lineage_path, model_lineage_path, tracker_lineage_path, reid_lineage_path],
    "figure_feature_family_heatmap": [oof_path, proposal_path, sem_path],
    "figure_feature_level_heatmap_supplement": [oof_path, proposal_path, sem_path],
    "figure_calibration_transfer": [cal_path, sel_path, full_dec_path],
}
for name, inputs in figure_inputs.items():
    write_json(FIG / f"{name}_provenance.json", {
        "created_utc": NOW,
        "input_paths": [str(p.relative_to(ROOT)) for p in inputs],
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "generation_command": f"/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python {OUT.relative_to(ROOT)}/control/build_evidence.py",
        "population": (
            "frozen training OOF; one maximum-OOF-score row per video, grouped by frozen fold"
            if "heatmap" in name else
            "342-video calibration registry and one fixed-policy point on 332-video sealed validation"
            if name == "figure_calibration_transfer" else
            "frozen executable lineage"
        ),
        "status": "report-only model attribution" if "heatmap" in name else "primary/diagnostic as labelled",
        "output_pdf": f"figures/{name}.pdf",
        "output_sha256": sha(FIG / f"{name}.pdf"),
    })

# ---------------------------------------------------------------------------
# A6. Timestamp diagnostic and paper evidence ledgers
# ---------------------------------------------------------------------------
ts_path = TIMESTAMP / "calibration_timestamp_results.csv"
ts = pd.read_csv(ts_path)
tsa = ts[ts.rule_id == "TSA"].iloc[0]
tsc = ts[ts.rule_id == "TSC"].iloc[0]
(OUT / "timestamp_diagnostic_summary.md").write_text(
    "# Calibration-only timestamp diagnostic\n\n"
    f"The frozen event-time control produced TP={int(tsa.tp)}, recall={tsa.recall:.4f}, "
    f"and {int(tsa.wrong_window_alerts)} wrong-window alerts on calibration. Reassigning "
    f"each already-finalized event to its episode midpoint (TSC) produced TP={int(tsc.tp)}, "
    f"recall={tsc.recall:.4f}, and {int(tsc.wrong_window_alerts)} wrong-window alerts; "
    f"it recovered {int(tsc.recovered_d4_count)} D4 cases but lost {int(tsc.lost_frozen_tp_count)} "
    "existing true positive. TSC was not selected because the frozen calibration rule "
    "required zero loss of existing true positives. It was not evaluated on validation "
    "and is a calibration diagnostic only.\n"
)

stats = [
    ("sotad_validation_full_precision", .5102040816, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_recall", .5434782609, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_f1", .5263157895, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_fph", 5.0310, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_tp", 25, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_normal_fp_events", 8, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_fn", 21, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_full_wrong_window", 16, "sealed validation", "primary", full_dec_path),
    ("sotad_validation_reduced_precision", .4716981132, "sealed validation", "primary", red_dec_path),
    ("sotad_validation_reduced_recall", .5434782609, "sealed validation", "primary", red_dec_path),
    ("sotad_validation_reduced_f1", .5050505051, "sealed validation", "primary", red_dec_path),
    ("sotad_validation_reduced_fph", 6.2887, "sealed validation", "primary", red_dec_path),
    ("sotad_validation_reduced_normal_fp_events", 10, "sealed validation", "primary", red_dec_path),
    ("sotad_validation_reduced_wrong_window", 18, "sealed validation", "primary", red_dec_path),
    ("calibration_selected_recall", selected["recall"], "342-video calibration", "selection evidence", sel_path),
    ("calibration_selected_fph", selected["fph"], "342-video calibration", "selection evidence", sel_path),
    ("timestamp_tsc_recall", tsc.recall, "342-video calibration", "diagnostic only", ts_path),
]
stat_df = pd.DataFrame(stats, columns=["statistic_id", "value", "population", "status", "source_path"])
stat_df["source_path"] = stat_df.source_path.map(lambda p: str(Path(p).relative_to(ROOT)))
stat_df["source_sha256"] = stat_df.source_path.map(lambda p: sha(ROOT / p))
stat_df["generation_command"] = f"/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python {OUT.relative_to(ROOT)}/control/build_evidence.py"
stat_df.to_csv(OUT / "paper_statistic_ledger.csv", index=False)

claims = [
    ("C01", "The executable upstream checkpoint is YOLOv8s-derived and was fixed before downstream evaluation.", "upstream_perception_contract.csv", "verified lineage"),
    ("C02", "TrackFeat uses an exact ordered 132-feature causal contract and a frozen 52-feature development reduction.", "full_132_feature_contract.csv;reduced_52_feature_contract.csv", "verified contract"),
    ("C03", "The 52-feature reduction was selected using training-OOF stability and redundancy evidence, not sealed validation.", "feature_selection_lineage.md", "development selection"),
    ("C04", "Full-132 achieved P=.5102, R=.5435, F1=.5263 and 5.031 FPH on sealed validation.", "paper_statistic_ledger.csv", "primary validation"),
    ("C05", "Reduced-52 had the same validation recall but more normal false events and wrong-window alerts.", "paired_validation_bootstrap.csv;paper_statistic_ledger.csv", "primary validation"),
    ("C06", "The event policy was selected only on calibration by the frozen recall-first rule.", "calibration_policy_registry.csv;calibration_pareto_frontier.csv", "selection evidence"),
    ("C07", "TSC improved calibration recall but lost one established TP, was not selected, and was never evaluated on validation.", "timestamp_diagnostic_summary.md", "diagnostic only"),
    ("C08", "Feature-family heatmaps report training-OOF model attribution, not causal effects.", "feature_family_shap_summary.csv", "report-only attribution"),
]
pd.DataFrame(claims, columns=["claim_id", "claim", "evidence_artifact", "evidence_status"]).to_csv(
    OUT / "paper_claim_evidence_map.csv", index=False
)

manifest_inputs = [
    frozen_lineage_path, model_lineage_path, tracker_lineage_path, reid_lineage_path,
    det_ckpt, reid_ckpt, sem_path, proposal_path, full_dec_path, red_dec_path,
    cal_path, sel_path, oof_path, ts_path,
]
manifest_outputs = [
    OUT / "upstream_perception_contract.csv", OUT / "upstream_selection_evidence.md",
    OUT / "full_132_feature_contract.csv", OUT / "reduced_52_feature_contract.csv",
    OUT / "feature_contract_comparison.csv", OUT / "feature_selection_lineage.md",
    OUT / "paired_validation_bootstrap.csv", OUT / "calibration_policy_registry.csv",
    OUT / "calibration_pareto_frontier.csv", OUT / "feature_family_shap_summary.csv",
    OUT / "feature_level_shap_summary.csv", OUT / "feature_family_paper_statistics.csv",
    OUT / "paper_statistic_ledger.csv", OUT / "paper_claim_evidence_map.csv",
    OUT / "timestamp_diagnostic_summary.md",
] + [FIG / f"{n}.pdf" for n in figure_inputs]
write_json(OUT / "paper_evidence_manifest.json", {
    "created_utc": NOW,
    "experiment_type": "frozen-evidence publication closure; no training or tuning",
    "official_test_accessed": False,
    "validation_predictions_rescored": False,
    "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in manifest_inputs},
    "output_hashes": {str(p.relative_to(OUT)): sha(p) for p in manifest_outputs},
    "bootstrap": {"replicates": N_BOOT, "seed": SEED, "unit": "paired video", "strata": ["accident", "normal"]},
    "attribution": {
        "population": "frozen training OOF only",
        "sampling": "one maximum-full-OOF-score row per video",
        "folds": 3,
        "interpretation": "model attribution; not causal effect",
    },
})
print(json.dumps({
    "status": "evidence_complete",
    "output_root": str(OUT),
    "validation_rows": len(fd),
    "feature_count_full": len(contract),
    "feature_count_reduced": int(contract.included_reduced_52.sum()),
    "training_oof_attribution_videos": int(shap_frame.canonical_video_id.nunique()),
    "calibration_policies": len(cal),
}, indent=2))
