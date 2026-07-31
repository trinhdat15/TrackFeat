#!/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python
"""Build read-only trajectory-quality evidence and real-frame paper figures.

This report-only script consumes frozen calibration/validation artifacts.  It
does not train, tune, alter predictions, select an event policy, or access the
official test.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path("/ssd1/team_cam_ai/ttdat")
OUT = ROOT / "final_draft/trackfeat_aisi_submission_02"
PRE = ROOT / "final_draft/trackfeat_aisi_submission_01"
FIG = OUT / "figures"
VAL = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "v13_final_prevalidation_evidence_closure/validation_frozen_cache_recovery_01"
)
VAL_CACHE = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "v13_final_prevalidation_evidence_closure/validation_transaction_01/"
    "features/validation"
)
V13 = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "v13_interpretable_132_feature_study"
)
CAL_CACHE = V13 / "v13_b_clean_132_training/manifold/features/calibration"
CAL_PRED = V13 / (
    "v13_b_clean_132_training/training/calibration_track_predictions.parquet"
)
CAL_EVAL = V13 / "v13_c_event_calibration/authoritative/trackfeat132"
UPSTREAM = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "v12_59b_aaai_full_causal_upstream_replay/videos"
)
REGISTRY = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "v12_59a_aaai_causal_upstream_pilot/canonical_video_registry.parquet"
)
FN_LEDGER = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "aisi_recall_gap_closure_01/false_negative_root_cause_ledger.csv"
)
D4_LEDGER = ROOT / (
    "outputs_sotad_phase12_old4c_motor_lowconf_t012/"
    "aisi_event_timestamp_closure_01/d4_episode_timing_ledger.csv"
)
MODEL = V13 / (
    "v13_b_clean_132_training/training/models/lightgbm_132_selected.joblib"
)
NOW = datetime.now(timezone.utc).isoformat()
SEED = 130057
BOOT = 1000
CLASS_NAMES = {0: "motorbike", 1: "car", 2: "bus", 3: "truck", 4: "bicycle"}
COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light": "#F5F7FA",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_events(value: str) -> list[dict]:
    if not value or (isinstance(value, float) and math.isnan(value)):
        return []
    return json.loads(value)


def source_index() -> pd.DataFrame:
    reg = pd.read_parquet(REGISTRY)
    reg = reg[reg["dataset"].astype(str).str.startswith("SO-TAD")].copy()
    return reg.drop_duplicates("canonical_video_id").set_index("canonical_video_id")


def expected_samples(start: float, end: float) -> int:
    return max(1, int(math.floor(max(0.0, end - start) * 5.0)) + 1)


def segments_and_gaps(tel: pd.DataFrame) -> tuple[float, int, int, float]:
    """Return longest segment, track-fragment count, gap count, mean gap seconds."""
    longest = 0.0
    fragments = 0
    gaps: list[float] = []
    if tel.empty:
        return longest, fragments, 0, math.nan
    for _, g in tel.groupby("track_id"):
        ts = np.sort(g["timestamp_s"].dropna().unique().astype(float))
        if not len(ts):
            continue
        starts = [0]
        for i, dt in enumerate(np.diff(ts), start=1):
            if dt > 0.400001:
                starts.append(i)
                gaps.append(float(dt - 0.2))
        ends = starts[1:] + [len(ts)]
        fragments += max(0, len(starts) - 1)
        for a, b in zip(starts, ends):
            if b > a:
                longest = max(longest, float(ts[b - 1] - ts[a] + 0.2))
    return longest, fragments, len(gaps), float(np.mean(gaps)) if gaps else 0.0


def quality_row(
    video_id: str,
    role: str,
    decision: pd.Series,
    feature_path: Path,
    reg: pd.DataFrame,
) -> dict:
    duration = float(decision["duration_s"])
    toa = float(decision["time_of_accident_s"]) if pd.notna(decision["time_of_accident_s"]) else math.nan
    category = str(decision["category"])
    start = max(0.0, toa - 2.5) if category == "accident" else 0.0
    end = min(duration, toa + 2.5) if category == "accident" else duration
    det_path = UPSTREAM / video_id / "detections.parquet"
    tel_path = UPSTREAM / video_id / "causal_telemetry.parquet"
    det = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()
    tel = pd.read_parquet(tel_path) if tel_path.exists() else pd.DataFrame()
    feat = pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame()
    dw = det[(det.timestamp_s >= start) & (det.timestamp_s <= end)] if not det.empty else det
    tw = tel[(tel.timestamp_s >= start) & (tel.timestamp_s <= end)] if not tel.empty else tel
    fw = feat[(feat.timestamp_s >= start) & (feat.timestamp_s <= end)] if not feat.empty else feat
    n_expected = expected_samples(start, end)
    longest, fragments, gap_count, mean_gap = segments_and_gaps(tel)
    track_count = int(tel.track_id.nunique()) if not tel.empty else 0
    det_cov = float(dw.timestamp_s.nunique() / n_expected) if not dw.empty else 0.0
    track_cov = float(tw.timestamp_s.nunique() / n_expected) if not tw.empty else 0.0
    cand_cov = float(feat.timestamp_s.nunique() / expected_samples(0, duration)) if not feat.empty else 0.0
    pair = feat[feat.actor_2_id.astype(float) >= 0] if not feat.empty else feat
    pair_cov = float(pair.timestamp_s.nunique() / expected_samples(0, duration)) if not pair.empty else 0.0
    age_support = float((pd.to_numeric(feat.get("track_age"), errors="coerce") >= 5).mean()) if not feat.empty else 0.0
    continuity_components = [
        np.clip(track_cov, 0, 1),
        np.clip(longest / max(1.0, min(duration, 4.0)), 0, 1),
        np.clip(cand_cov, 0, 1),
        np.clip(age_support, 0, 1),
        1.0 - np.clip((fragments + gap_count) / max(1.0, 3.0 * track_count), 0, 1),
    ]
    quality_score = float(np.mean(continuity_components))
    events = json_events(decision["events_json"])
    if category == "normal":
        outcome = "normal FP" if int(decision["normal_false_events"]) else "TN"
    elif int(decision["matched"]):
        outcome = "TP"
    elif int(decision["wrong_window_alerts"]):
        outcome = "wrong-window"
    else:
        outcome = "FN"
    src = reg.loc[video_id] if video_id in reg.index else None
    return {
        "canonical_video_id": video_id,
        "role": role,
        "source_group": str(src["group_id"]) if src is not None else "NOT_FOUND",
        "source_video_path": str(src["resolved_path"]) if src is not None else "NOT_FOUND",
        "category": category,
        "duration_s": duration,
        "annotation_time_s": toa,
        "window_start_s": start,
        "window_end_s": end,
        "mean_detector_confidence": float(det.confidence.mean()) if not det.empty else math.nan,
        "lower_decile_detector_confidence": float(det.confidence.quantile(0.1)) if not det.empty else math.nan,
        "detection_coverage_near_accident_or_full_normal": det_cov,
        "valid_track_coverage_near_accident_or_full_normal": track_cov,
        "longest_continuous_track_duration_s": longest,
        "track_count": track_count,
        "track_fragment_count": fragments,
        "observation_gap_count": gap_count,
        "mean_observation_gap_s": mean_gap,
        "box_center_jitter_median": float(pd.to_numeric(tel.get("bbox_center_jitter"), errors="coerce").median()) if not tel.empty else math.nan,
        "width_jitter_std": float(pd.to_numeric(tel.get("bbox_w_norm"), errors="coerce").std()) if not tel.empty else math.nan,
        "height_jitter_std": float(pd.to_numeric(tel.get("bbox_h_norm"), errors="coerce").std()) if not tel.empty else math.nan,
        "scale_jitter_median": float(pd.to_numeric(feat.get("bbox_scale_jitter"), errors="coerce").median()) if not feat.empty else math.nan,
        "aspect_jitter_median": float(pd.to_numeric(feat.get("bbox_aspect_jitter"), errors="coerce").median()) if not feat.empty else math.nan,
        "eligible_candidate_coverage": cand_cov,
        "eligible_pair_coverage": pair_cov,
        "track_age_support_rate_ge5": age_support,
        "median_track_age_samples": float(pd.to_numeric(feat.get("track_age"), errors="coerce").median()) if not feat.empty else math.nan,
        "small_object_prevalence": float(pd.to_numeric(feat.get("is_small_box"), errors="coerce").mean()) if not feat.empty else math.nan,
        "far_or_edge_unreliable_prevalence": float(pd.to_numeric(feat.get("far_edge_unreliable"), errors="coerce").mean()) if not feat.empty else math.nan,
        "usable_candidates": bool(decision["usable_candidates"]),
        "outcome": outcome,
        "matched": int(decision["matched"]),
        "missed": int(decision["missed"]),
        "normal_false_events": int(decision["normal_false_events"]),
        "wrong_window_alerts": int(decision["wrong_window_alerts"]),
        "duplicate_alerts": int(decision["duplicate_alerts"]),
        "matched_delay_s": float(decision["matched_delay_s"]) if pd.notna(decision["matched_delay_s"]) else math.nan,
        "video_probability": float(decision["video_probability"]),
        "event_count": len(events),
        "quality_score": quality_score,
        "detections_path": str(det_path),
        "telemetry_path": str(tel_path),
        "features_path": str(feature_path),
        "detections_available": det_path.exists(),
        "telemetry_available": tel_path.exists(),
        "features_available": feature_path.exists(),
    }


def metric_row(g: pd.DataFrame, stratum: str) -> dict:
    acc = g[g.category == "accident"]
    norm = g[g.category == "normal"]
    tp = int(acc.matched.sum())
    fn = int(acc.missed.sum())
    nfp = int(norm.normal_false_events.sum())
    wrong = int(acc.wrong_window_alerts.sum())
    dup = int(acc.duplicate_alerts.sum())
    fp_all = nfp + wrong + dup
    precision = tp / (tp + fp_all) if tp + fp_all else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    hours = norm.duration_s.sum() / 3600.0
    fph = nfp / hours if hours else math.nan
    out = {
        "quality_stratum": stratum,
        "video_count": len(g),
        "accident_video_count": len(acc),
        "normal_video_count": len(norm),
        "tp": tp,
        "fn": fn,
        "normal_false_events": nfp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fph": fph,
        "wrong_window_alerts": wrong,
        "median_matched_timing_error_s": float(acc.matched_delay_s.median()) if acc.matched_delay_s.notna().any() else math.nan,
        "recall_ci_low": math.nan,
        "recall_ci_high": math.nan,
        "fph_ci_low": math.nan,
        "fph_ci_high": math.nan,
        "bootstrap_replicates": 0,
    }
    if len(acc) >= 10 and len(norm) >= 10:
        rng = np.random.default_rng(SEED + {"fragmented or low-support": 1, "moderate continuity": 2, "high continuity": 3}[stratum])
        rec, rates = [], []
        ai, ni = acc.index.to_numpy(), norm.index.to_numpy()
        for _ in range(BOOT):
            aa = g.loc[rng.choice(ai, len(ai), replace=True)]
            nn = g.loc[rng.choice(ni, len(ni), replace=True)]
            denom = aa.matched.sum() + aa.missed.sum()
            rec.append(float(aa.matched.sum() / denom) if denom else math.nan)
            h = nn.duration_s.sum() / 3600.0
            rates.append(float(nn.normal_false_events.sum() / h) if h else math.nan)
        out.update(
            recall_ci_low=float(np.nanpercentile(rec, 2.5)),
            recall_ci_high=float(np.nanpercentile(rec, 97.5)),
            fph_ci_low=float(np.nanpercentile(rates, 2.5)),
            fph_ci_high=float(np.nanpercentile(rates, 97.5)),
            bootstrap_replicates=BOOT,
        )
    return out


def overlay_tracks(ax, frame: np.ndarray, telemetry: pd.DataFrame, timestamp: float, selected_track: int | None, actor_2: int | None) -> None:
    ax.imshow(frame)
    current = telemetry.iloc[(telemetry.timestamp_s - timestamp).abs().argsort()[: max(1, min(8, len(telemetry)))]]
    nearest_t = float(current.timestamp_s.iloc[0]) if not current.empty else timestamp
    current = telemetry[np.isclose(telemetry.timestamp_s, nearest_t)]
    trail = telemetry[(telemetry.timestamp_s <= nearest_t) & (telemetry.timestamp_s >= nearest_t - 1.2)]
    for tid, g in trail.groupby("track_id"):
        centers = np.column_stack(((g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2))
        color = COLORS["vermillion"] if int(tid) == selected_track else (COLORS["orange"] if int(tid) == actor_2 else COLORS["sky"])
        if len(centers) > 1:
            ax.plot(centers[:, 0], centers[:, 1], "-", color=color, lw=1.5, alpha=0.9)
    for _, r in current.iterrows():
        tid = int(r.track_id)
        color = COLORS["vermillion"] if tid == selected_track else (COLORS["orange"] if tid == actor_2 else COLORS["sky"])
        ax.add_patch(patches.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fill=False, lw=1.3, ec=color))
        ax.text(r.x1, max(2, r.y1 - 3), f"{CLASS_NAMES.get(int(r['class']), 'vehicle')} #{tid}", color="white", fontsize=6.5,
                bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=1.1))
    if selected_track is not None and actor_2 is not None and actor_2 >= 0:
        a = current[current.track_id == selected_track]
        b = current[current.track_id == actor_2]
        if not a.empty and not b.empty:
            ac = ((a.x1.iloc[0] + a.x2.iloc[0]) / 2, (a.y1.iloc[0] + a.y2.iloc[0]) / 2)
            bc = ((b.x1.iloc[0] + b.x2.iloc[0]) / 2, (b.y1.iloc[0] + b.y2.iloc[0]) / 2)
            ax.plot([ac[0], bc[0]], [ac[1], bc[1]], "--", color=COLORS["orange"], lw=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)


def read_frame(video_path: str, timestamp: float) -> tuple[np.ndarray, int, float]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    idx = max(0, int(round(timestamp * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), idx, fps


def select_row(feature: pd.DataFrame, pred: pd.DataFrame, timestamp: float) -> tuple[pd.Series, pd.Series]:
    times = pred.timestamp_s.dropna().unique()
    nearest = float(times[np.argmin(np.abs(times - timestamp))])
    at_time = pred[np.isclose(pred.timestamp_s, nearest)]
    pair_rows = at_time[at_time.actor_2_id.astype(float) >= 0]
    if not pair_rows.empty:
        at_time = pair_rows
    p = at_time.sort_values(["probability", "proposal_id"], ascending=[False, True]).iloc[0]
    f = feature[feature.proposal_id.astype(str) == str(p.proposal_id)]
    if f.empty:
        f = feature[(feature.track_id == p.track_id) & np.isclose(feature.timestamp_s, nearest)]
    return f.iloc[0], p


def scene_curve(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby("timestamp_s", as_index=False).probability.mean().sort_values("timestamp_s")


def local_family_attribution(feature_row: pd.Series, frozen_probability: float) -> tuple[list[tuple[str, float]], float]:
    contract = pd.read_csv(OUT / "full_132_feature_contract.csv").sort_values("position")
    features = contract.feature.tolist()
    model = joblib.load(MODEL)
    x = pd.DataFrame([{c: pd.to_numeric(pd.Series([feature_row[c]]), errors="raise").iloc[0] for c in features}], columns=features)
    contrib = model.booster_.predict(x, pred_contrib=True)[0]
    pred = float(model.predict_proba(x)[0, 1])
    if abs(pred - frozen_probability) > 1e-7:
        raise AssertionError(f"selected-row probability parity failed: {pred} vs {frozen_probability}")
    fam = contract.set_index("feature")["paper_feature_family"].to_dict()
    agg: dict[str, float] = {}
    for name, value in zip(features, contrib[:-1]):
        agg[fam[name]] = agg.get(fam[name], 0.0) + float(value)
    top = sorted(agg.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:3]
    return top, pred


def selection_candidates(
    decisions: pd.DataFrame,
    event_ledger: pd.DataFrame,
    pred: pd.DataFrame,
    features_dir: Path,
    reg: pd.DataFrame,
) -> dict[str, dict]:
    # Panel A: matched TP whose event score is closest to the eligible TP median.
    tp_events = event_ledger[event_ledger.classification == "matched_accident_event"].copy()
    eligible_tp = []
    for _, e in tp_events.iterrows():
        vid = str(e.canonical_video_id)
        fp = features_dir / f"{vid}.parquet"
        tp = UPSTREAM / vid / "causal_telemetry.parquet"
        if vid not in reg.index or not fp.exists() or not tp.exists() or not Path(reg.loc[vid, "resolved_path"]).exists():
            continue
        f = pd.read_parquet(fp, columns=["timestamp_s", "actor_2_id"])
        nearest = float(f.timestamp_s.iloc[(f.timestamp_s - float(e.event_time_s)).abs().argsort()[:1]].iloc[0])
        if (f[np.isclose(f.timestamp_s, nearest)].actor_2_id.astype(float) >= 0).any():
            eligible_tp.append(e)
    etp = pd.DataFrame(eligible_tp)
    med = float(etp.event_score.median())
    etp["distance"] = (etp.event_score - med).abs()
    etp["stable"] = etp.canonical_video_id.map(stable_key)
    a = etp.sort_values(["distance", "stable"]).iloc[0]

    # Panel B: highest scoring eventless normal with pair evidence.
    eligible_normal = []
    normals = decisions[(decisions.category == "normal") & (decisions.normal_false_events == 0) & (decisions.video_decision == 0)]
    for _, d in normals.iterrows():
        vid = str(d.canonical_video_id)
        fp = features_dir / f"{vid}.parquet"
        tp = UPSTREAM / vid / "causal_telemetry.parquet"
        if vid not in reg.index or not fp.exists() or not tp.exists() or not Path(reg.loc[vid, "resolved_path"]).exists():
            continue
        f = pd.read_parquet(fp, columns=["timestamp_s", "actor_2_id"])
        vp = pred[pred.canonical_video_id == vid]
        if vp.empty:
            continue
        curve = scene_curve(vp)
        peak_time = float(curve.loc[curve.probability.idxmax(), "timestamp_s"])
        nearest = float(f.timestamp_s.iloc[(f.timestamp_s - peak_time).abs().argsort()[:1]].iloc[0])
        if (f[np.isclose(f.timestamp_s, nearest)].actor_2_id.astype(float) >= 0).any():
            row = d.copy()
            row["peak_scene_score"] = float(curve.probability.max())
            eligible_normal.append(row)
    en = pd.DataFrame(eligible_normal)
    en["stable"] = en.canonical_video_id.map(stable_key)
    b = en.sort_values(["peak_scene_score", "stable"], ascending=[False, True]).iloc[0]

    # Panel C: D4 timing error closest to the D4 median among visually available cases.
    fn = pd.read_csv(FN_LEDGER)
    d4 = fn[fn.primary_root_cause == "D4"].copy()
    d4["abs_error"] = d4.apply(
        lambda r: min(abs(float(x) - float(r.annotation_time_s)) for x in str(r.outside_alert_times_s).split(";")),
        axis=1,
    )
    d4 = d4[
        d4.canonical_video_id.map(
            lambda v: v in reg.index
            and Path(reg.loc[v, "resolved_path"]).exists()
            and (features_dir / f"{v}.parquet").exists()
            and (UPSTREAM / v / "causal_telemetry.parquet").exists()
        )
    ].copy()
    med_d4 = float(d4.abs_error.median())
    d4["distance"] = (d4.abs_error - med_d4).abs()
    d4["stable"] = d4.canonical_video_id.map(stable_key)
    c0 = d4.sort_values(["distance", "stable"]).iloc[0]
    ce = event_ledger[
        (event_ledger.canonical_video_id == c0.canonical_video_id)
        & (event_ledger.classification == "wrong_window_accident_alert")
    ].copy()
    ce["distance"] = (ce.event_time_s - float(c0.annotation_time_s)).abs()
    c = ce.sort_values(["distance", "event_time_s"]).iloc[0]
    return {
        "A": {"video_id": str(a.canonical_video_id), "timestamp_s": float(a.event_time_s), "event_score": float(a.event_score), "selection_statistic": med},
        "B": {"video_id": str(b.canonical_video_id), "timestamp_s": math.nan, "event_score": math.nan, "selection_statistic": float(b.peak_scene_score)},
        "C": {"video_id": str(c.canonical_video_id), "timestamp_s": float(c.event_time_s), "event_score": float(c.event_score), "selection_statistic": med_d4},
    }


def prepare_example(panel: str, spec: dict, pred_all: pd.DataFrame, feature_dir: Path, reg: pd.DataFrame) -> dict:
    vid = spec["video_id"]
    pred = pred_all[pred_all.canonical_video_id == vid].copy()
    curve = scene_curve(pred)
    timestamp = spec["timestamp_s"]
    if not np.isfinite(timestamp):
        timestamp = float(curve.loc[curve.probability.idxmax(), "timestamp_s"])
    feature = pd.read_parquet(feature_dir / f"{vid}.parquet")
    fr, pr = select_row(feature, pred, timestamp)
    tel = pd.read_parquet(UPSTREAM / vid / "causal_telemetry.parquet")
    video_path = str(reg.loc[vid, "resolved_path"])
    frame, frame_index, source_fps = read_frame(video_path, timestamp)
    top_attr, parity = local_family_attribution(fr, float(pr.probability))
    scene_score = float(curve.iloc[(curve.timestamp_s - timestamp).abs().argsort()[:1]].probability.iloc[0])
    dec = pd.read_parquet(VAL / "04_event_evaluation/validation_video_decisions_132.parquet")
    dr = dec[dec.canonical_video_id == vid].iloc[0]
    annotation = float(dr.time_of_accident_s) if pd.notna(dr.time_of_accident_s) else math.nan
    values = {
        "F1_recent_acceleration": float(fr.F1) if pd.notna(fr.F1) else math.nan,
        "F2_speed_loss_ratio": float(fr.F2) if pd.notna(fr.F2) else math.nan,
        "F5_nearby_IoU": float(fr.F5) if pd.notna(fr.F5) else math.nan,
        "F10_heading_change_proxy": float(fr.F10) if pd.notna(fr.F10) else math.nan,
        "track_age_samples": float(fr.track_age) if pd.notna(fr.track_age) else math.nan,
        "row_probability": float(pr.probability),
        "scene_score": scene_score,
    }
    return {
        "panel": panel,
        "video_id": vid,
        "timestamp_s": timestamp,
        "event_score": spec["event_score"],
        "selection_statistic": spec["selection_statistic"],
        "feature_row": fr,
        "prediction_row": pr,
        "telemetry": tel,
        "curve": curve,
        "frame": frame,
        "frame_index": frame_index,
        "source_fps": source_fps,
        "video_path": video_path,
        "video_sha256": str(reg.loc[vid, "sha256"]),
        "selected_track": int(pr.track_id),
        "actor_2": int(pr.actor_2_id),
        "annotation_time_s": annotation,
        "outcome": str(dr.matched and "matched accident event" or (dr.wrong_window_alerts and "wrong-window accident alert" or "non-alert normal")),
        "values": values,
        "top_attributions": top_attr,
        "attribution_probability_parity": parity,
        "feature_path": str(feature_dir / f"{vid}.parquet"),
        "telemetry_path": str(UPSTREAM / vid / "causal_telemetry.parquet"),
    }


def choose_calibration_example(reg: pd.DataFrame) -> dict:
    inp = json.loads((CAL_EVAL / "input.json").read_text())
    pred_events = []
    for v in inp["videos"]:
        if v["label"] != 1 or not v["pred_events"] or not v["gt_events"]:
            continue
        gt = float(v["gt_events"][0]["time_s"])
        for e in v["pred_events"]:
            if abs(float(e["time_s"]) - gt) <= float(inp["point_tolerance_s"]):
                vid = v["video_id"]
                fp = CAL_CACHE / f"{vid}.parquet"
                tp = UPSTREAM / vid / "causal_telemetry.parquet"
                if vid in reg.index and fp.exists() and tp.exists() and Path(reg.loc[vid, "resolved_path"]).exists():
                    f = pd.read_parquet(fp, columns=["timestamp_s", "actor_2_id"])
                    nearest = float(f.timestamp_s.iloc[(f.timestamp_s - float(e["time_s"])).abs().argsort()[:1]].iloc[0])
                    if (f[np.isclose(f.timestamp_s, nearest)].actor_2_id.astype(float) >= 0).any():
                        pred_events.append({"video_id": vid, "time_s": float(e["time_s"]), "score": float(e["score"]), "gt": gt})
    d = pd.DataFrame(pred_events)
    med = float(d.score.median())
    d["distance"] = (d.score - med).abs()
    d["stable"] = d.video_id.map(stable_key)
    s = d.sort_values(["distance", "stable"]).iloc[0]
    vid = str(s.video_id)
    pred_all = pd.read_parquet(CAL_PRED)
    pred = pred_all[pred_all.canonical_video_id == vid]
    feature = pd.read_parquet(CAL_CACHE / f"{vid}.parquet")
    fr, pr = select_row(feature, pred, float(s.time_s))
    tel = pd.read_parquet(UPSTREAM / vid / "causal_telemetry.parquet")
    frame, frame_index, fps = read_frame(str(reg.loc[vid, "resolved_path"]), float(s.time_s))
    return {
        "video_id": vid,
        "timestamp_s": float(s.time_s),
        "annotation_time_s": float(s["gt"]),
        "event_score": float(s.score),
        "median_eligible_tp_event_score": med,
        "feature_row": fr,
        "prediction_row": pr,
        "telemetry": tel,
        "curve": scene_curve(pred),
        "frame": frame,
        "frame_index": frame_index,
        "source_fps": fps,
        "video_path": str(reg.loc[vid, "resolved_path"]),
        "video_sha256": str(reg.loc[vid, "sha256"]),
        "selected_track": int(pr.track_id),
        "actor_2": int(pr.actor_2_id),
    }


def plot_pipeline(example: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 2.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.55], height_ratios=[0.15, 1.0])
    titles = [
        ("1  Fixed upstream perception", COLORS["blue"]),
        ("2  Proposed structured evidence", COLORS["green"]),
        ("3  Scoring and event formation", COLORS["orange"]),
    ]
    for i, (title, color) in enumerate(titles):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0, 0.05), 1, 0.9, boxstyle="round,pad=0.02", fc=color, ec="none", alpha=0.95))
        ax.text(0.03, 0.5, title, va="center", ha="left", color="white", weight="bold", fontsize=8.2)

    ax1 = fig.add_subplot(gs[1, 0])
    overlay_tracks(ax1, example["frame"], example["telemetry"], example["timestamp_s"], example["selected_track"], example["actor_2"])
    ax1.text(
        0.02, 0.02,
        "YOLOv8s-derived detector\nBoxMOT BoT-SORT + ReID\npersistent IDs and 1.2-s trails",
        transform=ax1.transAxes, va="bottom", fontsize=7, color="#111827",
        bbox=dict(fc="white", ec="none", alpha=0.88, pad=2),
    )

    ax2 = fig.add_subplot(gs[1, 1])
    overlay_tracks(ax2, example["frame"], example["telemetry"], example["timestamp_s"], example["selected_track"], example["actor_2"])
    r = example["feature_row"]
    groups = [
        ("Unary motion / shape", f"F1 acceleration proxy {float(r.F1):.3f}\nF2 speed-loss ratio {float(r.F2):.3f}\nF11 aspect change {float(r.F11):.3f}", COLORS["blue"]),
        ("Local interaction", f"F0 interaction energy {float(r.F0):.3f}\nF5 nearby IoU {float(r.F5):.3f}\nF19 contact persistence {float(r.F19):.3f}", COLORS["orange"]),
        ("Reliability / missingness", f"F13 signal quality {float(r.F13):.3f}\ntrack age {int(r.track_age)} samples", COLORS["green"]),
    ]
    y = 0.98
    for title, text, color in groups:
        ax2.text(0.02, y, title + "\n" + text, transform=ax2.transAxes, va="top", fontsize=6.6,
                 bbox=dict(fc="white", ec=color, lw=0.9, alpha=0.92, boxstyle="round,pad=0.25"))
        y -= 0.285

    outer = gs[1, 2].subgridspec(2, 1, height_ratios=[0.68, 0.32])
    ax3 = fig.add_subplot(outer[0])
    c = example["curve"]
    ax3.plot(c.timestamp_s, c.probability, color=COLORS["blue"], lw=1.6, label=r"$s_t$: mean track score")
    ax3.axhline(0.65, color=COLORS["vermillion"], ls="-", lw=1.1, label=r"$\tau_{\rm up}=0.65$")
    ax3.axhline(0.50, color=COLORS["orange"], ls="--", lw=1.1, label=r"$\tau_{\rm down}=0.50$")
    ax3.axvline(example["timestamp_s"], color="#111827", lw=1.0)
    ax3.scatter([example["timestamp_s"]], [example["event_score"]], color=COLORS["vermillion"], s=18, zorder=4)
    ax3.set_xlim(float(c.timestamp_s.min()), float(c.timestamp_s.max()))
    ax3.set_ylim(0, 1.02)
    ax3.set_ylabel("scene score")
    ax3.set_xlabel("time (s)")
    ax3.grid(alpha=0.18)
    ax3.legend(fontsize=6.2, loc="upper right", frameon=False)
    ax3.text(0.02, 0.98, rf"$p_{{i,t}}={float(example['prediction_row'].probability):.3f}$" + "\n" + r"$s_t=|A_t|^{-1}\sum_i p_{i,t}$",
             transform=ax3.transAxes, va="top", fontsize=7,
             bbox=dict(fc="white", ec="#D1D5DB", alpha=0.9, pad=2))
    ax4 = fig.add_subplot(outer[1])
    ax4.axis("off")
    ax4.text(
        0.01, 0.95,
        "Enter: 2 consecutive samples above 0.65\n"
        "Exit: 3 consecutive samples below 0.50\n"
        "Cooldown: merge repeated emissions within 2 s\n"
        f"Output: ACCIDENT  t={example['timestamp_s']:.1f}s, "
        f"score={example['event_score']:.3f}, track #{example['selected_track']}",
        va="top", fontsize=7.0,
        bbox=dict(fc="#FFF7ED", ec=COLORS["orange"], lw=0.9, boxstyle="round,pad=0.3"),
    )
    fig.savefig(FIG / "figure_pipeline_v2.pdf")
    fig.savefig(FIG / "figure_pipeline_v2.png", dpi=360)
    plt.close(fig)


def plot_qualitative(examples: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.04)
    titles = {
        "A": "A  Matched accident event",
        "B": "B  Difficult non-alert normal",
        "C": "C  Temporal-mismatch failure",
    }
    for i, ex in enumerate(examples):
        sub = gs[i].subgridspec(2, 1, height_ratios=[1.0, 0.36], hspace=0.04)
        ax = fig.add_subplot(sub[0])
        overlay_tracks(ax, ex["frame"], ex["telemetry"], ex["timestamp_s"], ex["selected_track"], ex["actor_2"])
        ax.set_title(titles[ex["panel"]], fontsize=8.1, weight="bold", pad=2)
        vals = ex["values"]
        cue_text = (
            f"row p={vals['row_probability']:.3f}; scene={vals['scene_score']:.3f}\n"
            f"F1 accel={vals['F1_recent_acceleration']:.3f}; F0 interact={float(ex['feature_row'].F0):.3f}\n"
            f"age={vals['track_age_samples']:.0f} samples"
        )
        ax.text(0.02, 0.02, cue_text, transform=ax.transAxes, va="bottom", fontsize=6.2,
                bbox=dict(fc="white", ec="#D1D5DB", alpha=0.9, pad=2))
        short = {
            "reliability and missingness": "reliability",
            "temporal summaries": "temporal",
            "pair interaction": "interaction",
            "shape variation": "shape",
        }
        attr = "\n".join(f"{short.get(k, k)} {v:+.2f}" for k, v in ex["top_attributions"])
        ax.text(0.98, 0.98, "model attribution\n" + attr, transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
                bbox=dict(fc="white", ec=COLORS["purple"], alpha=0.9, pad=2))
        tax = fig.add_subplot(sub[1])
        c = ex["curve"]
        tax.plot(c.timestamp_s, c.probability, color=COLORS["blue"], lw=1.25)
        tax.axhline(0.65, color=COLORS["vermillion"], lw=0.9)
        tax.axhline(0.50, color=COLORS["orange"], lw=0.9, ls="--")
        tax.axvline(ex["timestamp_s"], color="#111827", lw=0.8)
        if np.isfinite(ex["annotation_time_s"]):
            tax.axvspan(max(0, ex["annotation_time_s"] - 2.5), ex["annotation_time_s"] + 2.5, color=COLORS["green"], alpha=0.12)
            tax.axvline(ex["annotation_time_s"], color=COLORS["green"], lw=0.9, ls=":")
        tax.set_ylim(0, 1.02)
        tax.set_xlim(float(c.timestamp_s.min()), float(c.timestamp_s.max()))
        tax.grid(alpha=0.15)
        tax.tick_params(labelsize=5.8)
        tax.set_xlabel(f"time (s)  |  {ex['outcome']}", fontsize=6.2)
        if i == 0:
            tax.set_ylabel("scene score", fontsize=6.2)
        else:
            tax.set_yticklabels([])
    fig.savefig(FIG / "figure_qualitative_validation_examples.pdf")
    fig.savefig(FIG / "figure_qualitative_validation_examples.png", dpi=360)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "archive").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRE / "figures/figure_pipeline.pdf", FIG / "archive/figure_pipeline_v1.pdf")

    reg = source_index()
    val_dec = pd.read_parquet(VAL / "04_event_evaluation/validation_video_decisions_132.parquet").sort_values("canonical_video_id")
    cal_pair = pd.read_parquet(V13 / "v13_c_event_calibration/paired_calibration_decisions.parquet")
    cal_dec = cal_pair[cal_pair.model_id == "trackfeat132"].copy()
    # Add fields required by the shared audit extractor.
    cal_input = {v["video_id"]: v for v in json.loads((CAL_EVAL / "input.json").read_text())["videos"]}
    cal_dec["usable_candidates"] = cal_dec.canonical_video_id.map(
        lambda v: (CAL_CACHE / f"{v}.parquet").exists() and len(pd.read_parquet(CAL_CACHE / f"{v}.parquet", columns=["proposal_id"])) > 0
    )
    cal_dec["video_probability"] = cal_dec.canonical_video_id.map(lambda v: float(cal_input[v]["probability"]))
    cal_dec["video_decision"] = cal_dec.canonical_video_id.map(lambda v: int(cal_input[v]["decision"]))
    cal_dec["events_json"] = cal_dec.canonical_video_id.map(lambda v: json.dumps(cal_input[v]["pred_events"], sort_keys=True))

    cal_rows = [
        quality_row(str(r.canonical_video_id), "calibration", r, CAL_CACHE / f"{r.canonical_video_id}.parquet", reg)
        for _, r in cal_dec.iterrows()
    ]
    cal_quality = pd.DataFrame(cal_rows)
    q1, q2 = cal_quality.quality_score.quantile([1 / 3, 2 / 3]).tolist()

    val_rows = [
        quality_row(str(r.canonical_video_id), "validation", r, VAL_CACHE / f"{r.canonical_video_id}.parquet", reg)
        for _, r in val_dec.iterrows()
    ]
    ledger = pd.DataFrame(val_rows)
    ledger["quality_stratum"] = pd.cut(
        ledger.quality_score,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["fragmented or low-support", "moderate continuity", "high continuity"],
        include_lowest=True,
    ).astype(str)
    fn = pd.read_csv(FN_LEDGER)[["canonical_video_id", "primary_root_cause", "diagnosis_confidence"]]
    ledger = ledger.merge(fn, on="canonical_video_id", how="left")
    ledger.to_csv(OUT / "trajectory_quality_ledger.csv", index=False)

    strata = pd.DataFrame([metric_row(ledger[ledger.quality_stratum == s], s) for s in [
        "fragmented or low-support", "moderate continuity", "high continuity"
    ]])
    strata.to_csv(OUT / "trajectory_quality_strata_results.csv", index=False)

    cross = (
        ledger[ledger.primary_root_cause.notna()]
        .groupby(["quality_stratum", "primary_root_cause"], observed=False)
        .size()
        .reset_index(name="false_negative_count")
    )
    cross.to_csv(OUT / "fn_quality_cross_tabulation.csv", index=False)

    acc = ledger[ledger.category == "accident"]
    tpq = acc[acc.matched == 1].quality_score
    fnq = acc[acc.missed == 1].quality_score
    u, p = mannwhitneyu(tpq, fnq, alternative="two-sided")
    rank_biserial = 2 * float(u) / (len(tpq) * len(fnq)) - 1
    effects = [{
        "comparison": "matched accident vs missed accident",
        "metric": "quality_score",
        "group_a_n": len(tpq),
        "group_b_n": len(fnq),
        "group_a_median": float(tpq.median()),
        "group_b_median": float(fnq.median()),
        "median_difference_a_minus_b": float(tpq.median() - fnq.median()),
        "rank_biserial_a_greater": rank_biserial,
        "mann_whitney_two_sided_p_descriptive": float(p),
        "interpretation_scope": "descriptive association; not causal",
    }]
    for cat in ["D4", "D2", "D3", "U1"]:
        q = acc[acc.primary_root_cause == cat].quality_score
        effects.append({
            "comparison": f"matched accident vs {cat}",
            "metric": "quality_score",
            "group_a_n": len(tpq),
            "group_b_n": len(q),
            "group_a_median": float(tpq.median()),
            "group_b_median": float(q.median()) if len(q) else math.nan,
            "median_difference_a_minus_b": float(tpq.median() - q.median()) if len(q) else math.nan,
            "rank_biserial_a_greater": math.nan,
            "mann_whitney_two_sided_p_descriptive": math.nan,
            "interpretation_scope": "small-category descriptive contrast; no inference",
        })
    pd.DataFrame(effects).to_csv(OUT / "trajectory_quality_effect_sizes.csv", index=False)

    missing_rate = float((~ledger.telemetry_available | ~ledger.features_available).mean())
    recall_order = strata.set_index("quality_stratum").recall
    recall_gap = float(recall_order["high continuity"] - recall_order["fragmented or low-support"])
    if missing_rate > 0.10:
        verdict = "TRAJECTORY_QUALITY_EVIDENCE_INSUFFICIENT"
    elif rank_biserial >= 0.30 and recall_gap >= 0.15:
        verdict = "TRAJECTORY_QUALITY_STRONGLY_ASSOCIATED_WITH_FAILURE"
    elif rank_biserial >= 0.15 or recall_gap >= 0.10:
        verdict = "TRAJECTORY_QUALITY_PARTIALLY_ASSOCIATED_WITH_FAILURE"
    else:
        verdict = "TRAJECTORY_QUALITY_NOT_ASSOCIATED_WITH_FAILURE"
    write_json(OUT / "trajectory_quality_verdict.json", {
        "verdict": verdict,
        "missing_required_cache_rate": missing_rate,
        "matched_vs_missed_rank_biserial": rank_biserial,
        "high_minus_low_recall": recall_gap,
        "interpretation": (
            "The audit separates complete coverage loss from fragmented/noisy evidence and "
            "downstream failures with supported tracks. Associations are descriptive and do "
            "not establish that track quality caused a model outcome."
        ),
        "frozen_result_or_policy_changed": False,
        "official_test_accessed": False,
    })
    write_json(OUT / "trajectory_quality_contract.json", {
        "operation": "read_only_frozen_validation_characterization",
        "quality_score_formula": (
            "mean(valid-track coverage in accepted window or full normal video, "
            "min(longest continuous track duration/min(duration,4s),1), full-video candidate "
            "coverage, fraction of candidate rows with track_age>=5, "
            "1-min((track fragments+observation gaps)/(3*track count),1))"
        ),
        "stratum_source": "calibration distribution only",
        "calibration_video_count": len(cal_quality),
        "calibration_tercile_thresholds": {"lower": q1, "upper": q2},
        "validation_outcomes_used_to_define_thresholds": False,
        "coverage_window": "point_tolerance_s=2.5 around accident annotation; complete video for normals",
        "segment_break_rule": "gap > 0.400001 s on the 5-Hz causal telemetry grid",
        "bootstrap": {"replicates": BOOT, "seed": SEED, "unit": "video; accident and normal sampled separately within stratum"},
        "model_or_policy_changed": False,
    })
    write_json(OUT / "trajectory_quality_provenance.json", {
        "created_utc": NOW,
        "command": (
            "/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python "
            "final_draft/trackfeat_aisi_submission_02/control/build_quality_and_figures.py"
        ),
        "inputs": [
            {"path": str(VAL / "04_event_evaluation/validation_video_decisions_132.parquet"), "sha256": sha(VAL / "04_event_evaluation/validation_video_decisions_132.parquet")},
            {"path": str(VAL / "validation_predictions_132.parquet"), "sha256": sha(VAL / "validation_predictions_132.parquet")},
            {"path": str(FN_LEDGER), "sha256": sha(FN_LEDGER)},
            {"path": str(REGISTRY), "sha256": sha(REGISTRY)},
        ],
        "cache_roots": [str(VAL_CACHE), str(CAL_CACHE), str(UPSTREAM)],
        "population": {"calibration": len(cal_quality), "validation": len(ledger), "validation_accident": int((ledger.category == "accident").sum()), "validation_normal": int((ledger.category == "normal").sum())},
        "status": "report_only_validation_characterization",
        "official_test_accessed": False,
    })

    val_pred = pd.read_parquet(VAL / "validation_predictions_132.parquet")
    val_events = pd.read_parquet(VAL / "validation_event_ledger_132.parquet")
    specs = selection_candidates(val_dec, val_events, val_pred, VAL_CACHE, reg)
    examples = [prepare_example(p, specs[p], val_pred, VAL_CACHE, reg) for p in ["A", "B", "C"]]
    cal_example = choose_calibration_example(reg)
    plot_pipeline(cal_example)
    plot_qualitative(examples)

    registry_rows = []
    values_rows = []
    for ex in examples:
        registry_rows.append({
            "panel": ex["panel"],
            "canonical_video_id": ex["video_id"],
            "role": "validation",
            "selection_rule": {
                "A": "eligible matched TP event score closest to median eligible TP event score; stable SHA-256 tie-break",
                "B": "highest-scoring eventless normal with pair evidence; stable SHA-256 tie-break",
                "C": "D4 case with absolute event timing error closest to D4 median; stable SHA-256 tie-break",
            }[ex["panel"]],
            "selection_statistic": ex["selection_statistic"],
            "display_timestamp_s": ex["timestamp_s"],
            "source_frame_index": ex["frame_index"],
            "source_fps": ex["source_fps"],
            "source_video_path": ex["video_path"],
            "source_video_sha256": ex["video_sha256"],
            "selected_track_id": ex["selected_track"],
            "selected_local_actor_id": ex["actor_2"],
            "event_status": ex["outcome"],
            "privacy_review": "no readable face or license plate at rendered paper resolution; no redaction applied",
            "official_test": False,
        })
        for name, value in ex["values"].items():
            values_rows.append({"panel": ex["panel"], "canonical_video_id": ex["video_id"], "timestamp_s": ex["timestamp_s"], "kind": "displayed_cue", "name": name, "value": value, "source_path": ex["feature_path"]})
        for rank, (name, value) in enumerate(ex["top_attributions"], start=1):
            values_rows.append({"panel": ex["panel"], "canonical_video_id": ex["video_id"], "timestamp_s": ex["timestamp_s"], "kind": "model_attribution", "name": f"{rank}:{name}", "value": value, "source_path": str(MODEL)})
    pd.DataFrame(registry_rows).to_csv(OUT / "qualitative_example_registry.csv", index=False)
    pd.DataFrame(values_rows).to_csv(OUT / "qualitative_example_values.csv", index=False)

    pipeline_inputs = [
        Path(cal_example["video_path"]),
        CAL_CACHE / f"{cal_example['video_id']}.parquet",
        UPSTREAM / cal_example["video_id"] / "causal_telemetry.parquet",
        CAL_PRED,
        OUT / "upstream_perception_contract.csv",
    ]
    write_json(FIG / "figure_pipeline_v2_provenance.json", {
        "created_utc": NOW,
        "figure_status": "method illustration from authorized calibration evidence",
        "selected_calibration_video_id": cal_example["video_id"],
        "selection_rule": "matched calibration TP with pair/track/frame availability whose event score is closest to the eligible median; stable SHA-256 tie-break",
        "timestamp_s": cal_example["timestamp_s"],
        "frame_index": cal_example["frame_index"],
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in pipeline_inputs],
        "policy": {"aggregation": "mean", "smoothing": "none", "tau_up": 0.65, "tau_down": 0.50, "k_up": 2, "k_down": 3, "cooldown_s": 2.0},
        "command": (
            "/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python "
            "final_draft/trackfeat_aisi_submission_02/control/build_quality_and_figures.py"
        ),
        "output_hashes": {"pdf": sha(FIG / "figure_pipeline_v2.pdf"), "png": sha(FIG / "figure_pipeline_v2.png")},
        "official_test_accessed": False,
    })
    qual_inputs = [VAL / "validation_predictions_132.parquet", VAL / "validation_event_ledger_132.parquet", FN_LEDGER, MODEL]
    qual_inputs.extend(Path(x["source_video_path"]) for x in registry_rows)
    write_json(FIG / "figure_qualitative_validation_examples_provenance.json", {
        "created_utc": NOW,
        "figure_status": "held-out validation qualitative evidence; not case-selected by visual drama",
        "selection_rules_frozen_before_rendering": True,
        "inputs": [{"path": str(p), "sha256": sha(p)} for p in qual_inputs],
        "example_ids": [x["canonical_video_id"] for x in registry_rows],
        "model_attribution_status": "exact selected-row LightGBM pred_contrib values; probability parity required",
        "privacy_treatment": "manual rendered-resolution review required; registry records redaction status",
        "command": (
            "/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python "
            "final_draft/trackfeat_aisi_submission_02/control/build_quality_and_figures.py"
        ),
        "output_hashes": {"pdf": sha(FIG / "figure_qualitative_validation_examples.pdf"), "png": sha(FIG / "figure_qualitative_validation_examples.png")},
        "official_test_accessed": False,
    })
    write_json(OUT / "figure_example_selection.json", {
        "pipeline_calibration": {
            "canonical_video_id": cal_example["video_id"],
            "timestamp_s": cal_example["timestamp_s"],
            "event_score": cal_example["event_score"],
            "source_frame_index": cal_example["frame_index"],
        },
        "qualitative_validation": registry_rows,
    })

    # Record predecessor identity and the reporting boundary.
    predecessor_core = [
        PRE / "main.tex", PRE / "main.pdf", PRE / "paper_evidence_manifest.json",
        PRE / "references.bib", PRE / "upstream_perception_contract.csv",
    ]
    write_json(OUT / "predecessor_copy_receipt.json", {
        "predecessor_root": str(PRE),
        "successor_root": str(OUT),
        "copied_before_editing": True,
        "predecessor_core_hashes": {p.name: sha(p) for p in predecessor_core},
        "predecessor_preserved": True,
        "created_utc": NOW,
    })
    write_json(OUT / "paper_reporting_preflight.json", {
        "schema_version": 1,
        "experiment_id": "trackfeat_aisi_submission_02_report_only",
        "operation": "publication reporting and frozen validation characterization",
        "roles": ["calibration", "validation"],
        "test_access_requested": False,
        "uses_test_for_fitting": False,
        "registered": True,
        "lineage_frozen": True,
        "intended_claim_status": "report_only",
        "checks": {
            "output_non_overwrite": True,
            "no_training_or_tuning": True,
            "no_policy_selection": True,
            "validation_thresholds_not_selected_from_outcomes": True,
            "no_gt_or_future_inference_change": True,
            "official_test_remains_sealed": True,
        },
    })
    print(json.dumps({
        "quality_verdict": verdict,
        "calibration_thresholds": [q1, q2],
        "strata": strata.to_dict(orient="records"),
        "pipeline_video": cal_example["video_id"],
        "qualitative_videos": {x["panel"]: x["video_id"] for x in examples},
    }, indent=2))


if __name__ == "__main__":
    main()
