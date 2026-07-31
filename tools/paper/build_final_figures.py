#!/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python
"""Build final paper figures from frozen calibration/validation evidence only."""
from __future__ import annotations

import hashlib
import importlib.util
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


ROOT = Path("/ssd1/team_cam_ai/ttdat")
OUT = ROOT / "final_draft/trackfeat_aisi_submission_03"
PRE = ROOT / "final_draft/trackfeat_aisi_submission_02"
FIG = OUT / "figures"
BASE_SOURCE = OUT / "control/build_quality_and_figures.py"
COMMAND = (
    "/home/team_cam_ai/miniconda3/envs/rtdetr_env/bin/python "
    "final_draft/trackfeat_aisi_submission_03/control/build_final_figures.py"
)
NOW = datetime.now(timezone.utc).isoformat()

spec = importlib.util.spec_from_file_location("sealed_figure_base", BASE_SOURCE)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
base.OUT = OUT
base.PRE = PRE
base.FIG = FIG

COLORS = base.COLORS
MODEL = base.MODEL
CAL_CACHE = base.CAL_CACHE
CAL_PRED = base.CAL_PRED
CAL_EVAL = base.CAL_EVAL
UPSTREAM = base.UPSTREAM
VAL = base.VAL
VAL_CACHE = base.VAL_CACHE

RAW_ALIASES = {
    "F0": "interaction energy",
    "F1": "acceleration proxy",
    "F2": "speed-loss ratio",
    "F5": "nearby IoU",
    "F6": "nearby-track speed",
    "F7": "direction similarity",
    "F10": "heading-change proxy",
    "F11": "aspect change",
    "F13": "signal quality",
    "F14": "reserved constant",
    "F15": "edge proximity",
    "F16": "normalized speed",
    "F17": "far-field weight",
    "F18": "context distance",
    "F19": "contact persistence",
    "track_age": "track age",
    "spawn_burst": "spawn burst",
    "bbox_w_norm": "box width",
    "bbox_h_norm": "box height",
    "bbox_area_norm": "box area",
    "bbox_scale_jitter": "scale jitter",
    "bbox_aspect_jitter": "aspect jitter",
    "bbox_center_jitter": "center jitter",
    "far_edge_unreliable": "edge/far unreliability",
    "track_fragment_risk": "fragmentation risk",
    "valid_physics_mask": "motion-validity mask",
}
SUFFIX_ALIASES = {
    "lag1": "lag 1", "lag2": "lag 2", "lag3": "lag 3", "lag5": "lag 5",
    "lag8": "lag 8", "diff1": "one-step change", "rollmean10": "trailing mean",
    "rollmax10": "trailing maximum", "rollstd10": "trailing std.",
    "rollsum10": "trailing sum", "ewm": "causal EWMA",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def readable_alias(name: str) -> str:
    if "__" in name:
        raw, suffix = name.split("__", 1)
        return f"{RAW_ALIASES.get(raw, raw.replace('_', ' '))}: {SUFFIX_ALIASES.get(suffix, suffix.replace('_', ' '))}"
    return RAW_ALIASES.get(name, name.replace("_", " "))


def overlay_tracks(
    ax,
    frame: np.ndarray,
    telemetry: pd.DataFrame,
    timestamp: float,
    selected_track: int,
    actor_2: int,
) -> None:
    ax.imshow(frame)
    trail = telemetry[
        (telemetry.timestamp_s <= timestamp + 1e-8)
        & (telemetry.timestamp_s >= timestamp - 1.2)
    ].sort_values("timestamp_s")
    current = telemetry[np.isclose(telemetry.timestamp_s, timestamp, atol=0.11)]
    if current.empty:
        nearest = float(telemetry.timestamp_s.iloc[(telemetry.timestamp_s - timestamp).abs().argsort()[:1]].iloc[0])
        current = telemetry[np.isclose(telemetry.timestamp_s, nearest)]
    for tid, group in trail.groupby("track_id"):
        centers = np.column_stack(((group.x1 + group.x2) / 2, (group.y1 + group.y2) / 2))
        color = COLORS["vermillion"] if int(tid) == selected_track else (
            COLORS["orange"] if int(tid) == actor_2 else COLORS["sky"]
        )
        if len(centers) > 1:
            ax.plot(centers[:, 0], centers[:, 1], "-", color=color, lw=1.6, alpha=0.9)
    for _, row in current.iterrows():
        tid = int(row.track_id)
        color = COLORS["vermillion"] if tid == selected_track else (
            COLORS["orange"] if tid == actor_2 else COLORS["sky"]
        )
        ax.add_patch(
            patches.Rectangle(
                (row.x1, row.y1), row.x2 - row.x1, row.y2 - row.y1,
                fill=False, lw=1.4, ec=color
            )
        )
        ax.text(
            row.x1,
            max(2, row.y1 - 3),
            f"{base.CLASS_NAMES.get(int(row['class']), 'vehicle')} #{tid}",
            color="white",
            fontsize=7.5,
            bbox=dict(facecolor=color, alpha=0.92, edgecolor="none", pad=1.1),
        )
    if actor_2 >= 0:
        first = current[current.track_id == selected_track]
        second = current[current.track_id == actor_2]
        if not first.empty and not second.empty:
            ac = ((first.x1.iloc[0] + first.x2.iloc[0]) / 2, (first.y1.iloc[0] + first.y2.iloc[0]) / 2)
            bc = ((second.x1.iloc[0] + second.x2.iloc[0]) / 2, (second.y1.iloc[0] + second.y2.iloc[0]) / 2)
            ax.plot([ac[0], bc[0]], [ac[1], bc[1]], "--", color=COLORS["orange"], lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)


def calibration_representation_example() -> dict:
    """Select by feature completeness inside frozen event episodes; labels are ignored."""
    prediction = pd.read_parquet(CAL_PRED)
    frozen_input = json.loads((CAL_EVAL / "input.json").read_text())
    event_times = {
        str(video["video_id"]): [float(event["time_s"]) for event in video.get("pred_events", [])]
        for video in frozen_input["videos"]
        if video.get("pred_events")
    }
    unary = ["F1", "F2", "F10", "F11", "F16"]
    interaction = ["F0", "F5", "F6", "F7", "F18", "F19"]
    reliability = ["F13", "track_age", "spawn_burst"]
    required = list(dict.fromkeys([
        "proposal_id", "timestamp_s", "track_id", "actor_2_id", "track_age",
        *sorted(set(unary + interaction + reliability)),
    ]))
    candidates: list[dict] = []
    for video_id, times in sorted(event_times.items()):
        path = CAL_CACHE / f"{video_id}.parquet"
        telemetry_path = UPSTREAM / video_id / "causal_telemetry.parquet"
        if not path.exists() or not telemetry_path.exists():
            continue
        frame = pd.read_parquet(path, columns=required)
        frame["track_age"] = pd.to_numeric(frame["track_age"], errors="coerce")
        for event_time in times:
            nearest = float(frame.timestamp_s.iloc[(frame.timestamp_s - event_time).abs().argsort()[:1]].iloc[0])
            rows = frame[
                np.isclose(frame.timestamp_s, nearest)
                & (pd.to_numeric(frame.actor_2_id, errors="coerce") >= 0)
                & (frame.track_age >= 5)
            ]
            for _, row in rows.iterrows():
                counts = {}
                for family, names in [
                    ("unary", unary), ("interaction", interaction), ("reliability", reliability)
                ]:
                    values = pd.to_numeric(row[names], errors="coerce")
                    counts[family] = int((values.notna() & (values.abs() > 1e-12)).sum())
                family_count = sum(count > 0 for count in counts.values())
                if family_count < 3:
                    continue
                candidates.append(
                    {
                        "video_id": video_id,
                        "event_time": event_time,
                        "timestamp": nearest,
                        "proposal_id": str(row.proposal_id),
                        "family_count": family_count,
                        "nonzero_count": sum(counts.values()),
                        "stable": hashlib.sha256(
                            f"{video_id}|{nearest:.6f}|{row.proposal_id}".encode()
                        ).hexdigest(),
                    }
                )
    if not candidates:
        raise RuntimeError("no representation-complete calibration candidate")
    selected = pd.DataFrame(candidates).sort_values(
        ["family_count", "nonzero_count", "stable"],
        ascending=[False, False, True],
    ).iloc[0]
    video_id = str(selected.video_id)
    feature = pd.read_parquet(CAL_CACHE / f"{video_id}.parquet")
    row = feature[feature.proposal_id.astype(str).eq(str(selected.proposal_id))].iloc[0]
    pred_video = prediction[prediction.canonical_video_id.eq(video_id)]
    pred_row = pred_video[pred_video.proposal_id.astype(str).eq(str(selected.proposal_id))].iloc[0]
    curve = base.scene_curve(pred_video)
    telemetry = pd.read_parquet(UPSTREAM / video_id / "causal_telemetry.parquet")
    registry = base.source_index()
    video_path = Path(registry.loc[video_id, "resolved_path"])
    image, frame_index, source_fps = base.read_frame(str(video_path), float(selected.timestamp))

    values = {}
    for family, names in [
        ("Unary", unary), ("Interaction", interaction), ("Reliability", reliability)
    ]:
        valid = [
            (name, float(row[name]))
            for name in names
            if pd.notna(row[name]) and abs(float(row[name])) > 1e-12
        ]
        valid.sort(key=lambda item: (-abs(item[1]), item[0]))
        values[family] = valid[:2]

    exit_time = float(curve.timestamp_s.max())
    after = curve[curve.timestamp_s >= float(selected.event_time)]
    below_run = 0
    for _, point in after.iterrows():
        below_run = below_run + 1 if float(point.probability) < 0.50 else 0
        if below_run >= 3:
            exit_time = float(point.timestamp_s)
            break
    return {
        "video_id": video_id,
        "timestamp_s": float(selected.timestamp),
        "event_time_s": float(selected.event_time),
        "event_exit_s": exit_time,
        "event_score": float(
            curve.iloc[(curve.timestamp_s - float(selected.event_time)).abs().argsort()[:1]].probability.iloc[0]
        ),
        "row": row,
        "pred_row": pred_row,
        "curve": curve,
        "telemetry": telemetry,
        "frame": image,
        "frame_index": frame_index,
        "source_fps": source_fps,
        "video_path": str(video_path),
        "video_sha256": str(registry.loc[video_id, "sha256"]),
        "selected_track": int(row.track_id),
        "actor_2": int(row.actor_2_id),
        "display_values": values,
        "candidate_pool_count": len(candidates),
        "family_count": int(selected.family_count),
        "nonzero_count": int(selected.nonzero_count),
    }


def plot_pipeline(example: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.35))
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.045, top=0.975, wspace=0.075, hspace=0.08)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.12, 0.88], width_ratios=[1, 1, 1])
    headers = [
        ("1  Fixed upstream perception", COLORS["blue"]),
        ("2  Proposed structured evidence", COLORS["green"]),
        ("3  Scoring and event formation", COLORS["orange"]),
    ]
    for column, (title, color) in enumerate(headers):
        ax = fig.add_subplot(grid[0, column])
        ax.set_axis_off()
        ax.add_patch(patches.FancyBboxPatch(
            (0.015, 0.10), 0.97, 0.80, boxstyle="round,pad=0.015",
            fc=color, ec="none", clip_on=False
        ))
        ax.text(0.5, 0.50, title, ha="center", va="center", color="white", weight="bold", fontsize=8.0)

    left = grid[1, 0].subgridspec(2, 1, height_ratios=[0.75, 0.25], hspace=0.02)
    ax = fig.add_subplot(left[0])
    overlay_tracks(ax, example["frame"], example["telemetry"], example["timestamp_s"], example["selected_track"], example["actor_2"])
    detail = fig.add_subplot(left[1])
    detail.set_axis_off()
    detail.text(
        0.02, 0.95,
        "Task-adapted YOLOv8s-derived detector\n"
        "BoxMOT BoT-SORT + verified ReID\n"
        "Persistent IDs and 1.2-s causal trails",
        va="top", fontsize=7.5,
        bbox=dict(fc="#F8FAFC", ec=COLORS["blue"], lw=0.8, boxstyle="round,pad=0.25"),
    )

    middle = grid[1, 1].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0.02)
    ax = fig.add_subplot(middle[0])
    overlay_tracks(ax, example["frame"], example["telemetry"], example["timestamp_s"], example["selected_track"], example["actor_2"])
    evidence = fig.add_subplot(middle[1])
    evidence.set_axis_off()
    text_lines = []
    for family in ["Unary", "Interaction", "Reliability"]:
        values = example["display_values"][family]
        for index, (name, value) in enumerate(values):
            prefix = f"{family}: " if index == 0 else " " * (len(family) + 2)
            text_lines.append(f"{prefix}{readable_alias(name)}={value:.3g}")
    evidence.text(
        0.02, 0.98, "\n".join(text_lines), va="top", fontsize=7.5, linespacing=1.05,
        bbox=dict(fc="#F3FAF7", ec=COLORS["green"], lw=0.8, boxstyle="round,pad=0.18"),
    )

    right = grid[1, 2].subgridspec(3, 1, height_ratios=[0.10, 0.49, 0.41], hspace=0.08)
    legend_ax = fig.add_subplot(right[0])
    legend_ax.set_axis_off()
    legend_ax.plot([], [], color=COLORS["blue"], lw=1.7, label=r"scene $s_t$")
    legend_ax.plot([], [], color=COLORS["vermillion"], lw=1.1, label=r"upper $\tau_{\rm up}$")
    legend_ax.plot([], [], color=COLORS["orange"], lw=1.1, ls="--", label=r"lower $\tau_{\rm down}$")
    legend_ax.add_patch(patches.Rectangle((0, 0), 0, 0, fc=COLORS["green"], alpha=0.13, label="event interval"))
    legend_ax.legend(loc="center", ncol=2, fontsize=7.5, frameon=False, columnspacing=0.8, handlelength=1.5)

    timeline = fig.add_subplot(right[1])
    curve = example["curve"]
    timeline.axvspan(example["event_time_s"], example["event_exit_s"], color=COLORS["green"], alpha=0.13)
    timeline.plot(curve.timestamp_s, curve.probability, color=COLORS["blue"], lw=1.7)
    timeline.axhline(0.65, color=COLORS["vermillion"], lw=1.1)
    timeline.axhline(0.50, color=COLORS["orange"], lw=1.1, ls="--")
    timeline.axvline(example["event_time_s"], color="#111827", lw=1.0)
    timeline.scatter([example["event_time_s"]], [example["event_score"]], color=COLORS["vermillion"], s=20, zorder=4)
    timeline.set_xlim(float(curve.timestamp_s.min()), float(curve.timestamp_s.max()))
    timeline.set_ylim(0, 1.02)
    timeline.set_xlabel("time (s)", fontsize=7.5)
    timeline.set_ylabel("score", fontsize=7.5, labelpad=1)
    timeline.tick_params(labelsize=7.5)
    timeline.grid(alpha=0.16)

    policy = fig.add_subplot(right[2])
    policy.set_axis_off()
    policy.text(
        0.02, 0.98,
        rf"Row $p_{{i,t}}={float(example['pred_row'].probability):.3f}$; "
        rf"scene $s_t=\mathrm{{mean}}_i\,p_{{i,t}}$" "\n"
        r"Enter: 2 samples $\geq0.65$" "\n"
        r"Exit: 3 samples $<0.50$" "\n"
        rf"Cooldown: 2 s; event $t={example['event_time_s']:.1f}$ s" "\n"
        rf"ACCIDENT; score={example['event_score']:.3f}; track #{example['selected_track']}",
        va="top", fontsize=7.5, linespacing=1.10,
        bbox=dict(fc="#FFF7ED", ec=COLORS["orange"], lw=0.8, boxstyle="round,pad=0.20"),
    )
    fig.savefig(FIG / "figure_pipeline_v3.pdf")
    fig.savefig(FIG / "figure_pipeline_v3.png", dpi=360)
    plt.close(fig)


def exact_feature_attribution(feature_row: pd.Series, frozen_probability: float) -> list[dict]:
    contract = pd.read_csv(OUT / "full_132_feature_contract.csv").sort_values("position")
    names = contract.feature.tolist()
    model = joblib.load(MODEL)
    matrix = pd.DataFrame(
        [{name: pd.to_numeric(pd.Series([feature_row[name]]), errors="raise").iloc[0] for name in names}],
        columns=names,
    )
    contributions = model.booster_.predict(matrix, pred_contrib=True)[0][:-1]
    probability = float(model.predict_proba(matrix)[0, 1])
    if abs(probability - frozen_probability) > 1e-7:
        raise AssertionError(f"probability parity failed: {probability} vs {frozen_probability}")
    rows = [
        {
            "feature": name,
            "alias": readable_alias(name),
            "feature_value": float(matrix.iloc[0][name]),
            "shap_contribution": float(contribution),
            "direction": "increases score" if contribution > 0 else "decreases score",
        }
        for name, contribution in zip(names, contributions)
    ]
    return sorted(rows, key=lambda row: (-abs(row["shap_contribution"]), row["feature"]))[:3]


def qualitative_examples() -> list[dict]:
    selection = json.loads((OUT / "figure_example_selection.json").read_text())["qualitative_validation"]
    registry = base.source_index()
    predictions = pd.read_parquet(VAL / "validation_predictions_132.parquet")
    examples = []
    for row in selection:
        spec = {
            "video_id": row["canonical_video_id"],
            "timestamp_s": row["display_timestamp_s"],
            "event_score": row["selection_statistic"],
            "selection_statistic": row["selection_statistic"],
        }
        example = base.prepare_example(
            row["panel"], spec, predictions, VAL_CACHE, registry
        )
        example["exact_top_features"] = exact_feature_attribution(
            example["feature_row"], float(example["prediction_row"].probability)
        )
        example["accepted_window_start_s"] = (
            example["annotation_time_s"] - 2.5
            if np.isfinite(example["annotation_time_s"]) else math.nan
        )
        example["accepted_window_end_s"] = (
            example["annotation_time_s"] + 2.5
            if np.isfinite(example["annotation_time_s"]) else math.nan
        )
        if example["panel"] == "C":
            example["mismatch_direction"] = (
                "early" if example["timestamp_s"] < example["accepted_window_start_s"] else "late"
            )
        examples.append(example)
    return examples


def plot_qualitative(examples: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.0, 3.55))
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.10, top=0.94, wspace=0.08)
    grid = fig.add_gridspec(1, 3)
    titles = {
        "A": "A  Matched validation event",
        "B": "B  Difficult non-alert normal",
        "C": "C  Temporal-mismatch failure",
    }
    for column, example in enumerate(examples):
        sub = grid[column].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0.09)
        image = fig.add_subplot(sub[0])
        overlay_tracks(
            image, example["frame"], example["telemetry"], example["timestamp_s"],
            example["selected_track"], example["actor_2"]
        )
        image.set_title(titles[example["panel"]], fontsize=8.2, weight="bold", pad=2)
        lines = []
        for item in example["exact_top_features"]:
            direction = "\u2191" if item["shap_contribution"] > 0 else "\u2193"
            lines.append(f"{item['alias']}={item['feature_value']:.3g} (SHAP {direction})")
        lines.append(f"track age={float(example['feature_row'].track_age):.0f} samples")
        image.text(
            0.02, 0.02, "\n".join(lines), transform=image.transAxes,
            va="bottom", fontsize=7.5,
            bbox=dict(fc="white", ec=COLORS["purple"], alpha=0.92, pad=2),
        )

        timeline = fig.add_subplot(sub[1])
        curve = example["curve"]
        timeline.plot(curve.timestamp_s, curve.probability, color=COLORS["blue"], lw=1.5, label="scene")
        timeline.axhline(0.65, color=COLORS["vermillion"], lw=1.0, label="upper")
        timeline.axhline(0.50, color=COLORS["orange"], lw=1.0, ls="--", label="lower")
        timeline.axvline(example["timestamp_s"], color="#111827", lw=1.0)
        if np.isfinite(example["annotation_time_s"]):
            timeline.axvspan(
                max(0, example["accepted_window_start_s"]),
                example["accepted_window_end_s"],
                color=COLORS["green"], alpha=0.12,
            )
            timeline.axvline(example["annotation_time_s"], color=COLORS["green"], lw=1.0, ls=":")
        timeline.set_xlim(float(curve.timestamp_s.min()), float(curve.timestamp_s.max()))
        timeline.set_ylim(0, 1.02)
        timeline.grid(alpha=0.15)
        timeline.tick_params(labelsize=7.5)
        timeline.set_xlabel(
            f"time (s); row p={float(example['prediction_row'].probability):.3f}, "
            f"scene={float(example['values']['scene_score']):.3f}",
            fontsize=7.5,
        )
        if column == 0:
            timeline.set_ylabel("scene score", fontsize=7.5)
        else:
            timeline.set_yticklabels([])
        if example["panel"] == "C":
            timeline.text(
                0.02, 0.96,
                f"event {example['timestamp_s']:.1f}s ({example['mismatch_direction']}); "
                f"impact {example['annotation_time_s']:.1f}s",
                transform=timeline.transAxes, va="top", fontsize=7.5,
                bbox=dict(fc="white", ec="#D1D5DB", alpha=0.9, pad=1.5),
            )
    fig.savefig(FIG / "figure_qualitative_validation_examples_v2.pdf")
    fig.savefig(FIG / "figure_qualitative_validation_examples_v2.png", dpi=360)
    plt.close(fig)


def main() -> None:
    (FIG / "archive").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PRE / "figures/figure_pipeline_v2.pdf",
        FIG / "archive/figure_pipeline_v2.pdf",
    )
    shutil.copy2(
        PRE / "figures/figure_qualitative_validation_examples.pdf",
        FIG / "archive/figure_qualitative_validation_examples_v1.pdf",
    )
    shutil.copy2(
        PRE / "qualitative_example_values.csv",
        OUT / "qualitative_example_values_v1.csv",
    )

    pipeline = calibration_representation_example()
    examples = qualitative_examples()
    plot_pipeline(pipeline)
    plot_qualitative(examples)

    value_rows = []
    for example in examples:
        for rank, item in enumerate(example["exact_top_features"], start=1):
            value_rows.append(
                {
                    "panel": example["panel"],
                    "canonical_video_id": example["video_id"],
                    "timestamp_s": example["timestamp_s"],
                    "rank": rank,
                    **item,
                    "row_probability": float(example["prediction_row"].probability),
                    "scene_score": float(example["values"]["scene_score"]),
                    "track_age_samples": float(example["feature_row"].track_age),
                    "source_feature_path": example["feature_path"],
                    "model_path": str(MODEL),
                    "model_sha256": sha(MODEL),
                    "probability_parity": True,
                }
            )
    pd.DataFrame(value_rows).to_csv(OUT / "qualitative_example_values.csv", index=False)

    pipeline_inputs = [
        Path(pipeline["video_path"]),
        CAL_CACHE / f"{pipeline['video_id']}.parquet",
        UPSTREAM / pipeline["video_id"] / "causal_telemetry.parquet",
        CAL_PRED,
        CAL_EVAL / "input.json",
        OUT / "full_132_feature_contract.csv",
    ]
    write_json(
        FIG / "figure_pipeline_v3_provenance.json",
        {
            "created_utc": NOW,
            "figure_status": "method illustration from authorized calibration evidence",
            "selection_rule": (
                "among rows within existing frozen calibration event episodes: valid pair, "
                "track_age>=5, finite nonzero evidence in unary, interaction, and reliability "
                "families; maximize family count then nonzero count; stable SHA-256 tie-break; "
                "annotation labels and model correctness ignored"
            ),
            "selected_calibration_video_id": pipeline["video_id"],
            "timestamp_s": pipeline["timestamp_s"],
            "frame_index": pipeline["frame_index"],
            "candidate_pool_count": pipeline["candidate_pool_count"],
            "family_count": pipeline["family_count"],
            "nonzero_count": pipeline["nonzero_count"],
            "display_values": pipeline["display_values"],
            "policy": {
                "aggregation": "mean", "smoothing": "none",
                "tau_up": 0.65, "tau_down": 0.50,
                "k_up": 2, "k_down": 3, "cooldown_s": 2.0,
            },
            "inputs": [{"path": str(path), "sha256": sha(path)} for path in pipeline_inputs],
            "command": COMMAND,
            "output_hashes": {
                "pdf": sha(FIG / "figure_pipeline_v3.pdf"),
                "png": sha(FIG / "figure_pipeline_v3.png"),
            },
            "official_test_accessed": False,
        },
    )
    qualitative_inputs = [
        VAL / "validation_predictions_132.parquet",
        VAL / "validation_event_ledger_132.parquet",
        MODEL,
        OUT / "figure_example_selection.json",
    ] + [Path(example["video_path"]) for example in examples]
    write_json(
        FIG / "figure_qualitative_validation_examples_v2_provenance.json",
        {
            "created_utc": NOW,
            "figure_status": "report-only held-out validation qualitative evidence",
            "same_predecessor_examples": True,
            "example_ids": [example["video_id"] for example in examples],
            "feature_display": (
                "top three exact model inputs by absolute local LightGBM SHAP magnitude; "
                "actual values plus signed direction; probability parity required"
            ),
            "model_attribution_not_causal": True,
            "privacy_treatment": (
                "paper-resolution visual review; no readable face or plate; no redaction applied"
            ),
            "inputs": [{"path": str(path), "sha256": sha(path)} for path in qualitative_inputs],
            "command": COMMAND,
            "output_hashes": {
                "pdf": sha(FIG / "figure_qualitative_validation_examples_v2.pdf"),
                "png": sha(FIG / "figure_qualitative_validation_examples_v2.png"),
            },
            "official_test_accessed": False,
        },
    )
    write_json(
        OUT / "figure_final_selection.json",
        {
            "pipeline_calibration": {
                "canonical_video_id": pipeline["video_id"],
                "timestamp_s": pipeline["timestamp_s"],
                "source_frame_index": pipeline["frame_index"],
                "selection_uses_annotation_or_correctness": False,
            },
            "qualitative_validation": [
                {
                    "panel": example["panel"],
                    "canonical_video_id": example["video_id"],
                    "timestamp_s": example["timestamp_s"],
                    "role": "validation",
                    "official_test": False,
                }
                for example in examples
            ],
        },
    )
    print(
        json.dumps(
            {
                "pipeline_video": pipeline["video_id"],
                "pipeline_timestamp_s": pipeline["timestamp_s"],
                "pipeline_display_values": pipeline["display_values"],
                "qualitative_ids": [example["video_id"] for example in examples],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
