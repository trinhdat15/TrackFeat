#!/usr/bin/env python3
"""V12.58e corrected authoritative deterministic metrics for registered traffic-event predictions.

EVAL-001 correction: normal_false_events_per_hour uses only alerts on normal videos
    in the numerator, divided by total normal-video duration. Wrong-window and duplicate
    accident-video alerts are not included in the normal FPH numerator.

EVAL-002 correction: point-event annotations use tolerance-based one-to-one matching
    and a confidence-ranked point_event_ap_at_2_5s metric. Interval-IoU AP is not
    emitted for point-annotation datasets.

Predecessor evaluator SHA-256: 94faadc6bd2c0e97ea6d2b410a95ae33610e67dfa2fbad7c3d0bf82c8929082c
Version: V12.58e-AAAI
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

BENCHMARKS = {
    "accident_only_localization", "accident_vs_normal_alarm",
    "synthetic_actor_pair", "historical_contaminated_development",
}
RESULT_STATUSES = {
    "scientific_held_out", "validation_development",
    "historical_contaminated_development", "synthetic",
    "audit_only_oracle", "report_only",
}

# Default point-event matching tolerance for V12.59c protocol
DEFAULT_POINT_TOLERANCE_S = 2.5

# Schema version for this corrected evaluator
SCHEMA_VERSION = 2


def safe_div(num: float, den: float):
    """Return None when denominator is zero; never divide by epsilon."""
    return None if den == 0 else num / den


def prf(rows: list[dict]) -> dict:
    tp = sum(int(x["label"]) == 1 and int(x["decision"]) == 1 for x in rows)
    fp = sum(int(x["label"]) == 0 and int(x["decision"]) == 1 for x in rows)
    fn = sum(int(x["label"]) == 1 and int(x["decision"]) == 0 for x in rows)
    tn = sum(int(x["label"]) == 0 and int(x["decision"]) == 0 for x in rows)
    precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f1 = (None if precision is None or recall is None or precision + recall == 0
          else 2 * precision * recall / (precision + recall))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def step_average_precision(labels: list[int], scores: list[float]) -> float | None:
    """Step-function Average Precision consistent with project historical pr_auc convention.

    Denominator is total positives (NOT total predictions). Missed positives
    are included in the denominator via the recall axis.
    """
    positives = sum(labels)
    if positives == 0:
        return None
    # Sort by descending score, then ascending index for determinism
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    tp = 0
    area = 0.0
    previous_recall = 0.0
    for rank, index in enumerate(order, 1):
        tp += labels[index]
        recall = tp / positives
        precision = tp / rank
        if labels[index]:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


# ---------------------------------------------------------------------------
# Annotation-type detection
# ---------------------------------------------------------------------------

def is_point_event(event: dict) -> bool:
    """Return True if this event is a point annotation (has time_s, lacks both start_s+end_s)."""
    has_time = "time_s" in event
    has_start = "start_s" in event
    has_end = "end_s" in event
    if has_start and has_end:
        return False
    if has_time and not has_start and not has_end:
        return True
    if has_time and (has_start or has_end):
        # Ambiguous: time_s plus one interval bound
        raise ValueError(f"ambiguous_annotation_type: event has time_s and partial interval: {event}")
    if not has_time and (has_start or has_end) and not (has_start and has_end):
        # One interval bound only — ambiguous
        raise ValueError(f"ambiguous_annotation_type: event has only one of start_s/end_s: {event}")
    # Neither time_s nor interval bounds: ambiguous
    raise ValueError(f"ambiguous_annotation_type: event has no valid temporal fields: {event}")


def classify_video_annotation_type(video: dict) -> str:
    """Return 'point', 'interval', or raise for mixed/ambiguous."""
    gt_events = list(video.get("gt_events", []))
    if not gt_events:
        return "point"  # No GT events; any annotation type is acceptable
    types = set()
    for event in gt_events:
        if is_point_event(event):
            types.add("point")
        else:
            types.add("interval")
    if len(types) > 1:
        raise ValueError(
            f"mixed_annotation_types_not_supported: video {video.get('video_id')} "
            f"has both point and interval GT events"
        )
    return types.pop()


def classify_dataset_annotation_type(videos: list[dict]) -> str:
    """Return 'point', 'interval', or raise for mixed-type datasets."""
    video_types = set()
    for video in videos:
        if video.get("gt_events"):
            video_types.add(classify_video_annotation_type(video))
    if len(video_types) > 1:
        # Mixed dataset — raise
        raise ValueError(
            "mixed_annotation_types_not_supported: dataset has both point and "
            "interval annotated videos; use separate evaluator runs"
        )
    if not video_types:
        return "point"  # No GT events at all
    return video_types.pop()


# ---------------------------------------------------------------------------
# Event time helpers
# ---------------------------------------------------------------------------

def event_time(event: dict) -> float:
    """Canonical event time: time_s for point; midpoint for interval."""
    if "time_s" in event:
        return float(event["time_s"])
    start = float(event.get("start_s", 0))
    end = float(event.get("end_s", 0))
    return (start + end) / 2


# ---------------------------------------------------------------------------
# Temporal IoU (interval only)
# ---------------------------------------------------------------------------

def temporal_iou(left: dict, right: dict) -> float:
    """Temporal IoU for interval annotations only. Never call for point annotations."""
    if is_point_event(left) or is_point_event(right):
        raise ValueError(
            "temporal_iou called with point annotation; use point_matches_gt() instead"
        )
    intersection = max(
        0.0,
        min(float(left["end_s"]), float(right["end_s"]))
        - max(float(left["start_s"]), float(right["start_s"]))
    )
    union = (
        max(float(left["end_s"]), float(right["end_s"]))
        - min(float(left["start_s"]), float(right["start_s"]))
    )
    return 0.0 if union <= 0 else intersection / union


# ---------------------------------------------------------------------------
# Point-event matching helpers
# ---------------------------------------------------------------------------

def point_matches_gt(prediction: dict, target: dict, tolerance_s: float) -> bool:
    """Return True if prediction is within tolerance of GT point event."""
    err = abs(event_time(prediction) - event_time(target))
    return err <= tolerance_s


def point_match_score(prediction: dict, target: dict, tolerance_s: float) -> float:
    """Score for greedy matching: 1.0 if within tolerance else 0.0."""
    return 1.0 if point_matches_gt(prediction, target, tolerance_s) else 0.0


# ---------------------------------------------------------------------------
# Per-video matching (routes by annotation type)
# ---------------------------------------------------------------------------

def match_video_point(
    video: dict, tolerance_s: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """One-to-one point-event matching by tolerance.

    Predictions sorted descending confidence, deterministic tie-breaking.
    Returns (matches, unmatched_predictions, unmatched_gt).
    """
    gt = list(video.get("gt_events", []))
    predictions = sorted(
        video.get("pred_events", []),
        key=lambda x: (-float(x.get("score", 1.0)),
                       str(x.get("video_id", video.get("video_id", ""))),
                       str(x.get("event_id", "")))
    )
    available = set(range(len(gt)))
    matches = []
    unmatched_predictions = []

    for prediction in predictions:
        # Among all available GT events, find valid matches
        valid = [
            (abs(event_time(prediction) - event_time(gt[i])), i)
            for i in available
            if point_match_score(prediction, gt[i], tolerance_s) > 0
        ]
        if valid:
            # Select nearest; tie-break by GT event_id
            valid.sort(key=lambda x: (x[0], gt[x[1]].get("event_id", "")))
            _, index = valid[0]
            available.remove(index)
            target = gt[index]
            error_s = event_time(prediction) - event_time(target)
            matches.append({
                "video_id": video["video_id"],
                "pred_event_id": prediction.get("event_id", ""),
                "gt_event_id": target.get("event_id", ""),
                "match_score": 1.0,
                "match_type": "point_tolerance",
                "error_s": error_s,
                "error_frames": error_s * float(video.get("fps", 1.0)),
                "prediction": prediction,
                "target": target,
            })
        else:
            unmatched_predictions.append(prediction)

    unmatched_gt = [gt[index] for index in sorted(available)]
    return matches, unmatched_predictions, unmatched_gt


def match_video_interval(
    video: dict, threshold: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """One-to-one interval-event matching by temporal IoU threshold.

    Predictions sorted descending confidence, deterministic tie-breaking.
    Returns (matches, unmatched_predictions, unmatched_gt).
    """
    gt = list(video.get("gt_events", []))
    predictions = sorted(
        video.get("pred_events", []),
        key=lambda x: (-float(x.get("score", 1.0)),
                       str(x.get("video_id", video.get("video_id", ""))),
                       str(x.get("event_id", "")))
    )
    available = set(range(len(gt)))
    matches = []
    unmatched_predictions = []

    for prediction in predictions:
        scored = [
            (temporal_iou(prediction, gt[index]), index)
            for index in available
        ]
        ranked = sorted(scored, reverse=True)
        if ranked and ranked[0][0] >= threshold:
            iou, index = ranked[0]
            available.remove(index)
            target = gt[index]
            error_s = event_time(prediction) - event_time(target)
            matches.append({
                "video_id": video["video_id"],
                "pred_event_id": prediction.get("event_id", ""),
                "gt_event_id": target.get("event_id", ""),
                "match_score": iou,
                "match_type": "interval_iou",
                "error_s": error_s,
                "error_frames": error_s * float(video.get("fps", 1.0)),
                "prediction": prediction,
                "target": target,
            })
        else:
            unmatched_predictions.append(prediction)

    unmatched_gt = [gt[index] for index in sorted(available)]
    return matches, unmatched_predictions, unmatched_gt


# ---------------------------------------------------------------------------
# Average Precision by annotation type
# ---------------------------------------------------------------------------

def point_event_average_precision(
    videos: list[dict], tolerance_s: float
) -> float | None:
    """Confidence-ranked, one-to-one, tolerance-based point-event AP.

    Metric name: point_event_ap_at_{tolerance_s}s
    Denominator includes all GT events (missed contribute to denominator via recall axis).
    AP convention: step-function (consistent with historical pr_auc).
    """
    total_gt = sum(len(video.get("gt_events", [])) for video in videos)
    if total_gt == 0:
        return None

    # Build all predictions with video context; sort globally by confidence (deterministic)
    all_preds = []
    for video in videos:
        for event in video.get("pred_events", []):
            all_preds.append((
                float(event.get("score", 1.0)),
                str(video["video_id"]),
                str(event.get("event_id", "")),
                video["video_id"],
                event,
            ))
    all_preds.sort(key=lambda x: (-x[0], x[1], x[2]))  # deterministic

    gt_by_video = {video["video_id"]: list(video.get("gt_events", [])) for video in videos}
    used = {video_id: set() for video_id in gt_by_video}
    labels = []
    scores = []

    for score, _, _, video_id, prediction in all_preds:
        gt_list = gt_by_video[video_id]
        valid = [
            (abs(event_time(prediction) - event_time(gt_list[i])), i)
            for i in range(len(gt_list))
            if i not in used[video_id]
            and point_match_score(prediction, gt_list[i], tolerance_s) > 0
        ]
        if valid:
            valid.sort(key=lambda x: (x[0], gt_list[x[1]].get("event_id", "")))
            _, index = valid[0]
            used[video_id].add(index)
            labels.append(1)
        else:
            labels.append(0)
        scores.append(score)

    if not all_preds:
        return 0.0

    # Step-function AP with total_gt as denominator
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    tp = 0
    area = 0.0
    previous_recall = 0.0
    for rank, index in enumerate(order, 1):
        tp += labels[index]
        recall = tp / total_gt
        precision = tp / rank
        if labels[index]:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


def interval_event_average_precision(
    videos: list[dict], threshold: float
) -> float | None:
    """Temporal-IoU AP for interval annotations only.

    Metric names: interval_event_ap_at_iou_{threshold}
    """
    total_gt = sum(len(video.get("gt_events", [])) for video in videos)
    if total_gt == 0:
        return None

    all_preds = []
    for video in videos:
        for event in video.get("pred_events", []):
            all_preds.append((
                float(event.get("score", 1.0)),
                str(video["video_id"]),
                str(event.get("event_id", "")),
                video["video_id"],
                event,
            ))
    all_preds.sort(key=lambda x: (-x[0], x[1], x[2]))

    gt_by_video = {video["video_id"]: list(video.get("gt_events", [])) for video in videos}
    used = {video_id: set() for video_id in gt_by_video}
    labels = []
    scores = []

    for score, _, _, video_id, prediction in all_preds:
        gt_list = gt_by_video[video_id]
        ranked = sorted(
            ((temporal_iou(prediction, gt_list[i]), i)
             for i in range(len(gt_list))
             if i not in used[video_id]),
            reverse=True,
        )
        if ranked and ranked[0][0] >= threshold:
            used[video_id].add(ranked[0][1])
            labels.append(1)
        else:
            labels.append(0)
        scores.append(score)

    if not all_preds:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    tp = 0
    area = 0.0
    previous_recall = 0.0
    for rank, index in enumerate(order, 1):
        tp += labels[index]
        recall = tp / total_gt
        precision = tp / rank
        if labels[index]:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def ece(labels: list[int], probabilities: list[float], bins: int) -> float:
    total = len(labels)
    value = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        members = [
            i for i, p in enumerate(probabilities)
            if lo <= p < hi or (index == bins - 1 and p == 1.0)
        ]
        if members:
            confidence = sum(probabilities[i] for i in members) / len(members)
            accuracy = sum(labels[i] for i in members) / len(members)
            value += len(members) / total * abs(accuracy - confidence)
    return value


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def percentile(values: list[float], quantile: float) -> float | None:
    clean = sorted(x for x in values if x is not None and math.isfinite(x))
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return clean[low]
    return clean[low] * (high - position) + clean[high] * (position - low)


def bootstrap(videos: list[dict], seed: int, replicates: int) -> dict:
    rng = random.Random(seed)
    collected = {"precision": [], "recall": [], "f1": [], "pr_auc": [], "brier_score": []}
    for _ in range(replicates):
        sample = [videos[rng.randrange(len(videos))] for _ in videos]
        metrics = prf(sample)
        metrics["pr_auc"] = step_average_precision(
            [int(x["label"]) for x in sample],
            [float(x["probability"]) for x in sample],
        )
        metrics["brier_score"] = (
            sum((float(x["probability"]) - int(x["label"])) ** 2 for x in sample)
            / len(sample)
        )
        for key in collected:
            if metrics[key] is not None:
                collected[key].append(metrics[key])
    return {
        key: {
            "low": percentile(values, 0.025),
            "high": percentile(values, 0.975),
            "valid_replicates": len(values),
            "total_replicates": replicates,
        }
        for key, values in collected.items()
    }


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(payload: dict) -> tuple[dict, list[dict]]:
    benchmark = payload.get("benchmark_type")
    if benchmark not in BENCHMARKS:
        raise ValueError("invalid_or_missing_benchmark_type")
    status = payload.get("result_status")
    if status not in RESULT_STATUSES:
        raise ValueError("invalid_result_status")
    if (benchmark == "historical_contaminated_development"
            and status != "historical_contaminated_development"):
        raise ValueError("historical_benchmark_must_be_labeled_contaminated")
    videos = payload.get("videos", [])
    if not videos:
        raise ValueError("empty_video_manifest")
    ids = [video["video_id"] for video in videos]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_video_id")
    if payload.get("role") == "test" and not payload.get("test_access_approved", False):
        raise ValueError("test_access_without_approval")
    for video in videos:
        if not 0 <= float(video["probability"]) <= 1:
            raise ValueError(f"invalid_probability:{video['video_id']}")
        if int(video["decision"]) not in (0, 1) or int(video["label"]) not in (0, 1):
            raise ValueError(f"invalid_binary_value:{video['video_id']}")

    # -----------------------------------------------------------------------
    # Annotation-type routing (EVAL-002 fix)
    # -----------------------------------------------------------------------
    annotation_type = classify_dataset_annotation_type(videos)

    # Get point tolerance
    point_tolerance_s = float(
        payload.get("point_tolerance_s", DEFAULT_POINT_TOLERANCE_S)
    )

    # IoU thresholds for interval AP
    iou_thresholds = [
        float(x) for x in payload.get("temporal_iou_thresholds", [0.1, 0.3, 0.5])
    ]
    match_iou_threshold = float(payload.get("match_iou_threshold", 0.1))

    # -----------------------------------------------------------------------
    # Video-level metrics
    # -----------------------------------------------------------------------
    labels = [int(x["label"]) for x in videos]
    probabilities = [float(x["probability"]) for x in videos]
    alarm_applicable = benchmark in {"accident_vs_normal_alarm", "historical_contaminated_development"}
    video_metrics = (
        prf(videos) if alarm_applicable
        else {"not_applicable_reason": f"benchmark_type={benchmark}"}
    )
    if alarm_applicable:
        video_metrics["pr_auc"] = step_average_precision(labels, probabilities)
        video_metrics["false_alarm_videos"] = sorted(
            x["video_id"] for x in videos if not int(x["label"]) and int(x["decision"])
        )
        boot = payload.get("bootstrap", {"seed": 0, "replicates": 1000})
        video_metrics["bootstrap_ci"] = bootstrap(
            videos, int(boot["seed"]), int(boot["replicates"])
        )

    # -----------------------------------------------------------------------
    # Per-video event matching and error taxonomy
    # -----------------------------------------------------------------------
    all_matches: list[dict] = []
    all_unmatched: list[dict] = []  # with error_type

    # Error taxonomy counters
    wrong_window_accident_alerts = 0
    duplicate_accident_alerts = 0
    false_alerts_on_normal_videos = 0
    missed_accident_events_list: list[tuple[str, dict]] = []

    for video in videos:
        if annotation_type == "point":
            matches, unpred, ungt = match_video_point(video, point_tolerance_s)
        else:
            matches, unpred, ungt = match_video_interval(video, match_iou_threshold)

        all_matches.extend(matches)
        missed_accident_events_list.extend(
            (video["video_id"], x) for x in ungt
        )

        for prediction in unpred:
            if int(video["label"]):
                # Accident video — classify error type
                if annotation_type == "point":
                    # For point annotations: wrong-window = any unmatched on accident video
                    # Duplicate = if a prediction was already matched to the only GT
                    already_matched_gt_ids = {m["gt_event_id"] for m in matches
                                              if m["video_id"] == video["video_id"]}
                    gt_events = video.get("gt_events", [])
                    # Check: is there any GT within tolerance this prediction could reach
                    # but was already consumed?
                    could_have_matched = any(
                        point_match_score(prediction, gt, point_tolerance_s) > 0
                        for gt in gt_events
                    )
                    if could_have_matched:
                        duplicate_accident_alerts += 1
                        error_type = "duplicate_accident_alert"
                    else:
                        wrong_window_accident_alerts += 1
                        error_type = "wrong_window_accident_alert"
                else:
                    # Interval: use IoU-based classification
                    overlaps = [
                        temporal_iou(prediction, target)
                        for target in video.get("gt_events", [])
                    ]
                    if overlaps and max(overlaps) > 0:
                        duplicate_accident_alerts += 1
                        error_type = "duplicate_accident_alert"
                    else:
                        wrong_window_accident_alerts += 1
                        error_type = "wrong_window_accident_alert"
            else:
                # Normal video — false alert
                false_alerts_on_normal_videos += 1
                error_type = "false_alert_on_normal_video"

            all_unmatched.append({
                "video_id": video["video_id"],
                "prediction": prediction,
                "error_type": error_type,
            })

    # -----------------------------------------------------------------------
    # Temporal localization metrics
    # -----------------------------------------------------------------------
    errors_s = [x["error_s"] for x in all_matches]
    total_gt = len(all_matches) + len(missed_accident_events_list)
    matched_accident_events = len(all_matches)
    missed_events_count = len(missed_accident_events_list)

    gaussian = {
        str(sigma): (
            None if not errors_s
            else sum(math.exp(-(e * e) / (2 * sigma * sigma)) for e in errors_s)
            / len(errors_s)
        )
        for sigma in (0.5, 1.0, 2.0)
    }
    gt_gaussian = {
        str(sigma): safe_div(
            sum(math.exp(-(e * e) / (2 * sigma * sigma)) for e in errors_s),
            total_gt,
        )
        for sigma in (0.5, 1.0, 2.0)
    }

    # Point-only localization fields (no IoU labeling for point events)
    if annotation_type == "point":
        # Named as point-event localization, not IoU
        temporal = {
            "annotation_type": "point",
            "point_tolerance_s": point_tolerance_s,
            "matched_accident_events": matched_accident_events,
            "missed_accident_events": missed_events_count,
            "wrong_window_accident_alerts": wrong_window_accident_alerts,
            "duplicate_accident_alerts": duplicate_accident_alerts,
            "false_alerts_on_normal_videos": false_alerts_on_normal_videos,
            "matched_only_mean_signed_alert_delay_s": safe_div(sum(errors_s), len(errors_s)),
            "matched_only_mean_absolute_alert_delay_s": safe_div(
                sum(abs(x) for x in errors_s), len(errors_s)
            ),
            "matched_only_mean_absolute_alert_delay_frames": safe_div(
                sum(abs(x["error_frames"]) for x in all_matches), len(all_matches)
            ),
            # GT-normalized: missed GT events contribute zero similarity
            "gt_normalized_mean_absolute_alert_error_s": safe_div(
                sum(abs(x) for x in errors_s), total_gt
            ),
            "gt_normalized_mean_absolute_alert_error_s_note": (
                "missed GT events contribute zero to numerator; denominator=total_gt"
            ),
            "matched_only_mean_gaussian_temporal_similarity": gaussian,
            "gt_normalized_mean_gaussian_temporal_similarity": gt_gaussian,
            # Temporal IoU fields suppressed for point annotations
            "matched_only_mean_temporal_iou": None,
            "matched_only_mean_temporal_iou_note": (
                "not_applicable_for_point_annotations"
            ),
            "gt_normalized_mean_temporal_iou": None,
            "gt_normalized_mean_temporal_iou_note": (
                "not_applicable_for_point_annotations"
            ),
        }
        # Point-event AP (EVAL-002 fix)
        temporal["point_event_ap_at_2_5s"] = point_event_average_precision(
            videos, DEFAULT_POINT_TOLERANCE_S
        )
        if point_tolerance_s != DEFAULT_POINT_TOLERANCE_S:
            key = f"point_event_ap_at_{point_tolerance_s}s".replace(".", "_")
            temporal[key] = point_event_average_precision(videos, point_tolerance_s)
        # Interval-IoU AP not produced for point datasets
        temporal["interval_event_ap"] = None
        temporal["interval_event_ap_note"] = "not_applicable_for_point_annotations"
    else:
        # Interval annotation
        match_scores = [x["match_score"] for x in all_matches]
        temporal = {
            "annotation_type": "interval",
            "match_iou_threshold": match_iou_threshold,
            "matched_accident_events": matched_accident_events,
            "missed_accident_events": missed_events_count,
            "wrong_window_accident_alerts": wrong_window_accident_alerts,
            "duplicate_accident_alerts": duplicate_accident_alerts,
            "false_alerts_on_normal_videos": false_alerts_on_normal_videos,
            "matched_only_mean_signed_alert_delay_s": safe_div(sum(errors_s), len(errors_s)),
            "matched_only_mean_absolute_alert_delay_s": safe_div(
                sum(abs(x) for x in errors_s), len(errors_s)
            ),
            "matched_only_mean_absolute_alert_delay_frames": safe_div(
                sum(abs(x["error_frames"]) for x in all_matches), len(all_matches)
            ),
            "matched_only_mean_temporal_iou": safe_div(sum(match_scores), len(match_scores)),
            "gt_normalized_mean_temporal_iou": safe_div(sum(match_scores), total_gt),
            "matched_only_mean_gaussian_temporal_similarity": gaussian,
            "gt_normalized_mean_gaussian_temporal_similarity": gt_gaussian,
            # Point-event fields not applicable for interval
            "point_event_ap_at_2_5s": None,
            "point_event_ap_at_2_5s_note": "not_applicable_for_interval_annotations",
        }
        # Interval-event AP at multiple IoU thresholds
        temporal["interval_event_ap"] = {
            f"at_iou_{str(value).replace('.', '_')}": interval_event_average_precision(
                videos, value
            )
            for value in iou_thresholds
        }

    # -----------------------------------------------------------------------
    # EVAL-001 fix: normal_false_events_per_hour
    # -----------------------------------------------------------------------
    # Numerator: only alerts on normal (label==0) videos
    # Denominator: only normal-video duration
    normal_video_duration_s = sum(
        float(x["duration_s"]) for x in videos if not int(x["label"])
    )
    normal_video_duration_hours = normal_video_duration_s / 3600.0

    if normal_video_duration_s == 0:
        normal_false_events_per_hour = None
        normal_fph_note = (
            "no_normal_video_duration_in_manifest: metric is not applicable"
        )
    else:
        normal_false_events_per_hour = (
            false_alerts_on_normal_videos / normal_video_duration_hours
        )
        normal_fph_note = (
            "numerator=false_alerts_on_normal_videos_only; "
            "denominator=total_normal_video_duration_hours"
        )

    temporal["normal_false_events_per_hour"] = normal_false_events_per_hour
    temporal["normal_false_events_per_hour_note"] = normal_fph_note

    # Deprecated compatibility alias (clearly marked)
    temporal["_deprecated_false_events_per_hour"] = (
        "DEPRECATED_RENAMED_TO_normal_false_events_per_hour"
    )

    # -----------------------------------------------------------------------
    # Latency, spatial, actor
    # -----------------------------------------------------------------------
    latencies = []
    spatial_errors = []
    actor_correct = []
    for match in all_matches:
        prediction, target = match["prediction"], match["target"]
        detected = float(prediction.get("detected_at_s", event_time(prediction)))
        latencies.append(detected - event_time(target))
        if (prediction.get("location") is not None
                and target.get("location") is not None):
            spatial_errors.append(
                math.dist(
                    [float(x) for x in prediction["location"]],
                    [float(x) for x in target["location"]],
                )
            )
        if (prediction.get("actor_ids") is not None
                and target.get("actor_ids") is not None):
            actor_correct.append(
                set(map(str, prediction["actor_ids"]))
                == set(map(str, target["actor_ids"]))
            )

    temporal["detection_latency_s"] = {
        "mean": safe_div(sum(latencies), len(latencies)),
        "eligible": len(latencies),
        "negative_count": sum(x < 0 for x in latencies),
    }

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------
    calibration = {
        "ece": ece(labels, probabilities, int(payload.get("ece_bins", 10))),
        "ece_bins": int(payload.get("ece_bins", 10)),
        "brier_score": (
            sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
        ),
    }

    # -----------------------------------------------------------------------
    # Assemble final metrics output
    # -----------------------------------------------------------------------
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": "V12.58e",
        "predecessor_evaluator_sha256": (
            "94faadc6bd2c0e97ea6d2b410a95ae33610e67dfa2fbad7c3d0bf82c8929082c"
        ),
        "experiment_id": payload["experiment_id"],
        "benchmark_type": benchmark,
        "result_status": status,
        "role": payload.get("role"),
        "threshold": payload.get("threshold"),
        "threshold_provenance": payload.get("threshold_provenance"),
        "annotation_type": annotation_type,
        "point_tolerance_s": point_tolerance_s,
        "video_count": len(videos),
        "video_metrics": video_metrics,
        "temporal_metrics": temporal,
        "accident_video_error_taxonomy": {
            "matched_accident_events": matched_accident_events,
            "missed_accident_events": missed_events_count,
            "wrong_window_accident_alerts": wrong_window_accident_alerts,
            "duplicate_accident_alerts": duplicate_accident_alerts,
            "false_alerts_on_normal_videos": false_alerts_on_normal_videos,
            "definitions": {
                "matched_accident_event": (
                    "prediction within tolerance/IoU of an unmatched GT event on an accident video"
                ),
                "missed_accident_event": (
                    "GT event on accident video that has no matched prediction"
                ),
                "wrong_window_accident_alert": (
                    "unmatched prediction on accident video not within tolerance of any GT event"
                ),
                "duplicate_accident_alert": (
                    "unmatched prediction on accident video within tolerance of a GT event "
                    "that was already matched by a higher-confidence prediction"
                ),
                "false_alert_on_normal_video": (
                    "any prediction on a video with no GT accident event (label==0)"
                ),
            },
        },
        "spatial_collision_localization": {
            "mean_distance": safe_div(sum(spatial_errors), len(spatial_errors)),
            "eligible": len(spatial_errors),
            "unavailable_reason": (
                None if spatial_errors else "annotations_or_predictions_unavailable"
            ),
        },
        "exact_actor_pair_accuracy": {
            "accuracy": safe_div(sum(actor_correct), len(actor_correct)),
            "eligible": len(actor_correct),
            "unavailable_reason": (
                None if actor_correct else "actor_labels_or_predictions_unavailable"
            ),
        },
        "calibration": calibration,
        "systems": payload.get("systems", {
            "runtime_s": None,
            "peak_memory_mb": None,
            "peak_vram_mb": None,
            "measurement_method": "not_provided",
        }),
        "claim_restrictions": (
            ["historical_contaminated_development_not_scientific_test"]
            if status == "historical_contaminated_development"
            else []
        ),
    }

    all_events = all_matches + all_unmatched
    return metrics, all_events


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(output: Path, payload: dict, metrics: dict, events: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    # video_decisions.csv
    with (output.parent / "video_decisions.csv").open("w", newline="") as handle:
        fields = ["video_id", "label", "probability", "decision", "duration_s", "fps"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["videos"])

    # event_matches.csv
    rows = []
    for event in events:
        if "pred_event_id" in event:
            # Matched event
            rows.append({
                "video_id": event.get("video_id", ""),
                "pred_event_id": event.get("pred_event_id", ""),
                "gt_event_id": event.get("gt_event_id", ""),
                "match_score": event.get("match_score", ""),
                "match_type": event.get("match_type", ""),
                "error_s": event.get("error_s", ""),
                "error_frames": event.get("error_frames", ""),
                "error_type": "matched",
            })
        else:
            rows.append({
                "video_id": event["video_id"],
                "pred_event_id": event["prediction"].get("event_id", ""),
                "gt_event_id": "",
                "match_score": "",
                "match_type": "",
                "error_s": "",
                "error_frames": "",
                "error_type": event["error_type"],
            })
    with (output.parent / "event_matches.csv").open("w", newline="") as handle:
        fields = [
            "video_id", "pred_event_id", "gt_event_id",
            "match_score", "match_type", "error_s", "error_frames", "error_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # evaluation_compliance.json
    compliance = {
        "experiment_id": payload["experiment_id"],
        "benchmark_type": payload["benchmark_type"],
        "evaluator_version": "V12.58e",
        "registered_threshold": bool(payload.get("threshold_provenance")),
        "test_access_approved": (
            payload.get("role") != "test"
            or bool(payload.get("test_access_approved"))
        ),
        "historical_or_oracle_claim_restricted": bool(metrics["claim_restrictions"]),
        "passed": (
            bool(payload.get("threshold_provenance"))
            and (
                payload.get("role") != "test"
                or bool(payload.get("test_access_approved"))
            )
        ),
    }
    (output.parent / "evaluation_compliance.json").write_text(
        json.dumps(compliance, indent=2, sort_keys=True) + "\n"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="V12.58e corrected traffic-event evaluator"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    metrics, events = evaluate(payload)
    write_outputs(args.output, payload, metrics, events)
    print(json.dumps({
        "video_count": metrics["video_count"],
        "benchmark_type": metrics["benchmark_type"],
        "annotation_type": metrics["annotation_type"],
        "evaluator_version": metrics["evaluator_version"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
