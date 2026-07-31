# TrackFeat

**Interpretable Traffic Accident Detection Using Vehicle-Track Interactions**

TrackFeat is a research prototype for auditable traffic-accident detection from fixed-camera surveillance video. It combines past-only vehicle-track histories, local interaction context, reliability and missingness signals, and trailing temporal summaries in an ordered 132-feature contract. A LightGBM model scores active tracks; a separately calibrated hysteretic policy aggregates those scores into video-level alerts.

## Pipeline

```text
fixed-camera video
  -> YOLOv8s-derived vehicle detector
  -> BoT-SORT + frozen ResNet50-IBN-a ReID
  -> sanitized persistent tracks
  -> past-only TrackFeat evidence operator
  -> 132-feature LightGBM track scores
  -> scene aggregation + hysteretic event policy
  -> video-level accident alert
```

The method runs on a past-only 5 Hz grid. Its image-space evidence covers geometry, motion, direction, shape change, interaction/contact, temporal history, and explicit track reliability. Centered windows, future frames, annotations, and full-track or end-of-video statistics are excluded from inference features.

## Primary result: SO-TAD

The latest manuscript reports one frozen evaluation on a sealed, video-disjoint 332-video SO-TAD validation role containing 46 accident videos. The official test set was not opened.

| Representation | TP | Normal FP | FN | Wrong-window | Precision | Recall | F1 | Normal FPH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 132 features | 25 | 8 | 21 | 16 | 0.5102 | 0.5435 | 0.5263 | 5.0310 |
| Reduced 52 features | 25 | 10 | 21 | 18 | 0.4717 | 0.5435 | 0.5051 | 6.2887 |

The full and reduced systems use the same upstream perception, training population, classifier family, evaluator, and role assignments. The reduced contract was selected from training-OOF evidence only. Its development recall advantage did not transfer to held-out validation. Paired reduced-minus-full intervals are reported in the paper; precision and F1 intervals cross zero, so the observed full-model advantage is not claimed to be statistically decisive.

### Calibration transfer

The event policy was selected only on calibration from a finite registry of prespecified policies under a 5 false-events-per-camera-hour budget. The selected full-132 policy achieved calibration recall 0.6170 at 2.4414 FPH and transferred unchanged to validation, where it achieved recall 0.5435 at 5.0310 FPH. The validation point is an estimate of the frozen policy, not a validation tuning curve.

## Secondary structured analyses

The latest manuscript also reports bounded ACCIDENT structured-task analyses. These are grouped-OOF development results on the official IID-training role, not official-test results and not deployment claims.

| Task | Primary chain result | Interpretation |
| --- | ---: | --- |
| Impact time, T@1 | 0.3128 | Clip-midpoint control selected; exact time is not supported |
| Image-space location, S | 0.4127 | Strongest secondary capability |
| Collision category, C | 0.3432 | Hard-selected pair loses context retained by full-scene pooling |
| ACCIDENT aggregate, ACC S | 0.3515 | Chained primary result |

The paper separates primary alerting from impact-time, image-location, and collision-category capability analyses. Oracle-time and other phase-2 diagnostics are report-only and do not replace the primary chain.

## Experimental boundaries

The frozen roles are:

| Role | Accident | Normal | Total |
| --- | ---: | ---: | ---: |
| Training | 189 | 1,323 | 1,512 |
| Calibration | 47 | 295 | 342 |
| Validation | 46 | 286 | 332 |

Classifier selection uses grouped training folds. Event-policy selection uses calibration only. Validation is opened once after the model and policy are frozen. Metrics use one-to-one temporal matching and paired video-level bootstrap uncertainty.

## Limitations

- TrackFeat is evaluated conditional on its fixed detector, tracker, and ReID stack; it is not a detector-family comparison.
- Features are image-space quantities, not metric speed, physical acceleration, or calibrated time-to-collision.
- Training labels are accident-window labels broadcast to eligible rows, not actor-responsibility labels.
- SHAP values describe fitted-model attribution, not causal feature effects, accident causation, or legal responsibility.
- Held-out validation performance is not a field deployment guarantee; camera geometry, detector coverage, track continuity, and operating conditions remain deployment-specific.
- The implementation and upstream weights are not yet packaged here as a turnkey reproduction bundle.

## Reproducibility and manuscript status

The latest local manuscript snapshot is `final_draft/trackfeat_aisi_submission_05/main.pdf`. Its audit records report an eight-page compile, embedded fonts, no missing figures, no unresolved citations, no overfull boxes, no official-test access, and no changed scientific configuration during the final communication revision.

The paper is an anonymous review submission. Author, venue, and citation details should be updated when publication information is available.

## Citation

```text
TrackFeat: Interpretable Traffic Accident Detection Using Vehicle-Track Interactions.
Anonymous submission, 2026.
```
