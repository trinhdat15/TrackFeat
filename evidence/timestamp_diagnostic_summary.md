# Calibration-only timestamp diagnostic

The frozen event-time control produced TP=29, recall=0.6170, and 15 wrong-window alerts on calibration. Reassigning each already-finalized event to its episode midpoint (TSC) produced TP=32, recall=0.6809, and 12 wrong-window alerts; it recovered 4 D4 cases but lost 1 existing true positive. TSC was not selected because the frozen calibration rule required zero loss of existing true positives. It was not evaluated on validation and is a calibration diagnostic only.
