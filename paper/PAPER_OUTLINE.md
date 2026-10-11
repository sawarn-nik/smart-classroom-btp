# Paper outline: what each section must say (write it in your own words)

All numbers come from `results/metrics/*.json`. Tables are ready in `results/tables/*.tex`
(`\input{../results/tables/xxx.tex}`) and figures are in `results/figures/`.
Don't type a number that isn't in those files.

## Abstract: rewrite
- Remove 97.4% / F1 0.91 / mAP 88.6% / <67 ms / "new state-of-the-art". None of these were measured.
- Use the real numbers instead:
  - LFW verification: ArcFace 99.85%, FaceNet 99.38%
  - Classroom-sim attendance F1: 99.7%
  - SCB behaviour: mAP50 47.8%
  - Pipeline steady state: ~390 ms/frame on an M1 CPU for 17-40 students
  - Engagement: DAiSEE result (TODO once data arrives)

## 1 Introduction
- Keep the motivation (it's yours). Make the contribution list match what was actually built:
  1. One shared face detector feeding attendance + engagement, with identity caching via tracking.
  2. An evaluation of attendance robustness (corruptions, back-row face sizes, open-set visitors).
  3. A partially-labelled multi-subset training/evaluation protocol for SCB, including the train/val leakage you found.
  4. An empirical finding: face-level engagement cues fail in wide-angle shots, which motivates body-level behaviour detection.
- Drop "theoretical analysis of hyperspherical margin geometry" unless you actually write and understand that analysis.

## 2 Literature survey: rebuild from paper/REFERENCE_AUDIT.md
- Delete the 10 fabricated references and fix the 5 miscited ones (authors, venue, claims).
- Suggested subsections and real papers:
  - Attendance: Sunaryono 2021, Venugopal 2021, FaceNet, ArcFace, SCRFD
  - Engagement: DAiSEE, Abedi & Khan 2021, Liao 2021, Selim 2022, EngageNet (Singh 2023), Das & Dev 2025
  - Behaviour: SCB-Dataset, SCB-Dataset3, Gunawan 2024 (YOLOv9), Pyari 2026 (STAG-Net)
- Read at least the abstract of every paper you cite, so you can answer questions about it in the review.

## 3 Proposed method: update to match the code
- Attendance:
  - Detector: SCRFD-10G at 640 or 1280 px.
  - Embedding: pretrained ArcFace ResNet-50 (InsightFace buffalo_l, trained on WebFace600K), no training on our side.
  - Enrolment: mean of k images.
  - Matching: cosine score, threshold at FPIR=1% (0.295), one-to-one assignment, 3-vote confirmation per track.
- Engagement:
  - Cues: MediaPipe Face Landmarker (52 blendshapes ~ AUs, head pose from the transform matrix, iris-based gaze); 59-d per frame.
  - Sampling: 3 fps, 10-s window.
  - Clip descriptor: mean/std of each cue plus 5 behaviour statistics (visible ratio, head-away ratio, eyes-closed ratio, blink rate, head motion).
  - Classifiers: rule-based, logistic regression, gradient boosting, BiGRU.
  - Classes: DAiSEE 0+1 → disengaged, 2 → partial, 3 → engaged.
- Activity:
  - Model: YOLOv8n (3.1 M params), COCO-pretrained, fine-tuned 30 epochs, SGD lr 0.01 with cosine schedule, 640 px, batch 8.
  - Classes: hand-raising, reading, writing, bowing head, turning head, discussing.
  - Phone use and sleeping are NOT detected; remove those claims.
- Pipeline: IoU tracker; a behaviour box is linked to a face by box-contains-face-centre.
- Fix the TikZ diagram and the "Summary of novelty" comparison table: the old table compares against fabricated papers.

## 4 Datasets and experimental results (the template's TO-DO section)

### Dataset table
| Dataset | Size | Task |
|---|---|---|
| LFW | 13,233 images / 5,749 ids | attendance |
| SCB (Handrise-Read-write + BowTurnHead + Discuss) | 8,116 unique images after md5 dedup: 5,179 train / 552 val / 2,385 test | activity |
| DAiSEE | 9,068 clips / 112 subjects | engagement |

### Experimental setup
- Replace the template's "ResNet-50, dropout 0.5, SGD, Ubuntu" paragraph. That isn't what we ran.
- What we ran: Apple M1 8 GB, macOS, Python 3.11, PyTorch 2.14 (MPS), ultralytics 8.4, insightface 2.1, mediapipe 1.1.

### Results, with what each shows
- `attendance_lfw.tex` + `attendance_lfw_roc.pdf`
  - ArcFace beats FaceNet mainly in open-set use: TPIR@FPIR 1% of 99.9 vs 92.4 (k=3), and 99.9 vs 82.1 (k=1).
  - Closed-set rank-1 is close for both. This open-set gap is what decides whether visitors get falsely marked.
- `attendance_robustness.tex` + `attendance_robustness.pdf`
  - FaceNet collapses under eye occlusion (TPIR 27.0%) and 14-px faces (19.5%). ArcFace holds at 99.8% and 95.9%.
  - Caveat: WebFace600K may contain LFW celebrities, so ArcFace numbers on LFW can be optimistic.
- `attendance_classroom_sim.tex` + `classroom_sim_example_0.jpg`
  - ArcFace: F1 99.7%, exact roll in 88.3% of sessions, back row (38 px) recall 97.5%.
  - FaceNet: back-row recall 58.3%, never an exact roll.
  - Limitation: synthetic pasted composites, not real classroom video.
- `activity_scb.tex` + `activity_training_curves.png`
  - Source-aware mAP50 is 47.8% (the naive figure is 44.8%).
  - Head behaviours are weak: bowing AP50 27.9%, turning 22.6%, both with ~19% recall.
  - Discussing is best at 72.3%.
- `engagement_pose_by_behaviour.pdf` + `engagement_pose_validation_scb.json`
  - In wide shots MediaPipe returns cues for only 21% of faces (9% of writers).
  - Faces are a median 17-26 px wide.
  - Even 3D-68 head pose barely separates bowing or turning from other behaviours (AUC 0.53 / 0.54).
  - So face cues suit close-range video only. That's why the activity module exists.
- `pipeline_latency.tex`
  - Worst case is 1.06 s/frame (ArcFace on CPU, 64 ms/face).
  - Steady state is 389 ms/frame, about 2.6 fps.
  - Engagement cues now dominate (~200 ms). Options: run them on a subsample of frames, or move the work to the GPU.
- `engagement_daisee.tex` + `engagement_ablation.tex` + `engagement_confusion.pdf`: PENDING DAiSEE.
  - Compare 4-class accuracy against Gupta 2016 (57.9), Liao 2021 (58.84), Abedi & Khan 2021 (63.9), Selim 2022 (67.48).
  - Re-check each value against its PDF before quoting.
- Comparison with existing methods:
  - Only compare where the dataset and protocol are the same (DAiSEE 4-class).
  - Don't put numbers from different datasets in one column as if they were comparable.

## 5 Conclusion and future work
- Summarise what was actually achieved, then the limitations:
  - synthetic classroom simulation
  - possible LFW overlap with WebFace600K
  - partial SCB labels
  - no phone/sleep classes
  - face cues fail at distance
  - CPU latency
  - consent/privacy (needs institutional approval before real deployment)
- Future work: a real recorded and consented classroom dataset, a GPU or edge device, body-pose engagement cues.
