# Smart Classroom Analytics (BTP, Sem 7).

Attendance (face recognition), engagement (facial cues) and classroom
behaviour detection, with one shared face detector.

## Layout
```
src/attendance   SCRFD detection, ArcFace / FaceNet embeddings, gallery, corruptions, metrics
src/engagement   MediaPipe cues (blendshapes~AUs, head pose, gaze), clip features, models
src/activity     (training/eval in scripts/, YOLOv8 via ultralytics)
src/pipeline     IoU tracker + unified per-frame system
scripts/         every experiment (each writes results/metrics/*.json and figures)
tests/           pytest unit + smoke tests
data/raw         downloaded datasets (git-ignored)
data/processed   cached embeddings / cues / merged datasets (git-ignored)
checkpoints/     model weights + YOLO runs (git-ignored)
results/         metrics JSON, figures (PDF/PNG), LaTeX tables
paper/           LaTeX source (elsarticle template)
logs/            run logs
```

## Setup
```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests
```

## Data
| Dataset | Task | How |
|---|---|---|
| LFW (13,233 imgs, 5,749 ids) | attendance | auto-downloaded by scikit-learn |
| SCB-Dataset (HF `wintonYF/SCB-Dataset`: Handrise-Read-write, BowTurnHead, Discuss) | activity | `data/raw/scb`, then `python -m scripts.activity_prepare_scb` |
| DAiSEE | engagement | request via the official form, unzip to `data/raw/DAiSEE` |

## Experiments (in order)
```
python -m scripts.attendance_embed_lfw --conditions clean occlusion blur low_light low_res
python -m scripts.attendance_eval_lfw
python -m scripts.activity_train --model yolov8n.pt --epochs 50
python -m scripts.activity_eval --runs yolov8n_scb_e50
python -m scripts.attendance_classroom_sim --sessions 60
python -m scripts.pipeline_benchmark
python -m scripts.engagement_extract_daisee      # after DAiSEE arrives
python -m scripts.engagement_train
```

