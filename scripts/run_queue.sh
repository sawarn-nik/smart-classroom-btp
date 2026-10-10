#!/bin/zsh
# Sequential experiment queue (8 GB machine: never run two heavy jobs at once)
cd "$(dirname $0)/.."
export PYTHONWARNINGS=ignore
set -x
.venv/bin/python -m scripts.attendance_embed_lfw --conditions clean occlusion occlusion_eyes blur blur_strong low_light low_res low_res_20 low_res_14 > logs/attendance_embed.log 2>&1
.venv/bin/python -m scripts.attendance_eval_lfw > logs/attendance_eval.log 2>&1
.venv/bin/python -m scripts.attendance_classroom_sim --sessions 60 > logs/attendance_sim.log 2>&1
.venv/bin/python -m scripts.activity_train --model yolov8n.pt --epochs 30 --batch 8 > logs/activity_train_n.log 2>&1
.venv/bin/python -m scripts.activity_eval --runs yolov8n_scb_e30 > logs/activity_eval.log 2>&1
.venv/bin/python -m scripts.pipeline_benchmark --activity-run yolov8n_scb_e30 > logs/pipeline_benchmark.log 2>&1
echo QUEUE_DONE
