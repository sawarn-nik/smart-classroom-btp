#!/bin/zsh
# Sequential experiment queue (8 GB machine: avoid running heavy jobs concurrently)
cd "$(dirname $0)/.."
export PYTHONWARNINGS=ignore
while pgrep -f scripts.attendance_embed_lfw >/dev/null; do sleep 60; done
.venv/bin/python -m scripts.attendance_eval_lfw > logs/attendance_eval.log 2>&1
.venv/bin/python -m scripts.activity_train --model yolov8n.pt --epochs 50 > logs/activity_train_n.log 2>&1
.venv/bin/python -m scripts.attendance_classroom_sim --sessions 60 > logs/attendance_sim.log 2>&1
echo QUEUE_DONE
