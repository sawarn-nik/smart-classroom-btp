"""Fine-tune YOLOv8 (COCO-pretrained) on the merged SCB classroom-behaviour set.

Usage: python -m scripts.activity_train --model yolov8n.pt --epochs 50
"""
import argparse

from ultralytics import YOLO

from src.common.paths import CKPT, PROCESSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.model.split('.')[0]}_scb_e{args.epochs}"
    model = YOLO(str(CKPT / args.model) if (CKPT / args.model).exists() else args.model)
    model.train(data=str(PROCESSED / "scb_merged" / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=2, seed=0, patience=15,
                optimizer="SGD", lr0=0.01, momentum=0.937, cos_lr=True,
                project=str(CKPT / "activity"), name=name, exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
