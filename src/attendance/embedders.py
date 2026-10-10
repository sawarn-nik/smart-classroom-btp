"""Face detection + identity embedding.

ArcFace: InsightFace buffalo_l (SCRFD-10G detector, ResNet-50 trained with
ArcFace loss on WebFace600K, 512-d). FaceNet: InceptionResNetV1 trained on
VGGFace2 with triplet loss (facenet-pytorch), used as the comparison baseline.
Both share the same SCRFD detections so only the embedding step differs.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.common.paths import INSIGHTFACE_ROOT


@dataclass
class Face:
    bbox: np.ndarray   # x1, y1, x2, y2
    kps: np.ndarray    # 5 landmarks
    score: float


def l2norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-10)


class Detector:
    def __init__(self, det_size=640):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_ROOT),
                                allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=0.5)
        self.det = self.app.models["detection"]
        self.rec = self.app.models["recognition"]

    def detect(self, img_bgr, max_num=0):
        bboxes, kpss = self.det.detect(img_bgr, max_num=max_num, metric="default")
        return [Face(b[:4], k, float(b[4])) for b, k in zip(bboxes, kpss)]


class ArcFaceEmbedder:
    name = "ArcFace (ResNet-50)"

    def __init__(self, detector: Detector):
        self.rec = detector.rec

    def embed(self, img_bgr, faces):
        from insightface.utils import face_align
        if not faces:
            return np.zeros((0, 512), np.float32)
        crops = [face_align.norm_crop(img_bgr, f.kps, image_size=112) for f in faces]
        return l2norm(self.rec.get_feat(crops).astype(np.float32))


class FaceNetEmbedder:
    name = "FaceNet (InceptionResNetV1)"

    def __init__(self, device="cpu"):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch = torch
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(self, img_bgr, faces, margin=0.15):
        if not faces:
            return np.zeros((0, 512), np.float32)
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crops = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            mx, my = (x2 - x1) * margin, (y2 - y1) * margin
            x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
            c = cv2.resize(rgb[y1:y2, x1:x2], (160, 160)).astype(np.float32)
            crops.append((c - 127.5) / 128.0)
        t = self.torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            e = self.model(t).cpu().numpy()
        return l2norm(e.astype(np.float32))


def largest_face(faces):
    if not faces:
        return []
    areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
    return [faces[int(np.argmax(areas))]]
