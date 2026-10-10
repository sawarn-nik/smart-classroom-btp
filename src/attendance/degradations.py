"""Synthetic corruptions that mimic classroom capture conditions."""
import cv2
import numpy as np


def occlude_lower(img, frac=0.45):
    """Mask/hand covering the lower part of the face region (centre crop of LFW)."""
    out = img.copy()
    h, w = out.shape[:2]
    # LFW funneled images: face roughly in the central 50%
    y1 = int(h * (0.5 + 0.25 * (1 - frac)))
    out[y1:int(h * 0.78), int(w * 0.3):int(w * 0.7)] = (90, 90, 90)
    return out


def blur(img, k=9):
    return cv2.GaussianBlur(img, (k, k), 0)


def low_light(img, gamma=2.5, noise=8.0, seed=0):
    rng = np.random.default_rng(seed)
    x = (img / 255.0) ** gamma * 255.0
    x = x + rng.normal(0, noise, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def low_res(img, face_px=28):
    """Downscale so the face is ~face_px wide (back-row student), then upscale."""
    h, w = img.shape[:2]
    s = face_px / (0.4 * w)
    small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS = {
    "clean": lambda x: x,
    "occlusion": occlude_lower,
    "blur": blur,
    "low_light": low_light,
    "low_res": low_res,
}
