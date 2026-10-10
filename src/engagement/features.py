"""Per-frame engagement cues from MediaPipe Face Landmarker.

We use MediaPipe instead of OpenFace 2.0 (no maintained macOS/ARM build).
- 52 ARKit-style blendshapes, which map closely onto FACS action units
  (e.g. browInnerUp ~ AU1, browDown ~ AU4, eyeBlink ~ AU45, jawOpen ~ AU26)
- head pose (yaw, pitch, roll) from the facial transformation matrix
- eye gaze as the iris centre position inside each eye, horizontal and vertical
"""
import numpy as np

from src.common.paths import CKPT

MODEL = CKPT / "mediapipe" / "face_landmarker.task"

# eye landmark indices (MediaPipe 478-point mesh, with iris refinement)
L_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468}
R_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473}

POSE_NAMES = ["yaw", "pitch", "roll"]
GAZE_NAMES = ["gaze_lx", "gaze_ly", "gaze_rx", "gaze_ry"]


def rotation_to_euler(R):
    """Rotation matrix -> (yaw, pitch, roll) in degrees. Pitch > 0 = looking up."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees([yaw, pitch, roll])


def eye_gaze(lm, eye):
    """Iris centre normalised to the eye box: x in [-1,1] (outer->inner), y in [-1,1] (top->bottom)."""
    o, i, t, b, c = (lm[eye[k]][:2] for k in ("outer", "inner", "top", "bottom", "iris"))
    axis = i - o
    width = np.linalg.norm(axis) + 1e-6
    x = np.dot(c - o, axis) / width ** 2 * 2 - 1
    vert = b - t
    height = np.linalg.norm(vert) + 1e-6
    y = np.dot(c - t, vert) / height ** 2 * 2 - 1
    return float(x), float(y)


class FaceCueExtractor:
    def __init__(self, num_faces=1):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.4)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.blendshape_names = None

    def __call__(self, img_rgb):
        """Returns a list (one per face) of dicts: blendshapes(52), pose(3), gaze(4), landmarks."""
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
        res = self.landmarker.detect(img)
        out = []
        for k, lms in enumerate(res.face_landmarks):
            lm = np.array([[p.x, p.y, p.z] for p in lms])
            bs = res.face_blendshapes[k]
            if self.blendshape_names is None:
                self.blendshape_names = [c.category_name for c in bs]
            R = np.array(res.facial_transformation_matrixes[k])[:3, :3]
            gaze = (*eye_gaze(lm, L_EYE), *eye_gaze(lm, R_EYE))
            out.append({"blendshapes": np.array([c.score for c in bs], np.float32),
                        "pose": rotation_to_euler(R).astype(np.float32),
                        "gaze": np.array(gaze, np.float32),
                        "landmarks": lm})
        return out

    def frame_vector(self, img_rgb):
        """59-d vector for the most prominent face, or None if no face."""
        faces = self(img_rgb)
        if not faces:
            return None
        f = faces[0]
        return np.concatenate([f["blendshapes"], f["pose"], f["gaze"]])

    def feature_names(self):
        return list(self.blendshape_names or []) + POSE_NAMES + GAZE_NAMES


def clip_features(frames, fps_sampled):
    """Aggregate per-frame vectors (None = face lost) into a fixed clip descriptor.

    Descriptor = mean and std of each per-frame feature, plus behavioural
    statistics: face-visible ratio, off-screen-head ratio (|yaw|>25 or pitch<-20),
    eye-closed ratio, blink rate (per minute) and head-motion energy.
    """
    valid = [f for f in frames if f is not None]
    n = len(frames)
    if not valid:
        return None
    X = np.stack(valid)
    yaw, pitch = X[:, 52], X[:, 53]
    # eyeBlinkLeft / eyeBlinkRight are blendshapes 9 and 10 in MediaPipe's ordering
    eye_closed = (X[:, 9] + X[:, 10]) / 2 > 0.5
    blinks = int(np.sum(np.diff(eye_closed.astype(int)) == 1))
    motion = float(np.mean(np.abs(np.diff(X[:, 52:55], axis=0)))) if len(X) > 1 else 0.0
    extra = np.array([
        len(valid) / n,
        float(np.mean((np.abs(yaw) > 25) | (pitch < -20))),
        float(np.mean(eye_closed)),
        blinks / (n / fps_sampled) * 60.0,
        motion,
    ], np.float32)
    return np.concatenate([X.mean(0), X.std(0), extra])


CLIP_EXTRA_NAMES = ["face_visible_ratio", "head_away_ratio", "eyes_closed_ratio", "blink_rate_pm", "head_motion"]
