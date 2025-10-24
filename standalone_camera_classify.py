#!/usr/bin/env python3
"""
standalone_camera_classify.py

Realtime webcam image classification using your pretrained model.

- Opens the laptop camera.
- Extracts features (HOG / color_hist / raw) the same way as training.
- Loads the trained model + label encoder from joblib files.
- If the model is confident about a class, shows a black band at the bottom
  with the class name in bold white sans-serif text.
- If not confident, no band is drawn.

Keys:
  q / ESC → quit
  f       → toggle fullscreen (best effort)
"""

import os
import sys
import time
import cv2
import numpy as np
from joblib import load as joblib_load

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

import standalone_camera_classify_settings as settings


# ----------------- Feature extraction helpers -----------------

def normalize_hog_block_norm(candidate: str) -> str:
    if not candidate:
        return "L2-Hys"
    s = str(candidate).strip().lower()
    if s in ("l1", "l1-sqrt", "l1_sqrt", "l1sqrt"):
        return "L1-sqrt" if "sqrt" in s else "L1"
    if s in ("l2-hys", "l2_hys", "l2hys", "l2"):
        return "L2-Hys"
    return candidate


def extract_color_hist(img_bgr: np.ndarray) -> np.ndarray:
    if settings.COLOR_MODE.lower() == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None,
                            [settings.HIST_BINS_PER_CHANNEL],
                            [settings.HIST_RANGE[0], settings.HIST_RANGE[1]]).astype(np.float32).ravel()
        return hist / hist.sum() if hist.sum() > 0 else hist
    else:
        chans = cv2.split(img_bgr)
        feats = []
        for ch in chans:
            h = cv2.calcHist([ch], [0], None,
                             [settings.HIST_BINS_PER_CHANNEL],
                             [settings.HIST_RANGE[0], settings.HIST_RANGE[1]]).astype(np.float32).ravel()
            feats.append(h)
        feat = np.concatenate(feats)
        return feat / feat.sum() if feat.sum() > 0 else feat


def extract_raw(img_bgr: np.ndarray) -> np.ndarray:
    if settings.COLOR_MODE.lower() == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        arr = gray.astype(np.float32) / 255.0
        return arr.ravel()
    else:
        arr = img_bgr.astype(np.float32) / 255.0
        return arr.ravel()


def extract_hog_features(img_bgr: np.ndarray) -> np.ndarray:
    if not SKIMAGE_OK:
        raise RuntimeError("scikit-image is required for HOG feature extraction.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img_rgb)
    block_norm = normalize_hog_block_norm(settings.HOG_BLOCK_NORM)
    feats = hog(gray,
                orientations=settings.HOG_ORIENTATIONS,
                pixels_per_cell=settings.HOG_PIXELS_PER_CELL,
                cells_per_block=settings.HOG_CELLS_PER_BLOCK,
                block_norm=block_norm,
                transform_sqrt=settings.HOG_TRANSFORM_SQRT,
                feature_vector=True)
    return feats.astype(np.float32)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    method = settings.FEATURE_METHOD.lower()
    if method == "hog":
        return extract_hog_features(img_bgr)
    elif method == "color_hist":
        return extract_color_hist(img_bgr)
    elif method == "raw":
        return extract_raw(img_bgr)
    else:
        raise ValueError(f"Unknown FEATURE_METHOD: {settings.FEATURE_METHOD}")


def infer_expected_feature_length(pipeline):
    try:
        if hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
            scaler = pipeline.named_steps["scaler"]
            if hasattr(scaler, "mean_"):
                return int(scaler.mean_.shape[0])
    except Exception:
        pass
    try:
        clf = pipeline
        if hasattr(pipeline, "named_steps"):
            last = list(pipeline.named_steps.keys())[-1]
            clf = pipeline.named_steps[last]
        if hasattr(clf, "coef_"):
            return int(clf.coef_.shape[1])
    except Exception:
        pass
    return None


def adjust_feature_len(feat: np.ndarray, expected: int) -> np.ndarray:
    f = np.asarray(feat, dtype=np.float32).ravel()
    if expected is None or f.size == expected:
        return f
    if f.size < expected and settings.PAD_IF_MISMATCH:
        out = np.zeros((expected,), dtype=np.float32)
        out[:f.size] = f
        return out
    if f.size > expected and settings.PAD_IF_MISMATCH:
        return f[:expected]
    raise ValueError(f"feature length {f.size} != expected {expected}")


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores)
    if s.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-s))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)
    else:
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


# ----------------- Drawing helper -----------------

def draw_label_band(frame_bgr: np.ndarray, text: str):
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font,
                                         settings.BAND_BASE_FONT_SCALE,
                                         settings.BAND_FONT_THICKNESS)
    pad = settings.BAND_VERTICAL_PADDING
    band_h = th + baseline + 2 * pad
    y1 = h - band_h
    cv2.rectangle(frame_bgr, (0, y1), (w, h), (0, 0, 0), thickness=-1)
    x = (w - tw) // 2
    y = y1 + pad + th
    cv2.putText(frame_bgr, text, (x, y), font,
                settings.BAND_BASE_FONT_SCALE,
                (255, 255, 255),
                settings.BAND_FONT_THICKNESS,
                cv2.LINE_AA)


# ----------------- Main loop -----------------

def main():
    model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)
    labels_path = os.path.join(settings.MODEL_DIR, settings.LABELS_FILENAME)
    if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
        print("[ERROR] Model or label encoder not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    if settings.CAPTURE_WIDTH and settings.CAPTURE_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAPTURE_HEIGHT)

    pipeline = joblib_load(model_path)
    label_enc = joblib_load(labels_path)
    try:
        classes = list(label_enc.classes_)
    except Exception:
        classes = list(label_enc)

    expected_len = infer_expected_feature_length(pipeline)

    window_name = settings.WINDOW_TITLE
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        resized = cv2.resize(frame_bgr, settings.IMAGE_SIZE)
        feat = extract_features(resized)
        feat = adjust_feature_len(feat, expected_len)
        X = feat.reshape(1, -1)

        probs, preds = None, None
        try:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X)
            else:
                if hasattr(pipeline, "decision_function"):
                    scores = pipeline.decision_function(X)
                    probs = scores_to_probabilities(np.atleast_2d(scores))
                else:
                    preds = pipeline.predict(X)
        except Exception:
            pass

        if probs is None and preds is not None:
            out_probs = np.zeros((1, len(classes)), dtype=float)
            idx = int(preds[0])
            out_probs[0, idx] = 1.0
            probs = out_probs

        if probs is not None:
            row = probs[0]
            idx = int(np.argmax(row))
            prob = float(row[idx])
            if prob >= settings.MIN_CONFIDENCE:
                class_name = classes[idx] if idx < len(classes) else str(idx)
                draw_label_band(frame_bgr, class_name)

        # Show the frame
        try:
            cv2.imshow(window_name, frame_bgr)
        except cv2.error:
            break  # window was destroyed between loops

        # Pump GUI events first so window state updates
        key = cv2.waitKey(1) & 0xFF

        # Robustly detect that the window was closed
        closed = False
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                closed = True
        except cv2.error:
            closed = True

        if not closed:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
                    closed = True
            except cv2.error:
                closed = True

        if not closed and hasattr(cv2, "getWindowImageRect"):
            try:
                _, _, ww, hh = cv2.getWindowImageRect(window_name)
                if ww <= 0 or hh <= 0:
                    closed = True
            except cv2.error:
                closed = True

        if closed:
            break

        # Still allow keyboard quit
        if key in (27, ord('q')):  # ESC or q
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

