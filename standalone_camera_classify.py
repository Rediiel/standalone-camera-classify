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
import cv2
import numpy as np
from joblib import load as joblib_load

# AJOUT POUR LA TÂCHE 3 : Logging
import logging

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray

    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

import standalone_camera_classify_settings as settings


# ----------------- Configuration du Logging (TÂCHE 3) -----------------
# Configurer le logger avant tout usage
logging.basicConfig(
    level=logging.INFO,  # Niveau de base pour afficher les INFO, WARNING, ERROR
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CameraClassify")
# ----------------------------------------------------------------------


# ----------------- Feature extraction helpers -----------------


def normalize_hog_block_norm(candidate: str) -> str:
    """
    Normalise la chaîne de caractère du bloc HOG pour s'assurer qu'elle correspond
    à un type de normalisation valide (L1, L1-sqrt, L2-Hys).

    Parameters
    ----------
    candidate : str
        La chaîne de caractère représentant le type de normalisation du bloc.

    Returns
    -------
    str
        Le type de normalisation standardisé (e.g., 'L2-Hys').
    """
    if not candidate:
        return "L2-Hys"
    s = str(candidate).strip().lower()
    if s in ("l1", "l1-sqrt", "l1_sqrt", "l1sqrt"):
        return "L1-sqrt" if "sqrt" in s else "L1"
    if s in ("l2-hys", "l2_hys", "l2hys", "l2"):
        return "L2-Hys"
    return candidate


def extract_color_hist(img_bgr: np.ndarray) -> np.ndarray:
    """
    Calcule l'histogramme des couleurs de l'image.

    Le calcul est effectué sur un canal (gris) ou trois canaux (BGR) selon
    les paramètres du module `settings`. L'histogramme résultant est normalisé.

    Parameters
    ----------
    img_bgr : np.ndarray
        L'image d'entrée au format BGR (OpenCV).

    Returns
    -------
    np.ndarray
        Le vecteur des caractéristiques de l'histogramme des couleurs normalisé.
    """
    if settings.COLOR_MODE.lower() == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist = (
            cv2.calcHist(
                [gray],
                [0],
                None,
                [settings.HIST_BINS_PER_CHANNEL],
                [settings.HIST_RANGE[0], settings.HIST_RANGE[1]],
            )
            .astype(np.float32)
            .ravel()
        )
        return hist / hist.sum() if hist.sum() > 0 else hist
    else:
        chans = cv2.split(img_bgr)
        feats = []
        for ch in chans:
            h = (
                cv2.calcHist(
                    [ch],
                    [0],
                    None,
                    [settings.HIST_BINS_PER_CHANNEL],
                    [settings.HIST_RANGE[0], settings.HIST_RANGE[1]],
                )
                .astype(np.float32)
                .ravel()
            )
            feats.append(h)
        feat = np.concatenate(feats)
        return feat / feat.sum() if feat.sum() > 0 else feat


def extract_raw(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extrait les caractéristiques brutes (pixels) de l'image.

    L'image est redimensionnée (implicitement dans `main`), convertie en niveaux
    de gris si nécessaire, aplatie en un vecteur et normalisée (0.0 à 1.0).

    Parameters
    ----------
    img_bgr : np.ndarray
        L'image d'entrée au format BGR (OpenCV).

    Returns
    -------
    np.ndarray
        Le vecteur de caractéristiques brut et normalisé.
    """
    if settings.COLOR_MODE.lower() == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        arr = gray.astype(np.float32) / 255.0
        return arr.ravel()
    else:
        arr = img_bgr.astype(np.float32) / 255.0
        return arr.ravel()


def extract_hog_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Calcule les caractéristiques Histogram of Oriented Gradients (HOG) de l'image.

    La méthode utilise scikit-image et dépend des paramètres HOG définis.

    Parameters
    ----------
    img_bgr : np.ndarray
        L'image d'entrée au format BGR (OpenCV).

    Returns
    -------
    np.ndarray
        Le vecteur des caractéristiques HOG.

    Raises
    ------
    RuntimeError
        Si la bibliothèque scikit-image n'est pas disponible.
    """
    if not SKIMAGE_OK:
        logger.error(
            "scikit-image est requis pour l'extraction de HOG "
            "mais n'est pas disponible."
        )
        raise RuntimeError("scikit-image is required for HOG feature extraction.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img_rgb)
    block_norm = normalize_hog_block_norm(settings.HOG_BLOCK_NORM)
    feats = hog(
        gray,
        orientations=settings.HOG_ORIENTATIONS,
        pixels_per_cell=settings.HOG_PIXELS_PER_CELL,
        cells_per_block=settings.HOG_CELLS_PER_BLOCK,
        block_norm=block_norm,
        transform_sqrt=settings.HOG_TRANSFORM_SQRT,
        feature_vector=True,
    )
    return feats.astype(np.float32)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Fonction principale d'extraction des caractéristiques basée sur la méthode
    définie dans `settings.FEATURE_METHOD` ('hog', 'color_hist', ou 'raw').

    Parameters
    ----------
    img_bgr : np.ndarray
        L'image d'entrée au format BGR (OpenCV).

    Returns
    -------
    np.ndarray
        Le vecteur de caractéristiques extrait.

    Raises
    ------
    ValueError
        Si la méthode d'extraction configurée est inconnue.
    """
    method = settings.FEATURE_METHOD.lower()
    try:
        if method == "hog":
            return extract_hog_features(img_bgr)
        elif method == "color_hist":
            return extract_color_hist(img_bgr)
        elif method == "raw":
            return extract_raw(img_bgr)
        else:
            logger.error(
                "FEATURE_METHOD inconnu dans les settings : "
                f"{settings.FEATURE_METHOD}"
            )
            raise ValueError(f"Unknown FEATURE_METHOD: {settings.FEATURE_METHOD}")
    except Exception as e:
        logger.error(
            f"Échec de l'extraction des caractéristiques pour la méthode {method}: {e}"
        )
        raise e


def infer_expected_feature_length(pipeline):
    """
    Déduit la longueur attendue du vecteur de caractéristiques à partir
    du modèle de pipeline chargé (via le scaler ou les coefficients du classifieur).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline ou classifieur
        Le modèle chargé contenant les étapes d'entraînement.

    Returns
    -------
    int ou None
        La longueur attendue du vecteur de caractéristiques,
        ou None si elle ne peut être déduite.
    """
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
    """
    Ajuste la longueur du vecteur de caractéristiques (padding ou tronquage).

    L'action dépend de la valeur de `settings.PAD_IF_MISMATCH`.

    Parameters
    ----------
    feat : np.ndarray
        Le vecteur de caractéristiques extrait.
    expected : int
        La longueur attendue du vecteur (déduite du modèle).

    Returns
    -------
    np.ndarray
        Le vecteur de caractéristiques ajusté.

    Raises
    ------
    ValueError
        Si la longueur ne correspond pas et que `PAD_IF_MISMATCH` est False.
    """
    f = np.asarray(feat, dtype=np.float32).ravel()
    if expected is None or f.size == expected:
        return f
    if f.size < expected and settings.PAD_IF_MISMATCH:
        logger.warning(
            f"Remplissage des caractéristiques: longueur attendue {expected}, "
            f"trouvée {f.size}."
        )
        out = np.zeros((expected,), dtype=np.float32)
        out[: f.size] = f
        return out
    if f.size > expected and settings.PAD_IF_MISMATCH:
        logger.warning(
            f"Tronquage des caractéristiques: longueur attendue {expected}, "
            f"trouvée {f.size}."
        )
        return f[:expected]

    # Correction de l'erreur F541 (string simple) et E501
    logger.error(
        "Longueur des caractéristiques incompatible: taille "
        f"{f.size} != attendue {expected}. (PAD_IF_MISMATCH est Faux)"
    )
    raise ValueError(f"feature length {f.size} != expected {expected}")


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """
    Convertit les scores de décision (ou log-probabilités) d'un classifieur
    en probabilités (via la fonction logistique pour les modèles binaires, ou
    softmax pour les modèles multi-classes).

    Parameters
    ----------
    scores : np.ndarray
        Le score de décision du classifieur (1D pour binaire, 2D pour multi-classe).

    Returns
    -------
    np.ndarray
        Les probabilités converties.
    """
    s = np.asarray(scores)
    if s.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-s))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)
    else:
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


# ----------------- Drawing helper -----------------


def draw_label_band(frame_bgr: np.ndarray, text: str):
    """
    Dessine une bande noire en bas de l'image avec le texte de la classe
    identifiée centré en blanc.

    Parameters
    ----------
    frame_bgr : np.ndarray
        L'image d'entrée (sera modifiée in-place).
    text : str
        Le nom de la classe à afficher.
    """
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(
        text, font, settings.BAND_BASE_FONT_SCALE, settings.BAND_FONT_THICKNESS
    )
    pad = settings.BAND_VERTICAL_PADDING
    band_h = th + baseline + 2 * pad
    y1 = h - band_h
    cv2.rectangle(frame_bgr, (0, y1), (w, h), (0, 0, 0), thickness=-1)
    x = (w - tw) // 2
    y = y1 + pad + th
    # Correction E501 ici en cassant la ligne
    cv2.putText(
        frame_bgr,
        text,
        (x, y),
        font,
        settings.BAND_BASE_FONT_SCALE,
        (255, 255, 255),
        settings.BAND_FONT_THICKNESS,
        cv2.LINE_AA,
    )


# ----------------- Main loop -----------------


def main():
    """
    Fonction principale exécutant la boucle de classification en temps réel.

    Elle initialise la caméra, charge le modèle, extrait les caractéristiques
    de chaque image capturée et affiche le résultat de la classification
    si la confiance minimale est atteinte.

    Gère la fermeture propre de la caméra et des fenêtres.
    """
    logger.info("Démarrage du script de classification en temps réel.")

    model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)
    labels_path = os.path.join(settings.MODEL_DIR, settings.LABELS_FILENAME)

    # REMPLACEMENT du print() par logger.error()
    if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
        # Correction E501: Coupure du long message
        logger.error(
            "Modèle ou encodeur de labels non trouvés aux chemins : "
            f"{model_path} / {labels_path}"
        )
        sys.exit(1)

    logger.info("Modèle et encodeur de labels trouvés. Chargement...")

    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_ANY)

    # REMPLACEMENT du print() par logger.error()
    if not cap.isOpened():
        logger.error(
            f"Impossible d'ouvrir la caméra (index {settings.CAMERA_INDEX}). "
            "Vérifiez la connexion."
        )
        sys.exit(1)

    logger.info(
        "Caméra ouverte. Résolution demandée: "
        f"{settings.CAPTURE_WIDTH}x{settings.CAPTURE_HEIGHT}."
    )

    if settings.CAPTURE_WIDTH and settings.CAPTURE_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAPTURE_HEIGHT)

    pipeline = joblib_load(model_path)
    label_enc = joblib_load(labels_path)

    logger.info("Pipeline de classification chargé avec succès.")

    try:
        classes = list(label_enc.classes_)
    except Exception:
        classes = list(label_enc)

    expected_len = infer_expected_feature_length(pipeline)

    window_name = settings.WINDOW_TITLE
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logger.info(
        "Démarrage de la boucle de traitement de la vidéo "
        "(Appuyez sur 'q' ou 'ESC' pour quitter)."
    )

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            logger.warning(
                "Échec de la lecture d'une image de la caméra. Arrêt de la boucle."
            )
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
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du modèle : {e}")
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
                logger.debug(
                    "Classification %s avec probabilité %.2f" % (class_name, prob)
                )
                draw_label_band(frame_bgr, class_name)

        # Show the frame
        try:
            cv2.imshow(window_name, frame_bgr)
        except cv2.error:
            logger.warning("Fenêtre d'affichage détruite entre les boucles.")
            break  # window was destroyed between loops

        # Pump GUI events first so window state updates
        key = cv2.waitKey(1) & 0xFF

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
        if key in (27, ord("q")):  # ESC or q
            logger.info("Signal de fermeture reçu par l'utilisateur.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Arrêt de la caméra et fin de l'exécution du programme.")


if __name__ == "__main__":
    main()
