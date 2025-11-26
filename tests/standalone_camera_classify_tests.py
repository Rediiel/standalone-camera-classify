import numpy as np
import pytest

import standalone_camera_classify as scc
import standalone_camera_classify_settings as settings

# ----------------- Tests normalize_hog_block_norm -----------------


@pytest.mark.parametrize(
    "input_val,expected",
    [
        ("", "L2-Hys"),
        ("L1", "L1"),
        ("l1_sqrt", "L1-sqrt"),
        ("L2_hys", "L2-Hys"),
        ("other", "other"),
    ],
)
def test_normalize_hog_block_norm(input_val, expected):
    assert scc.normalize_hog_block_norm(input_val) == expected


# ----------------- Tests feature extraction -----------------


def test_extract_color_hist_rgb():
    settings.COLOR_MODE = "rgb"
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    hist = scc.extract_color_hist(img)
    assert hist.sum() > 0
    assert isinstance(hist, np.ndarray)


def test_extract_color_hist_gray():
    settings.COLOR_MODE = "gray"
    img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    hist = scc.extract_color_hist(img)
    assert hist.sum() > 0
    assert isinstance(hist, np.ndarray)


def test_extract_raw_rgb_and_gray():
    settings.COLOR_MODE = "rgb"
    img = np.ones((5, 5, 3), dtype=np.uint8) * 128
    raw_rgb = scc.extract_raw(img)
    assert np.all(raw_rgb <= 1.0) and np.all(raw_rgb >= 0.0)

    settings.COLOR_MODE = "gray"
    raw_gray = scc.extract_raw(img)
    assert np.all(raw_gray <= 1.0) and np.all(raw_gray >= 0.0)


def test_extract_hog_features_shape():
    if scc.SKIMAGE_OK:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        feats = scc.extract_hog_features(img)
        assert feats.ndim == 1


# ----------------- Tests adjust_feature_len -----------------


def test_adjust_feature_len_pad():
    settings.PAD_IF_MISMATCH = True
    arr = np.array([1, 2, 3], dtype=np.float32)
    out = scc.adjust_feature_len(arr, 5)
    assert out.size == 5
    assert np.all(out[:3] == arr)


def test_adjust_feature_len_truncate():
    settings.PAD_IF_MISMATCH = True
    arr = np.arange(10, dtype=np.float32)
    out = scc.adjust_feature_len(arr, 5)
    assert out.size == 5
    assert np.all(out == arr[:5])


def test_adjust_feature_len_no_change():
    arr = np.arange(4, dtype=np.float32)
    out = scc.adjust_feature_len(arr, 4)
    assert np.all(out == arr)


# ----------------- Tests scores_to_probabilities -----------------


def test_scores_to_probabilities_binary():
    s = np.array([0])
    probs = scc.scores_to_probabilities(s)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.shape[1] == 2


def test_scores_to_probabilities_multiclass():
    s = np.array([[1.0, 2.0, 3.0]])
    probs = scc.scores_to_probabilities(s)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs.shape[1] == 3


# ----------------- Tests draw_label_band -----------------


def test_draw_label_band_size():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    scc.draw_label_band(img, "test")
    assert img.shape == (50, 50, 3)


# ----------------- Optional: test infer_expected_feature_length -----------------


def test_infer_expected_feature_length_empty_pipeline():
    class DummyPipeline:
        pass

    pipeline = DummyPipeline()
    assert scc.infer_expected_feature_length(pipeline) is None
