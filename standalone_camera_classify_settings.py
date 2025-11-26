# Model files
MODEL_DIR = "model"
MODEL_FILENAME = "image_model.joblib"
LABELS_FILENAME = "label_encoder.joblib"

# Image preprocessing (must match training)
IMAGE_SIZE = (128, 128)  # width, height
COLOR_MODE = "rgb"  # "rgb" or "gray"

# Feature method (must match training)
FEATURE_METHOD = "hog"  # "hog", "color_hist", "raw"

# HOG params
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HOG_TRANSFORM_SQRT = True

# Color histogram params
HIST_BINS_PER_CHANNEL = 16
HIST_RANGE = (0, 256)

# Confidence threshold
MIN_CONFIDENCE = 0.20

# Band style
BAND_BASE_FONT_SCALE = 1.0
BAND_FONT_THICKNESS = 2
BAND_VERTICAL_PADDING = 10

# Camera settings
CAMERA_INDEX = 0
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Misc
PAD_IF_MISMATCH = True
VERBOSE = True
WINDOW_TITLE = "Realtime Image Classification"
