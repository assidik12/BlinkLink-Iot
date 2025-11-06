"""
Konfigurasi Global untuk Nusa # --- Performance Optimization Settings ---
FACE_DETECTION_SKIP_FRAMES = 5   # Deteksi wajah setiap N frame (lebih jarang = lebih smooth)
AUTH_CHECK_SKIP_FRAMES = 10      # Auth check setiap N frame (lebih jarang = lebih smooth)
LANDMARK_SKIP_FRAMES = 2         # Landmark prediction setiap N frame (untuk blink detection)
DETECTION_SCALE = 0.4            # Downscale untuk face detection (lebih kecil = lebih cepat)
CAMERA_BUFFER_SIZE = 1           # Reduce camera lag

# --- Blink Detection Settings ---
BLINK_DURATION_THRESHOLD_MS = 2000  # ✅ Minimal durasi mata tertutup (2 detik)
BLINK_COOLDOWN_MS = 1000            # ✅ Cooldown setelah trigger (1 detik) BCI System
"""

# --- Model Paths ---
DLIB_LANDMARK_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"
FACE_EMBEDDINGS_PATH = "./face_embeddings_tf.pkl"
FACE_MODEL_PATH = "./face_embedding_model.h5"

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/commands/eyeblink"
MQTT_MESSAGE = "TOGGLE"

# --- Screen & Camera Settings ---
CAM_WIDTH = 800
UI_WIDTH = 200
SCREEN_WIDTH = CAM_WIDTH + UI_WIDTH
SCREEN_HEIGHT = 600
CAM_HEIGHT = SCREEN_HEIGHT

# --- Color Palette ---
COLOR_STATUS_AUTH = (0, 255, 0)          # Green
COLOR_STATUS_UNAUTH = (0, 0, 255)        # Red
COLOR_STATUS_WAITING = (0, 165, 255)     # Orange
COLOR_UI_BACKGROUND = (30, 30, 30)       # Dark Gray
COLOR_BUTTON = (64, 224, 208)            # Turquoise
COLOR_BUTTON_TEXT = (0, 0, 0)            # Black
COLOR_HIGHLIGHT = (255, 140, 0)          # Dark Orange

# --- UI Timing ---
SCAN_INTERVAL_MS = 2500  # Interval scanner button (ms)

# --- Performance Optimization Settings ---
FACE_DETECTION_SKIP_FRAMES = 3   # Deteksi wajah setiap N frame
AUTH_CHECK_SKIP_FRAMES = 5       # Auth check setiap N frame
DETECTION_SCALE = 0.5            # Downscale untuk face detection (0.5 = 50%)
CAMERA_BUFFER_SIZE = 1           # Reduce camera lag

# --- Blink Detection Settings ---
BLINK_DURATION_THRESHOLD_MS = 1500  # ✅ Minimal durasi mata tertutup (2 detik)
BLINK_COOLDOWN_MS = 1000            # ✅ Cooldown setelah trigger (1 detik)

# --- Button Labels ---
BUTTON_LABELS = ["BANTUAN", "TOILET", "Vol +", "Vol -", "Ch +"]

# --- Font Settings ---
FONT_NAME = 'Arial'
FONT_SIZE = 32
FONT_BOLD = True

# --- Face Recognition Settings ---
FACE_RECOGNITION_TOLERANCE = 1.5

# --- Frame Rate ---
TARGET_FPS = 30

LANDMARK_SKIP_FRAMES = 2         # Landmark/blink setiap 2 frame