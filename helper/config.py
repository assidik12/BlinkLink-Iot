"""
Konfigurasi Global untuk Sistem Nusa Neurotech BCI.
File ini berisi semua konstanta dan pengaturan
yang digunakan di seluruh aplikasi.
"""

# --- 1. Model & Asset Paths ---
# Path ke file-file model yang dibutuhkan
DLIB_LANDMARK_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"
FACE_EMBEDDINGS_PATH = "./face_embeddings_tf.pkl"
FACE_MODEL_PATH = "./face_embedding_model.h5" # (Akan dihapus jika kita hanya pakai keras-facenet)

# --- 2. MQTT Configuration ---
# Pengaturan untuk koneksi ke MQTT Broker
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
# Topik MQTT untuk setiap mode state
MQTT_TOPIC_LIGHT = "iot/commands/eyeblink"
MQTT_TOPIC_TV = "nusaneuro/control/tv"
MQTT_TOPIC_MUSIC = "nusaneuro/control/music"
MQTT_TOPIC_ALERT = "nusaneuro/alert" # Topik Bahaya
MQTT_TOPIC_CONFIG = "nusaneuro/config/update" # Topik untuk update setting dari web

# --- 3. Screen, Camera & UI Settings ---
# Pengaturan Tampilan Pygame
CAM_WIDTH = 800
UI_WIDTH = 200
SCREEN_WIDTH = CAM_WIDTH + UI_WIDTH
SCREEN_HEIGHT = 600
CAM_HEIGHT = SCREEN_HEIGHT
TARGET_FPS = 30 # Target frame rate untuk aplikasi

# Label untuk tombol-tombol di UI
BUTTON_LABELS = ["LAMPU", "TELEVISI", "AC"] # Sesuaikan dengan MODE_LABELS

# Pengaturan Font
FONT_NAME = 'Arial'
FONT_SIZE = 32
FONT_BOLD = True

# Palet Warna
COLOR_STATUS_AUTH = (0, 255, 0)          # Hijau
COLOR_STATUS_UNAUTH = (0, 0, 255)        # Merah
COLOR_STATUS_WAITING = (0, 165, 255)     # Oranye
COLOR_UI_BACKGROUND = (30, 30, 30)       # Abu-abu Gelap
COLOR_BUTTON = (50, 50, 50)              # Abu-abu Tombol
COLOR_BUTTON_TEXT = (255, 255, 255)      # Putih
COLOR_BUTTON_ACTIVE = (64, 224, 208)     # Turquoise (untuk mode aktif)
# Daftar warna untuk teks mode (sesuai urutan MODE_LABELS)
COLOR_MODES = [
    (0, 255, 0),   # Mode 0 (Lampu) -> Hijau
    (0, 255, 255), # Mode 1 (TV) -> Cyan
    (255, 165, 0)  # Mode 2 (AC) -> Oranye
]

# Pengaturan UI Scanner (jika masih dipakai)
SCAN_INTERVAL_MS = 2500  # Interval scanner button (ms)

# --- 4. Performance Optimization ---
# Pengaturan untuk menyeimbangkan kecepatan vs akurasi
FACE_DETECTION_SKIP_FRAMES = 3   # Deteksi wajah setiap 3 frame
AUTH_CHECK_SKIP_FRAMES = 5       # Cek otentikasi setiap 5 frame
LANDMARK_SKIP_FRAMES = 2         # Prediksi landmark setiap 2 frame (untuk kedipan)
DETECTION_SCALE = 1.0            # Skala gambar untuk deteksi (0.5 = 50% lebih kecil)
CAMERA_BUFFER_SIZE = 1           # Mengurangi latensi kamera

# --- 5. Gesture Control Settings ---
# Pengaturan durasi untuk gestur kedipan
ACTION_BLINK_DURATION_MS = 2000     # 2 detik tahan kedip untuk "Aksi"
MODE_SWITCH_BLINK_DURATION_MS = 4000  # 4 detik tahan kedip untuk "Ganti Mode"
SOS_BLINK_DURATION_MS = 6000          # 6 detik tahan kedip untuk "SOS BAHAYA"
BLINK_COOLDOWN_MS = 1500              # 1.5 detik jeda setelah satu aksi
EAR_THRESHOLD_CLOSE = 0.12            # Batas Bawah (Harus lebih kecil dari ini untuk "TUTUP")
EAR_THRESHOLD_OPEN = 0.25             # Batas Atas (Harus lebih besar dari ini untuk "BUKA")

#0.18
#0.25

# --- PENGATURAN SENSITIVITAS GERAK KEPALA (FIXED) ---
HEAD_GESTURE_COOLDOWN_MS = 2000

# PERHATIKAN: Logika Geometris Wajah 2D
# Saat Mendongak (UP), jarak mata ke hidung MEMENDEK -> Nilai NEGATIF
# Saat Menunduk (DOWN), jarak mata ke hidung MEMANJANG -> Nilai POSITIF

PITCH_LOOK_UP_THRESHOLD = -0.10   # Batas Negatif (Lebih kecil dari ini = UP)
PITCH_LOOK_DOWN_THRESHOLD = 0.10   # Batas Positif (Lebih besar dari ini = DOWN)

# Menengok Kiri/Kanan
YAW_LEFT_THRESHOLD = -0.15        # Menengok Kiri (Negatif)
YAW_RIGHT_THRESHOLD = 0.15        # Menengok Kanan (Positif)

# --- 6. Face Recognition Settings ---
# Pengaturan untuk model FaceNet
FACE_RECOGNITION_TOLERANCE = 1.0  # Jarak maksimum. (0.9 lebih ketat, 1.1 lebih longgar)
MIN_FACE_WIDTH_FOR_BLINK = 60    # Wajah < 100px terlalu kecil untuk deteksi mata