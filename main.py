import pygame
import cv2
import numpy as np
import sys
import threading
import json # Untuk parsing data config
from flask import Flask, Response

# Import helper modules
from helper.mqtt import MQTTClientHandler
import helper.config as config
import helper.utils as utils

# --- Modul Kustom ---
try:
    from vision_controller.face_auth import FaceAuthenticator
    from vision_controller.mp_face_detector import FaceMeshDetector # NEW DETECTOR
    from preprocessing.blinker import BlinkProcessor
    from preprocessing.swing import HeadPoseProcessor
    from preprocessing.image_enhancement import LowLightEnhancer
except ImportError as e:
    print(f"❌ ERROR: Tidak dapat menemukan modul 'vision_controller'. Pastikan file ada.")
    print(f"   Detail: {e}")
    sys.exit()

# --- 1. Inisialisasi Flask (Untuk Video Streaming) ---
app = Flask(__name__)

# Variabel Global untuk Frame Streaming
global_frame = None
global_frame_lock = threading.Lock()

def generate_frames():
    """Generator function untuk streaming video ke web"""
    global global_frame
    while True:
        with global_frame_lock:
            if global_frame is None:
                continue
            
            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', global_frame)
            if not ret:
                continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route untuk akses video streaming"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Route untuk dashboard monitoring utama"""
    try:
        with open('monitoring_ui/index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File monitoring_ui/index.html tidak ditemukan."

def start_flask_app():
    """Menjalankan Flask di thread terpisah"""
    # Host='0.0.0.0' agar bisa diakses dari device lain dalam jaringan
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- 2. Inisialisasi Library ---
pygame.init()
pygame.mixer.init()

# --- 3. Helper Functions ---
helper_utils = utils.Utils(cfg=config)
sound_manager = utils.SoundManager()

# --- 4. Setup Display & Assets ---
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Nusa Neurotech - BCI Interface")
clock = pygame.time.Clock()

# Fonts
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX
try:
    FONT_BUTTON = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE, bold=config.FONT_BOLD)
except:
    FONT_BUTTON = pygame.font.Font(None, 40)

# --- 5. Load CV Models & Processors ---
try:
    # GANTI: Gunakan MediaPipe Wrapper
    face_mesh_detector = FaceMeshDetector() 
    print("✅ MediaPipe Face Mesh Detector siap.")
    
    # Authenticator tetap sama (menerima rect)
    authenticator = FaceAuthenticator(
        embeddings_path=config.FACE_EMBEDDINGS_PATH,
        tolerance=config.FACE_RECOGNITION_TOLERANCE
    )
    blink_processor = BlinkProcessor(config)
    head_pose_processor = HeadPoseProcessor(config)
    low_light_enhancer = LowLightEnhancer(clip_limit=3.0) # Inisialisasi Enhancer
except Exception as e:
    print(f"❌ Model load error: {e}")
    pygame.quit()
    sys.exit()

# Main Application State ---
running = True
is_authorized = False
authorized_user = None
current_frame_count = 0
last_auth_check_time = 0

# Greeting & Feedback State
has_greeted_user = False
last_progress_step = 0 # Melacak kelipatan 10% terakhir

# Local Device State (Simulation for feedback)
lamp_is_on = False
music_is_playing = False

# State Machine
current_mode = 0
NUM_MODES = len(config.BUTTON_LABELS)

# --- 6. Setup MQTT Handler (Pub & Sub) ---
def handle_mqtt_message(topic, payload):
    """Callback ketika menerima pesan dari MQTT (misal dari Website)"""
    global lamp_is_on, music_is_playing
    
    print(f"📩 Pesan Masuk: {topic} -> {payload}")
    
    # --- HANDLE KONFIGURASI DARI WEB ---
    if topic == config.MQTT_TOPIC_CONFIG:
        try:
            data = json.loads(payload)
            # Update durasi kedipan aksi jika ada
            if "action_duration" in data:
                new_duration = int(data["action_duration"])
                config.ACTION_BLINK_DURATION_MS = new_duration
                print(f"⚙️ CONFIG UPDATE: Durasi Aksi diubah ke {new_duration}ms")
                sound_manager.play_voice("auth_success") # Feedback suara setting tersimpan
        except Exception as e:
            print(f"❌ Config Error: {e}")

    # Handle Perintah Lampu dari Website
    elif topic == config.MQTT_TOPIC_LIGHT:
        payload = str(payload).upper()
        if payload == "TOGGLE":
            lamp_is_on = not lamp_is_on
            sound_manager.play_voice("lamp_on" if lamp_is_on else "lamp_off")
        elif payload == "ON":
            lamp_is_on = True
            sound_manager.play_voice("lamp_on")
        elif payload == "OFF":
            lamp_is_on = False
            sound_manager.play_voice("lamp_off")

mqtt_client = MQTTClientHandler(config.MQTT_BROKER, config.MQTT_PORT, on_message_callback=handle_mqtt_message)
if mqtt_client.connect():
    mqtt_client.start_publisher_thread()
    # Subscribe ke topik kontrol agar bisa dikendalikan via Website
    mqtt_client.subscribe(config.MQTT_TOPIC_LIGHT)
    mqtt_client.subscribe(config.MQTT_TOPIC_CONFIG) # Dengar update config

# --- 7. Setup Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
if not cap.isOpened():
    print("❌ Webcam Error")
    pygame.quit()
    sys.exit()

# --- 8. Start Flask Thread ---
# Jalankan server video streaming di background
flask_thread = threading.Thread(target=start_flask_app, daemon=True)
flask_thread.start()
print("🌐 Video Streaming aktif di http://localhost:5000/video_feed")

# --- 10. Main Loop ---
while running:
    current_time = pygame.time.get_ticks()
    current_frame_count += 1
    
    # --- 10.1 Event Handling (Pygame) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_m:
                current_mode = (current_mode + 1) % NUM_MODES

    # --- 10.2 Frame Processing ---
    ret, frame = cap.read()
    if not ret:
        frame_surface = pygame.Surface((config.CAM_WIDTH, config.CAM_HEIGHT))
        frame_surface.fill((0, 0, 0))
    else:
        frame = cv2.resize(frame, (config.CAM_WIDTH, config.CAM_HEIGHT))
        frame = cv2.flip(frame, 1) # Cerminkan frame

        # --- LOW LIGHT ENHANCEMENT ---
        is_dark = low_light_enhancer.is_low_light(frame)
        if is_dark:
            frame = low_light_enhancer.enhance(frame)
            # Visual feedback bahwa mode malam aktif
            cv2.putText(frame, "LOW LIGHT MODE", (10, config.CAM_HEIGHT - 10), 
                        FONT_CV, 0.6, (0, 255, 255), 1)

        found_authorized_user_this_frame = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konversi sekali saja
        
        # --- 10.3 Face Detection (MediaPipe) ---
        # MediaPipe memproses setiap frame untuk mendapatkan landmark real-time
        detected_faces = face_mesh_detector.detect(frame)

        # --- 10.4 State Controller Logic (per wajah) ---
        for face_data in detected_faces:
            rect = face_data['rect']
            landmarks = face_data['landmarks']
            
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            
            # --- CEK PENCAHAYAAN WAJAH (PERBAIKAN) ---
            # Jika wajah terdeteksi tapi gelap (misal membelakangi cahaya), terangkan area wajah.
            if low_light_enhancer.is_roi_dark(frame, x, y, w, h, threshold=90):
                # Enhance hanya pada area wajah
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    enhanced_roi = low_light_enhancer.enhance(roi)
                    frame[y:y+h, x:x+w] = enhanced_roi
                    # Visual feedback
                    cv2.putText(frame, "ENHANCED", (x, y - 25), FONT_CV, 0.5, (0, 255, 255), 1)

            # --- 10.4.1 Otentikasi (Cache) ---
            if current_frame_count % config.AUTH_CHECK_SKIP_FRAMES == 0:
                name, distance = authenticator.recognize_face(frame, rect)
                if name != "Unknown":
                    found_authorized_user_this_frame = True
                    is_authorized = True
                    authorized_user = name
                    last_auth_check_time = current_time
                    
                    # Logika Greeting (Sapaan)
                    if not has_greeted_user:
                        sound_manager.play_voice("auth_success")
                        has_greeted_user = True
                    
                    # Kalibrasi Head Pose
                    if not head_pose_processor.is_calibrated:
                         # Gunakan landmarks dari MediaPipe langsung
                        head_pose_processor.calibrate(landmarks)

            # --- 10.4.2 Proses Gestur (Hanya jika Terotorisasi) ---
            if is_authorized and authorized_user:
                
                # --- A. Jalankan Prosesor Gestur Universal (Blink) ---
                # Pass landmarks langsung ke blink processor
                blink_signal, blink_data, current_shape = blink_processor.process_frame(
                    gray, rect, None, current_time, current_frame_count, landmarks=landmarks
                )
                
                # --- LOGIKA SUARA PROGRES (Charging Sound) ---
                progress = blink_data["progress_percentage"]
                if progress == 0:
                    last_progress_step = 0 # Reset jika mata terbuka
                elif progress >= last_progress_step + 10:
                    # Bunyikan beep setiap naik 10%
                    # Nada naik seiring progress (220Hz - 800Hz)
                    pitch = 220 + (progress * 6) 
                    sound_manager.play_frequency(pitch, duration=0.05)
                    last_progress_step = (progress // 10) * 10 # Snap ke kelipatan 10 terdekat
                
                # --- B. Cek Sinyal Prioritas Tinggi (SOS & Ganti Mode) ---
                # [MODIFIKASI] Fitur SOS dan Ganti Mode dinonaktifkan sementara (di-comment)
                # agar fokus pada kontrol Lampu saja.
                
                # if blink_signal == "TRIGGER_SOS":
                #     mqtt_client.publish_async(config.MQTT_TOPIC_ALERT, "CRITICAL_SOS")
                #     sound_manager.play_voice('sos_alert') # Voice for SOS
                #     # Visual feedback lokal (opsional)
                #     cv2.putText(frame, "!!! SOS !!!", (x, y - 40), FONT_CV, 1.5, (0, 0, 255), 4)

                # elif blink_signal == "TRIGGER_MODE_SWITCH":
                #     current_mode = (current_mode + 1) % NUM_MODES
                #     # Play voice for new mode
                #     mode_keys = ["mode_lampu", "mode_tv", "mode_musik"]
                #     if current_mode < len(mode_keys):
                #          sound_manager.play_voice(mode_keys[current_mode])
                
                # --- C. Cek Sinyal Aksi (FOKUS: LAMPU) ---
                if blink_signal in ["TRIGGER_ACTION", "TRIGGER_MODE_SWITCH", "TRIGGER_SOS"]: 
                    # Kita paksa logika hanya untuk MODE LAMPU (Mode 0)
                    lamp_is_on = not lamp_is_on # Toggle state locally first
                    msg = "ON" if lamp_is_on else "OFF"
                    mqtt_client.publish_async(config.MQTT_TOPIC_LIGHT, msg)
                    pass

                # --- 10.4.3 Gambar Umpan Balik Visual (Auth & Gestur) ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_AUTH, 2)
                cv2.putText(frame, f"AUTH: {authorized_user}", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_AUTH, 2)
                
                # Gambar mata (Visualisasi MediaPipe)
                if blink_data["left_eye"] is not None:
                    eye_color = (0, 255, 0) if blink_data["is_closed"] else (255, 0, 0)
                    # Konversi titik float ke int untuk menggambar
                    l_eye_pts = np.array(blink_data["left_eye"], dtype=np.int32)
                    r_eye_pts = np.array(blink_data["right_eye"], dtype=np.int32)
                    
                    cv2.polylines(frame, [l_eye_pts], True, eye_color, 1)
                    cv2.polylines(frame, [r_eye_pts], True, eye_color, 1)
                
                # Gambar Progress Bar Kedipan
                if blink_data["progress_percentage"] > 0:
                    percentage = blink_data["progress_percentage"]
                    bar_x, bar_y = x, y + h + 30
                    bar_w, bar_h = 150, 20
                    
                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                    
                    # Progress
                    progress_w = int(bar_w * percentage / 100)
                    bar_color = (0, 255, 0) if percentage >= 100 else (0, 165, 255)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), bar_color, -1)
                    
                    # Border
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                    
                    # Text
                    text = "AKSI LAMPU" # Hardcoded untuk fokus Lampu
                    cv2.putText(frame, f"{text} {percentage}%", (bar_x, bar_y - 5), FONT_CV, 0.5, (255, 255, 255), 1)

            else: # Wajah tidak terotorisasi
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_UNAUTH, 2)
                cv2.putText(frame, "UNAUTH", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_UNAUTH, 2)

        # --- 10.5 Reset Otorisasi ---
        if not found_authorized_user_this_frame and (current_time - last_auth_check_time) > 1000:
            is_authorized = False
            authorized_user = None
            has_greeted_user = False # Reset sapaan agar bisa menyapa lagi nanti
            head_pose_processor.is_calibrated = False # Reset kalibrasi

        # --- 10.6 Gambar Umpan Balik Status Global ---
        status_text = f"AUTH: {authorized_user}" if is_authorized else "WAITING"
        status_color = config.COLOR_STATUS_AUTH if is_authorized else config.COLOR_STATUS_WAITING
        cv2.putText(frame, status_text, (10, 30), FONT_CV, 0.8, status_color, 2)
        
        # Tampilkan Mode Aktif (Hardcoded LAMPU)
        if is_authorized:
            mode_text = f"KONTROL: LAMPU"
            cv2.putText(frame, mode_text, (10, 65), FONT_CV, 0.8, (255, 255, 255), 2)
        
        # --- UPDATE GLOBAL FRAME UNTUK FLASK STREAMING ---
        with global_frame_lock:
            global_frame = frame.copy()

        # --- 10.7 Konversi ke Pygame Surface ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

    # --- 10.8 Rendering (Pygame) ---
    screen.fill(config.COLOR_UI_BACKGROUND)
    screen.blit(frame_surface, (0, 0))

    # [MODIFIKASI] Render UI Baru (Hanya Status Lampu)
    # Area UI Sebelah Kanan
    ui_x = config.CAM_WIDTH
    ui_width = config.UI_WIDTH
    
    # Judul
    title_rect = pygame.Rect(ui_x, 20, ui_width, 50)
    helper_utils.draw_text_center(screen, "STATUS", FONT_BUTTON, (200, 200, 200), title_rect)
    
    # Kotak Status Lampu Besar
    status_rect = pygame.Rect(ui_x + 20, 80, ui_width - 40, 150)
    
    if lamp_is_on:
        status_color = (255, 255, 0) # Kuning
        status_text = "MENYALA"
        text_color = (0, 0, 0)
    else:
        status_color = (50, 50, 50) # Abu Gelap
        status_text = "MATI"
        text_color = (255, 255, 255)
        
    pygame.draw.rect(screen, status_color, status_rect, border_radius=10)
    pygame.draw.rect(screen, (255, 255, 255), status_rect, 2, border_radius=10) # Border
    
    helper_utils.draw_text_center(screen, status_text, FONT_BUTTON, text_color, status_rect)
    
    # Info Durasi Setting
    duration_rect = pygame.Rect(ui_x, 250, ui_width, 30)
    duration_text = f"Durasi: {config.ACTION_BLINK_DURATION_MS/1000}s"
    helper_utils.draw_text_center(screen, duration_text, pygame.font.SysFont("Arial", 20), (150, 150, 150), duration_rect)

    pygame.display.flip()
    clock.tick(config.TARGET_FPS)

# --- 11. Cleanup ---
print("🛑 Shutting down...")
cap.release()
mqtt_client.stop()
pygame.quit()
sys.exit()