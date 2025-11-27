import pygame
import cv2
import numpy as np
import sys
import dlib

# Import helper modules
from helper.mqtt import MQTTPublisher
import helper.config as config
import helper.utils as utils

# --- Modul Kustom ---
try:
    from vision_controller.face_auth import FaceAuthenticator
    from preprocessing.blinker import BlinkProcessor
    from preprocessing.swing import HeadPoseProcessor
except ImportError as e:
    print(f"❌ ERROR: Tidak dapat menemukan modul 'vision_controller'. Pastikan file ada.")
    print(f"   Detail: {e}")
    sys.exit()

# --- 1. Inisialisasi Library ---
pygame.init()
pygame.mixer.init()

# --- 2. Helper Functions ---
helper = utils.Utils(cfg=config)

# --- 3. Setup Display & Assets ---
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Nusa Neurotech - BCI Interface")
clock = pygame.time.Clock()

# Fonts
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX
try:
    FONT_BUTTON = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE, bold=config.FONT_BOLD)
except:
    FONT_BUTTON = pygame.font.Font(None, 40)

# --- 4. Load CV Models & Processors ---
try:
    dlib_face_detector = dlib.get_frontal_face_detector()
    dlib_landmark_predictor = dlib.shape_predictor(config.DLIB_LANDMARK_MODEL_PATH)
    
    authenticator = FaceAuthenticator(
        embeddings_path=config.FACE_EMBEDDINGS_PATH,
        tolerance=config.FACE_RECOGNITION_TOLERANCE
    )
    blink_processor = BlinkProcessor(config)
    head_pose_processor = HeadPoseProcessor(config)
except Exception as e:
    print(f"❌ Model load error: {e}")
    pygame.quit()
    sys.exit()

# --- 5. Setup MQTT Publisher ---
mqtt_publisher = MQTTPublisher(config.MQTT_BROKER, config.MQTT_PORT)
if mqtt_publisher.connect():
    mqtt_publisher.start_publisher_thread()

# --- 6. Setup Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
if not cap.isOpened():
    print("❌ Webcam Error")
    pygame.quit()
    sys.exit()

# --- 7. Main Application State ---
running = True
is_authorized = False
authorized_user = None
current_frame_count = 0
last_auth_check_time = 0

# State Machine
current_mode = 0
NUM_MODES = len(config.BUTTON_LABELS)

# Optimization cache
cached_face_rects = []

# --- 8. Main Loop ---
while running:
    current_time = pygame.time.get_ticks()
    current_frame_count += 1
    
    # --- 8.1 Event Handling (Pygame) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_m:
                current_mode = (current_mode + 1) % NUM_MODES

    # --- 8.2 Frame Processing ---
    ret, frame = cap.read()
    if not ret:
        frame_surface = pygame.Surface((config.CAM_WIDTH, config.CAM_HEIGHT))
        frame_surface.fill((0, 0, 0))
    else:
        frame = cv2.resize(frame, (config.CAM_WIDTH, config.CAM_HEIGHT))
        frame = cv2.flip(frame, 1) # Cerminkan frame

        found_authorized_user_this_frame = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konversi sekali saja
        
        # --- 8.3 Face Detection (Cache) ---
        if current_frame_count % config.FACE_DETECTION_SKIP_FRAMES == 0:
            small_gray, scale = utils.resize_frame_for_detection(gray)
            face_rects_small = dlib_face_detector(small_gray, 0)
            cached_face_rects = [utils.scale_rect(r, config.DETECTION_SCALE) for r in face_rects_small]

        # --- 8.4 State Controller Logic (per wajah) ---
        for rect in cached_face_rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            
            # --- 8.4.1 Otentikasi (Cache) ---
            if current_frame_count % config.AUTH_CHECK_SKIP_FRAMES == 0:
                name, distance = authenticator.recognize_face(frame, rect)
                if name != "Unknown":
                    found_authorized_user_this_frame = True
                    is_authorized = True
                    authorized_user = name
                    last_auth_check_time = current_time
                    
                    # Kalibrasi Head Pose jika belum
                    if not head_pose_processor.is_calibrated:
                        shape = blink_processor.cached_shape # Gunakan shape dari blinker
                        if shape is None: # Jika blinker belum jalan, ambil shape baru
                            landmarks = dlib_landmark_predictor(gray, rect)
                            shape = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                        head_pose_processor.calibrate(shape)

            # --- 8.4.2 Proses Gestur (Hanya jika Terotorisasi) ---
            if is_authorized and authorized_user:
                
                # --- A. Jalankan Prosesor Gestur Universal (Blink) ---
                blink_signal, blink_data, current_shape = blink_processor.process_frame(
                    gray, rect, dlib_landmark_predictor, current_time, current_frame_count
                )
                
                # --- B. Cek Sinyal Prioritas Tinggi (Ganti Mode) ---
                if blink_signal == "TRIGGER_MODE_SWITCH":
                    current_mode = (current_mode + 1) % NUM_MODES
                
                # --- C. Cek Sinyal Aksi (Berdasarkan Mode Saat Ini) ---
                elif blink_signal == "TRIGGER_ACTION":
                    if current_mode == 0: # MODE: LAMPU
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_LIGHT, "TOGGLE")
                    elif current_mode == 2: # MODE: MUSIK
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_MUSIC, "play_pause")
                
                # MODE TV: Gunakan Head Pose
                if current_mode == 1:
                    head_signal, _ = head_pose_processor.process_frame(current_shape, current_time)
                    if head_signal == "VOL_PLUS":
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_TV, "vol_plus")
                    elif head_signal == "VOL_MINUS":
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_TV, "vol_min")
                    elif head_signal == "CH_PLUS":
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_TV, "ch_up")
                    elif head_signal == "CH_MINUS":
                        mqtt_publisher.publish_async(config.MQTT_TOPIC_TV, "ch_down")

                # --- 8.4.3 Gambar Umpan Balik Visual (Auth & Gestur) ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_AUTH, 2)
                cv2.putText(frame, f"AUTH: {authorized_user}", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_AUTH, 2)
                
                # Gambar mata
                if blink_data["left_eye"] is not None:
                    eye_color = (0, 255, 0) if blink_data["is_closed"] else (255, 0, 0)
                    cv2.drawContours(frame, [cv2.convexHull(blink_data["left_eye"])], -1, eye_color, 1)
                    cv2.drawContours(frame, [cv2.convexHull(blink_data["right_eye"])], -1, eye_color, 1)
                
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
                    text = "MODE" if percentage > 50 else "ACTION"
                    cv2.putText(frame, f"{text} {percentage}%", (bar_x, bar_y - 5), FONT_CV, 0.5, (255, 255, 255), 1)

            else: # Wajah tidak terotorisasi
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_UNAUTH, 2)
                cv2.putText(frame, "UNAUTH", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_UNAUTH, 2)

        # --- 8.5 Reset Otorisasi ---
        if not found_authorized_user_this_frame and (current_time - last_auth_check_time) > 1000:
            is_authorized = False
            authorized_user = None
            head_pose_processor.is_calibrated = False # Reset kalibrasi

        # --- 8.6 Gambar Umpan Balik Status Global ---
        status_text = f"AUTH: {authorized_user}" if is_authorized else "WAITING"
        status_color = config.COLOR_STATUS_AUTH if is_authorized else config.COLOR_STATUS_WAITING
        cv2.putText(frame, status_text, (10, 30), FONT_CV, 0.8, status_color, 2)
        
        # Tampilkan Mode Aktif
        if is_authorized:
            mode_text = f"MODE: {config.BUTTON_LABELS[current_mode]}"
            cv2.putText(frame, mode_text, (10, 65), FONT_CV, 0.8, (255, 255, 255), 2)

        # --- 8.7 Konversi ke Pygame Surface ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

    # --- 8.8 Rendering (Pygame) ---
    screen.fill(config.COLOR_UI_BACKGROUND)
    screen.blit(frame_surface, (0, 0))

    # Render tombol-tombol UI
    button_height = config.SCREEN_HEIGHT // len(config.BUTTON_LABELS)
    for i, label in enumerate(config.BUTTON_LABELS):
        rect = pygame.Rect(config.CAM_WIDTH, i * button_height, config.UI_WIDTH, button_height)
        
        # Highlight tombol yang modenya aktif
        if i == current_mode and is_authorized:
            pygame.draw.rect(screen, config.COLOR_BUTTON_ACTIVE, rect)
            pygame.draw.rect(screen, config.COLOR_BUTTON_TEXT, rect, 4)
        else:
            pygame.draw.rect(screen, config.COLOR_BUTTON, rect)
            pygame.draw.rect(screen, config.COLOR_BUTTON_TEXT, rect, 2)
            
        helper.draw_text_center(screen, label, FONT_BUTTON, config.COLOR_BUTTON_TEXT, rect)

    pygame.display.flip()
    clock.tick(config.TARGET_FPS)

# --- 9. Cleanup ---
print("🛑 Shutting down...")
cap.release()
mqtt_publisher.stop()
pygame.quit()
sys.exit()