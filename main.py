import pygame
import cv2
import numpy as np
import sys
import dlib

# Import helper modules
from helper.mqtt import MQTTPublisher
import helper.config as config

# --- Modul Kustom ---
try:
    from vision_controller.face_auth import FaceAuthenticator
    from vision_controller.blink_module import BlinkDetector
except ImportError:
    print("ERROR: Tidak dapat menemukan modul 'vision_controller'.")
    sys.exit()

# --- 1. Inisialisasi Library ---
pygame.init()
pygame.mixer.init()

# --- 2. Helper Functions ---
def draw_text_center(surface, text, font, color, rect):
    """Draw centered text on a rect"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)


def resize_frame_for_detection(frame, scale=config.DETECTION_SCALE):
    """⚡ Resize frame untuk deteksi lebih cepat"""
    small = cv2.resize(frame, None, fx=scale, fy=scale, 
                       interpolation=cv2.INTER_LINEAR)
    return small, scale


def scale_rect(rect, scale):
    """⚡ Scale dlib rectangle kembali ke ukuran asli"""
    return dlib.rectangle(
        int(rect.left() / scale),
        int(rect.top() / scale),
        int(rect.right() / scale),
        int(rect.bottom() / scale)
    )


# --- 3. Setup Display & Assets ---
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Nusa Neurotech - BCI Interface (Optimized)")
clock = pygame.time.Clock()

# Fonts
FONT_CV = cv2.FONT_HERSHEY_SIMPLEX
try:
    FONT_BUTTON = pygame.font.SysFont(config.FONT_NAME, config.FONT_SIZE, bold=config.FONT_BOLD)
except:
    FONT_BUTTON = pygame.font.Font(None, 40)

# --- 4. Setup Buttons ---
buttons = []
button_height = config.SCREEN_HEIGHT // len(config.BUTTON_LABELS)
for i, label in enumerate(config.BUTTON_LABELS):
    rect = pygame.Rect(config.CAM_WIDTH, i * button_height, config.UI_WIDTH, button_height)
    buttons.append({"rect": rect, "label": label})

# --- 5. Load CV Models ---
print("INFO: Memuat model CV...")
try:
    authenticator = FaceAuthenticator(
        embeddings_path=config.FACE_EMBEDDINGS_PATH,
        model_path=config.FACE_MODEL_PATH,
        tolerance=config.FACE_RECOGNITION_TOLERANCE
    )
    blinker = BlinkDetector()
    dlib_face_detector = dlib.get_frontal_face_detector()
    dlib_landmark_predictor = dlib.shape_predictor(config.DLIB_LANDMARK_MODEL_PATH)
    print("✅ Model CV loaded")
except Exception as e:
    print(f"❌ Model Error: {e}")
    pygame.quit()
    sys.exit()

# --- 6. Setup MQTT Publisher ---
mqtt_publisher = MQTTPublisher(config.MQTT_BROKER, config.MQTT_PORT)
if mqtt_publisher.connect():
    mqtt_publisher.start_publisher_thread()

# --- 7. Setup Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
if not cap.isOpened():
    print("❌ Webcam Error")
    pygame.quit()
    sys.exit()

# --- 8. Main Variables ---
running = True
is_authorized = False
authorized_user = None
current_scan_index = 0
last_scan_time = pygame.time.get_ticks()

# ⚡ Optimization cache
cached_face_rects = []
last_detection_time = 0
last_auth_check_time = 0
current_frame_count = 0

# ⚡ Frame caching
cached_is_closed = False
cached_ear = 0.3
cached_left_eye = None
cached_right_eye = None

# ✅ Blink Duration Tracking
eyes_closed_start_time = None
last_mqtt_trigger_time = 0
mqtt_triggered = False

# Wajah dengan lebar < 100 pixel terlalu kecil dan tidak akurat
# Anda mungkin perlu menyesuaikan nilai ini
MIN_FACE_WIDTH_FOR_BLINK = 100 


print("\n🚀 Aplikasi BCI (OPTIMIZED MODE) Started")

# --- 9. Main Loop ---
while running:
    current_time = pygame.time.get_ticks()
    current_frame_count += 1

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # UI Scanner Logic
    if current_time - last_scan_time > config.SCAN_INTERVAL_MS:
        current_scan_index = (current_scan_index + 1) % len(buttons)
        last_scan_time = current_time

    # ⚡ Frame Processing
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (config.CAM_WIDTH, config.CAM_HEIGHT))
        frame = cv2.flip(frame, 1)

        found_authorized_user_this_frame = False

        # ⚡ Face Detection (Skip frames)
        if current_frame_count % config.FACE_DETECTION_SKIP_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_gray, scale = resize_frame_for_detection(gray)
            face_rects_small = dlib_face_detector(small_gray, 0)
            cached_face_rects = [scale_rect(r, config.DETECTION_SCALE) for r in face_rects_small]
            last_detection_time = current_time

        # Process detected faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        is_closed_this_frame = cached_is_closed
        ear_this_frame = cached_ear
        left_eye_this_frame = cached_left_eye
        right_eye_this_frame = cached_right_eye
        
        eyes_closed_duration = 0
        show_progress_bar = False

        for rect in cached_face_rects:
            # ⚡ Auth check (Skip frames)
            if current_frame_count % config.AUTH_CHECK_SKIP_FRAMES == 0:
                name, distance = authenticator.recognize_face(frame, rect)
                if name != "Unknown":
                    found_authorized_user_this_frame = True
                    is_authorized = True
                    authorized_user = name
                    last_auth_check_time = current_time

            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())

            # Authorized user processing
            if is_authorized and authorized_user:
                
                if w < MIN_FACE_WIDTH_FOR_BLINK:
                    cv2.putText(frame, "Jarak terlalu jauh untuk deteksi mata", (x, y - 10), FONT_CV, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    eyes_closed_start_time = None
                    show_progress_bar = False
                    # Continue to next face
                    is_closed_this_frame = False
                else:
                    # ⚡ Landmark prediction (Skip frames)
                    if current_frame_count % config.LANDMARK_SKIP_FRAMES == 0:
                        landmarks = dlib_landmark_predictor(gray, rect)
                        shape = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                        is_closed_this_frame, ear_this_frame, left_eye_this_frame, right_eye_this_frame = blinker.get_ear_status(shape)

                        cached_is_closed = is_closed_this_frame
                        cached_ear = ear_this_frame
                        cached_left_eye = left_eye_this_frame
                        cached_right_eye = right_eye_this_frame

                    # ✅ Blink Duration Logic
                    if is_closed_this_frame:
                        # 2 Mata tertutup
                        if eyes_closed_start_time is None:
                            eyes_closed_start_time = current_time
                            print(f"👁️ Mata mulai tertutup...")
                            
                        eyes_closed_duration = current_time - eyes_closed_start_time
                        show_progress_bar = True
                    else:
                        if eyes_closed_start_time is not None:
                            closed_duration = current_time - eyes_closed_start_time
                            print(f"👁️ Mata dibuka, durasi: {closed_duration}ms")

                            if (closed_duration >= config.BLINK_DURATION_THRESHOLD_MS and
                                (current_time - last_mqtt_trigger_time) > config.BLINK_COOLDOWN_MS):

                                print(f"✅ TRIGGER! Mata tertutup {closed_duration}ms")
                                mqtt_publisher.publish_async(config.MQTT_TOPIC, config.MQTT_MESSAGE)
                                last_mqtt_trigger_time = current_time

                            eyes_closed_start_time = None

                # Visual Feedback
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_AUTH, 2)
                cv2.putText(frame, f"AUTH: {authorized_user}", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_AUTH, 2)
                
                if left_eye_this_frame is not None and right_eye_this_frame is not None:
                    eye_color = (0, 255, 0) if is_closed_this_frame else (255, 0, 0)
                    cv2.drawContours(frame, [cv2.convexHull(left_eye_this_frame)], -1, eye_color, 1)
                    cv2.drawContours(frame, [cv2.convexHull(right_eye_this_frame)], -1, eye_color, 1)
                
                cv2.putText(frame, f"EAR: {ear_this_frame:.2f}", (x, y + h + 15), FONT_CV, 0.5, (255, 255, 255), 1)

                # Progress Bar
                if show_progress_bar:
                    progress_percentage = min(100, int(eyes_closed_duration / config.BLINK_DURATION_THRESHOLD_MS * 100))
                    bar_width = 150
                    bar_height = 20
                    bar_x = x
                    bar_y = y + h + 30
                    
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    
                    progress_width = int(bar_width * progress_percentage / 100)
                    bar_color = (0, 255, 0) if progress_percentage >= 100 else (0, 165, 255)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255,255,255), 1)
                    
                    cv2.putText(frame, f"Tahan... {progress_percentage}%", (bar_x, bar_y - 5), FONT_CV, 0.5, (255, 255, 255), 1)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), config.COLOR_STATUS_UNAUTH, 2)
                cv2.putText(frame, "UNAUTH", (x, y - 10), FONT_CV, 0.7, config.COLOR_STATUS_UNAUTH, 2)

        # Reset auth
        if not found_authorized_user_this_frame and (current_time - last_auth_check_time) > 1000:
            is_authorized = False
            authorized_user = None
            eyes_closed_start_time = None
            mqtt_triggered = False

        # Status Text
        status_text = f"AUTH: {authorized_user}" if is_authorized else "WAITING"
        status_color = config.COLOR_STATUS_AUTH if is_authorized else config.COLOR_STATUS_WAITING
        cv2.putText(frame, status_text, (10, 30), FONT_CV, 0.8, status_color, 2)

        # Pygame Conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

    else:
        frame_surface = pygame.Surface((config.CAM_WIDTH, config.CAM_HEIGHT))
        frame_surface.fill((0, 0, 0))

    # Rendering
    screen.fill(config.COLOR_UI_BACKGROUND)
    screen.blit(frame_surface, (0, 0))

    for i, button in enumerate(buttons):
        pygame.draw.rect(screen, config.COLOR_BUTTON, button["rect"])
        pygame.draw.rect(screen, config.COLOR_BUTTON_TEXT, button["rect"], 2)
        draw_text_center(screen, button["label"], FONT_BUTTON, config.COLOR_BUTTON_TEXT, button["rect"])

    pygame.display.flip()
    clock.tick(config.TARGET_FPS)

# --- 10. Cleanup ---
print("🛑 Shutting down...")
cap.release()
mqtt_publisher.stop()
pygame.quit()
sys.exit()