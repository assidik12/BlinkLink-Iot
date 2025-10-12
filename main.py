import cv2
import dlib
import time
import paho.mqtt.client as mqtt
import numpy as np

# Impor modul kustom kita
from vision_controller.face_auth import FaceAuthenticator
from vision_controller.blink_module import BlinkDetector

# --- 1. KONFIGURASI UTAMA ---
DLIB_LANDMARK_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"
FACE_EMBEDDINGS_PATH = "./face_embeddings_tf.pkl"
FACE_MODEL_PATH = "./face_embedding_model.h5"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/commands/eyeblink"
MQTT_MESSAGE = "TOGGLE"

# --- 2. FUNGSI HELPER & INISIALISASI ---
def connect_mqtt():
    """Menyiapkan dan menghubungkan client MQTT."""
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ Berhasil terhubung ke MQTT Broker!")
        else:
            print(f"❌ Gagal terhubung ke MQTT Broker, return code {rc}")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        return client
    except Exception as e:
        print(f"❌ Tidak bisa terhubung ke broker MQTT: {e}")
        return None

print("🔄 Memulai inisialisasi sistem...")

# Inisialisasi modul-modul
authenticator = FaceAuthenticator(embeddings_path=FACE_EMBEDDINGS_PATH, model_path=FACE_MODEL_PATH, tolerance=1.5)
blinker = BlinkDetector()
dlib_face_detector = dlib.get_frontal_face_detector()
dlib_landmark_predictor = dlib.shape_predictor(DLIB_LANDMARK_MODEL_PATH)
mqtt_client = connect_mqtt()

# Inisialisasi webcam
print("📹 Memulai video stream...")
cap = cv2.VideoCapture(0)

# Set resolusi kamera (opsional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(1.0)

# --- 3. LOOP UTAMA ---
print("\n🚀 Sistem siap. Arahkan wajah Anda ke kamera untuk otorisasi.")

is_authorized = False
authorized_user = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_rects = dlib_face_detector(gray, 0)
    
    found_authorized_user_this_frame = False
    for rect in face_rects:
        # Langkah 1: Otentikasi Wajah
        name, distance = authenticator.recognize_face(frame, rect)
        
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())

        if name != "Unknown":
            found_authorized_user_this_frame = True
            is_authorized = True
            authorized_user = name

            # Langkah 2: Deteksi Kedipan (Hanya jika terotorisasi)
            landmarks = dlib_landmark_predictor(gray, rect)
            shape = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            
            blink_detected, ear, left_eye, right_eye = blinker.detect_blink(shape)

            if blink_detected:
                print(f"👁️ Kedipan terdeteksi dari {authorized_user}! Mengirim perintah...")
                if mqtt_client:
                    mqtt_client.publish(MQTT_TOPIC, MQTT_MESSAGE)

            # --- Umpan Balik Visual ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"AUTHORIZED: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (255, 0, 0), 1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "UNAUTHORIZED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not found_authorized_user_this_frame:
        is_authorized = False
        authorized_user = None

    status_text = f"Status: {'AUTHORIZED (' + str(authorized_user) + ')' if is_authorized else 'WAITING FOR AUTH'}"
    status_color = (0, 255, 0) if is_authorized else (0, 165, 255)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("Sistem Kontrol IoT Berbasis Visi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
print("\n🔌 Menutup sistem...")
cap.release()
cv2.destroyAllWindows()
if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
print("Sistem ditutup dengan aman.")

