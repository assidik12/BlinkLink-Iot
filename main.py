import cv2
import dlib
import time
import paho.mqtt.client as mqtt
import numpy as np
import config as config 
import queue

# Impor modul kustom
from vision_controller.face_auth import FaceAuthenticator
from vision_controller.blink_module import BlinkDetector
from vision_controller.voice_command import VoiceCommandHandler
from mqtt import Connect_mqtt 

class BlinkLinkApp:
    """
    Membungkus semua logika aplikasi ke dalam satu kelas.
    Ini mengelola state, inisialisasi modul, dan loop utama.
    """
    
    def __init__(self):
        """
        Inisialisasi semua komponen aplikasi.
        """
        print("🔄 Memulai inisialisasi sistem...")
        
        # --- 1. Inisialisasi State & Antrian ---
        self.command_queue = queue.Queue()
        self.is_authorized = False
        self.authorized_user = None
        
        # Variabel untuk shutdown yang bersih
        self.running = True

        # --- 2. Inisialisasi Modul Logika ---
        try:
            self.authenticator = FaceAuthenticator(
                embeddings_path=config.FACE_EMBEDDINGS_PATH,
                model_path=config.FACE_MODEL_PATH,
                tolerance=1.5
            )
            self.blinker = BlinkDetector()
            self.dlib_face_detector = dlib.get_frontal_face_detector()
            self.dlib_landmark_predictor = dlib.shape_predictor(config.DLIB_LANDMARK_MODEL_PATH)
        except Exception as e:
            print(f"❌ Error saat memuat model Vision: {e}")
            print("Pastikan semua file model (.dat, .pkl, .h5) ada di lokasi yang benar.")
            self.running = False
            return

        # --- 3. Inisialisasi Layanan & Hardware ---
        self.mqtt_client = Connect_mqtt(mqtt, config)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("❌ Error: Tidak dapat membuka webcam.")
            self.running = False
            return
            
        # --- 4. Mulai Thread Latar Belakang (Paralel) ---
        # Thread Suara
        self.voice_handler = VoiceCommandHandler(self.command_queue)
        self.voice_handler.start() # Memulai thread mendengarkan
        
        # Thread MQTT
        if self.mqtt_client:
            self.mqtt_client.loop_start()

        print("📹 Video stream dimulai...")
        time.sleep(1.0) # Beri waktu kamera untuk pemanasan

    def _handle_voice_commands(self):
        """
        Memeriksa antrian (queue) untuk perintah suara yang masuk 
        tanpa memblokir loop utama.
        """
        try:
            # get_nowait() tidak akan memblokir jika antrian kosong
            voice_command = self.command_queue.get_nowait()
            print(f"⚡ Perintah dari suara diterima: {voice_command}")
            
            if self.mqtt_client:
                # Perintah suara bisa dieksekusi kapan saja (otorisasi tidak diperlukan)
                if voice_command == "LIGHT_ON":
                    self.mqtt_client.publish(config.MQTT_MESSAGE, "TOGGLE")
                elif voice_command == "LIGHT_OFF":
                    self.mqtt_client.publish(config.MQTT_MESSAGE, "TOGGLE")
                elif voice_command == "LIGHT_TOGGLE":
                    self.mqtt_client.publish(config.MQTT_MESSAGE, "TOGGLE")
                    
        except queue.Empty:
            # Ini normal, berarti tidak ada perintah suara di antrian.
            pass
        except Exception as e:
            # Tangkap error lain jika ada
            print(f"Error saat memproses antrian suara: {e}")

    def _process_vision_frame(self, frame, gray):
        """
        Memproses satu frame video untuk deteksi wajah, otorisasi,
        dan deteksi kedipan.
        """
        face_rects = self.dlib_face_detector(gray, 0)
        found_authorized_user_this_frame = False

        for rect in face_rects:
            # Langkah 1: Otentikasi Wajah
            name, distance = self.authenticator.recognize_face(frame, rect)
            
            if name != "Unknown":
                found_authorized_user_this_frame = True
                self.is_authorized = True
                self.authorized_user = name

                # Langkah 2: Deteksi Kedipan (Hanya jika terotorisasi)
                landmarks = self.dlib_landmark_predictor(gray, rect)
                shape = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
                
                blink_detected, ear, left_eye, right_eye = self.blinker.detect_blink(shape)

                if blink_detected:
                    print(f"👁️ Kedipan terdeteksi dari {self.authorized_user}! Mengirim perintah...")
                    if self.mqtt_client:
                        self.mqtt_client.publish(config.MQTT_TOPIC, config.MQTT_MESSAGE)
                
                # Gambar umpan balik untuk user terotorisasi
                self._draw_feedback(frame, rect, f"AUTHORIZED: {name}", (0, 255, 0), left_eye, right_eye)
            else:
                # Gambar umpan balik untuk user tidak dikenal
                self._draw_feedback(frame, rect, "UNAUTHORIZED", (0, 0, 255))

        if not found_authorized_user_this_frame:
            # Jika user terotorisasi tidak lagi terlihat, reset status
            self.is_authorized = False
            self.authorized_user = None

    def _draw_feedback(self, frame, rect, text, color, left_eye=None, right_eye=None):
        """
        Helper untuk menggambar kotak dan teks umpan balik pada frame.
        """
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if left_eye is not None and right_eye is not None:
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (255, 0, 0), 1)

    def run(self):
        """
        Menjalankan loop aplikasi utama.
        """
        if not self.running:
            print("Aplikasi gagal diinisialisasi. Keluar.")
            return

        print("\n🚀 Sistem siap. Arahkan wajah Anda ke kamera untuk otorisasi.")

        while self.running:
            # --- 1. Handle Perintah Latar Belakang (Audio) ---
            self._handle_voice_commands()

            # --- 2. Baca Frame Video ---
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Gagal membaca frame. Menutup...")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- 3. Proses Frame (Vision) ---
            self._process_vision_frame(frame, gray)

            # --- 4. Tampilkan Status Global ---
            status_text = f"Status: {'AUTHORIZED (' + str(self.authorized_user) + ')' if self.is_authorized else 'WAITING FOR AUTH'}"
            status_color = (0, 255, 0) if self.is_authorized else (0, 165, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # # --- 5. Tampilkan Hasil ---
            cv2.imshow("Sistem Kontrol IoT Berbasis Visi", frame)

            # --- 6. Cek Perintah Keluar (Tombol 'q') ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False # Kirim sinyal untuk berhenti
        
        # Setelah loop selesai, jalankan cleanup
        self.stop()

    def stop(self):
        """
        Membersihkan semua resource saat aplikasi berhenti.
        """
        print("\n🔌 Menutup sistem...")
        
        # Hentikan thread latar belakang
        self.voice_handler.stop()
        
        # Hentikan hardware & koneksi
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        print("Sistem ditutup dengan aman.")

# --- Titik Masuk Aplikasi ---
if __name__ == "__main__":
    app = BlinkLinkApp()
    app.run()

