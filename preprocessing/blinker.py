# INFO: Module untuk mendeteksi gestur kedipan mata (Aksi & Ganti Mode)

import numpy as np
from scipy.spatial import distance as dist
import cv2

# --- KONFIGURASI LANDMARK ---
# DLIB 68 Indices
DLIB_L_EYE = list(range(42, 48))
DLIB_R_EYE = list(range(36, 42))

# MEDIAPIPE Face Mesh Indices
# P1, P2, P3, P4, P5, P6
# EAR Formula: (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
MP_L_EYE = [33, 160, 158, 133, 153, 144]
MP_R_EYE = [362, 385, 387, 263, 373, 380]

class BlinkProcessor:
    """
    Mengelola state dan logika untuk deteksi gestur kedipan mata.
    Mendeteksi dua jenis kedipan:
    1. ACTION_TRIGGER (Tahan 2 detik)
    2. MODE_SWITCH (Tahan 5 detik)
    """

    def __init__(self, config):
        self.config = config
        
        # State internal
        self.eyes_closed_start_time = None
        self.last_trigger_time = 0
        
        # Caching
        self.cached_is_closed = False
        self.cached_ear = 0.3
        self.cached_left_eye = None
        self.cached_right_eye = None
        self.cached_shape = None
        
        print("✅ Modul BlinkProcessor (Aksi & Ganti Mode) diinisialisasi.")

    def _calculate_ear(self, eye_points):
        """
        Menghitung Eye Aspect Ratio (EAR).
        eye_points: list/array of (x, y) coordinates. Expected 6 points.
        """
        # P2-P6 (Vertical 1)
        A = dist.euclidean(eye_points[1], eye_points[5])
        # P3-P5 (Vertical 2)
        B = dist.euclidean(eye_points[2], eye_points[4])
        # P1-P4 (Horizontal)
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Hindari pembagian dengan nol
        if C == 0:
            return 0.3 # Nilai netral
        ear = (A + B) / (2.0 * C)
        return ear

    def _get_ear_status(self, shape, is_mediapipe=False):
        """Mendapatkan status EAR dengan Hysteresis (Anti-Flicker)."""
        if is_mediapipe:
            left_indices = MP_L_EYE
            right_indices = MP_R_EYE
            # Gunakan config, jangan hardcode
            THRESH_CLOSE = getattr(self.config, 'EAR_THRESHOLD_CLOSE', 0.18)
            THRESH_OPEN = getattr(self.config, 'EAR_THRESHOLD_OPEN', 0.25)
        else:
            left_indices = DLIB_L_EYE
            right_indices = DLIB_R_EYE
            THRESH_CLOSE = getattr(self.config, 'EAR_THRESHOLD_CLOSE', 0.20) + 0.03 # Dlib biasanya lebih tinggi
            THRESH_OPEN = getattr(self.config, 'EAR_THRESHOLD_OPEN', 0.28) + 0.03
            
        leftEye = shape[left_indices]
        rightEye = shape[right_indices]
        
        leftEAR = self._calculate_ear(leftEye)
        rightEAR = self._calculate_ear(rightEye)
        
        # Rata-ratakan EAR
        ear = (leftEAR + rightEAR) / 2.0
        
        # Logika Hysteresis
        # Jika sekarang tertutup, kita butuh nilai EAR > THRESH_OPEN untuk membukanya kembali
        # Jika sekarang terbuka, kita butuh nilai EAR < THRESH_CLOSE untuk menutupnya
        
        if self.cached_is_closed:
            # Sedang tertutup. Apakah sudah cukup terbuka?
            is_closed = ear < THRESH_OPEN
        else:
            # Sedang terbuka. Apakah sudah cukup tertutup?
            is_closed = ear < THRESH_CLOSE
            
        return is_closed, ear, leftEye, rightEye

    def process_frame(self, gray, rect, dlib_predictor, current_time, frame_count, landmarks=None):
        """
        Memproses frame saat ini untuk gestur kedipan.
        
        Args:
            gray: Frame grayscale (tidak dipakai jika landmarks disediakan)
            rect: Bounding box wajah
            dlib_predictor: Dlib predictor (opsional jika landmarks disediakan)
            current_time: Waktu pygame saat ini
            frame_count: Counter frame
            landmarks: (Opsional) Numpy array landmarks dari MediaPipe
        
        Mengembalikan:
            str: Sinyal aksi ("ACTION_TRIGGER", "MODE_SWITCH") atau None.
            dict: Data untuk ditampilkan di UI.
        """
        
        action_triggered = None
        is_mediapipe = False
        
        # --- 1. Prediksi Landmark (di-cache atau langsung) ---
        if landmarks is not None:
            # Jika menggunakan MediaPipe, kita selalu dapat landmarks baru setiap frame
            # Tidak perlu caching skip frame yang agresif karena MP sudah cepat
            self.cached_shape = landmarks
            is_mediapipe = True
            
            # Hitung EAR setiap frame (atau bisa di-skip jika perlu performa ekstra)
            is_closed, ear, left_eye, right_eye = self._get_ear_status(self.cached_shape, is_mediapipe=True)
            
            self.cached_is_closed = is_closed
            self.cached_ear = ear
            self.cached_left_eye = left_eye
            self.cached_right_eye = right_eye
            
        elif frame_count % self.config.LANDMARK_SKIP_FRAMES == 0:
            # Legacy Dlib Path
            if dlib_predictor is not None:
                landmarks_dlib = dlib_predictor(gray, rect)
                self.cached_shape = np.array([(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(68)])
                
                is_closed, ear, left_eye, right_eye = self._get_ear_status(self.cached_shape, is_mediapipe=False)
                
                # Update cache
                self.cached_is_closed = is_closed
                self.cached_ear = ear
                self.cached_left_eye = left_eye
                self.cached_right_eye = right_eye
        
        # --- 2. Logika Durasi Kedipan ---
        
        # --- 2. Logika Durasi Kedipan ---
        closed_duration = 0
        
        if self.cached_is_closed:
            # Mata tertutup, mulai/lanjutkan timer
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = current_time
                print(f"👁️ Mata mulai tertutup...")
                
            closed_duration = current_time - self.eyes_closed_start_time
        
        else:
            # Mata terbuka, cek durasi jika sebelumnya tertutup
            if self.eyes_closed_start_time is not None:
                closed_duration = current_time - self.eyes_closed_start_time
                print(f"👁️ Mata dibuka, durasi: {closed_duration}ms")

                # Cek apakah dalam masa cooldown
                if (current_time - self.last_trigger_time) > self.config.BLINK_COOLDOWN_MS:
                    
                    # Cek SOS (Prioritas Paling TINGGI)
                    if closed_duration >= self.config.SOS_BLINK_DURATION_MS:
                        print(f"🚨 TRIGGER! SOS BAHAYA (durasi: {closed_duration}ms)")
                        action_triggered = "TRIGGER_SOS"
                        self.last_trigger_time = current_time

                    # Cek GANTI MODE
                    elif closed_duration >= self.config.MODE_SWITCH_BLINK_DURATION_MS:
                        print(f"✅ TRIGGER! Ganti Mode (durasi: {closed_duration}ms)")
                        action_triggered = "TRIGGER_MODE_SWITCH"
                        self.last_trigger_time = current_time
                    
                    # Cek AKSI
                    elif closed_duration >= self.config.ACTION_BLINK_DURATION_MS:
                        print(f"✅ TRIGGER! Aksi (durasi: {closed_duration}ms)")
                        action_triggered = "TRIGGER_ACTION"
                        self.last_trigger_time = current_time
                
                else:
                    print("... (Dalam masa cooldown, aksi diabaikan)")

                # Reset timer
                self.eyes_closed_start_time = None
        
        # --- 3. Siapkan Data Display ---
        progress_percentage = 0
        if self.eyes_closed_start_time is not None:
            # Kalibrasi progress bar ke durasi AKSI (karena user fokus ke aksi lampu)
            # Cap di 100% jika lebih
            progress_percentage = min(100, int(closed_duration / self.config.ACTION_BLINK_DURATION_MS * 100))
        
        display_data = {
            "is_closed": self.cached_is_closed,
            "ear": self.cached_ear,
            "left_eye": self.cached_left_eye,
            "right_eye": self.cached_right_eye,
            "closed_duration": closed_duration,
            "progress_percentage": progress_percentage
        }
        
        return action_triggered, display_data, self.cached_shape