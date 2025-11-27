# INFO: Module untuk mendeteksi gestur kedipan mata (Aksi & Ganti Mode)

import numpy as np
from scipy.spatial import distance as dist
import cv2

# Dlib landmark indices untuk mata
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

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

    def _calculate_ear(self, eye):
        """Menghitung Eye Aspect Ratio (EAR)."""
        # Jarak vertikal
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Jarak horizontal
        C = dist.euclidean(eye[0], eye[3])
        # Hindari pembagian dengan nol
        if C == 0:
            return 0.3 # Nilai netral
        ear = (A + B) / (2.0 * C)
        return ear

    def _get_ear_status(self, shape):
        """Mendapatkan status EAR dari 68 landmark."""
        leftEye = shape[L_START:L_END]
        rightEye = shape[R_START:R_END]
        leftEAR = self._calculate_ear(leftEye)
        rightEAR = self._calculate_ear(rightEye)
        
        # Rata-ratakan EAR
        ear = (leftEAR + rightEAR) / 2.0
        
        is_closed = ear < self.config.ACTION_BLINK_DURATION_MS # Ini sepertinya salah di config lama, harusnya EAR_THRESH
        # Mari kita hardcode threshold EAR untuk saat ini
        EAR_THRESHOLD = 0.23 # Threshold EAR standar
        
        is_closed = ear < EAR_THRESHOLD
        
        return is_closed, ear, leftEye, rightEye

    def process_frame(self, gray, rect, dlib_predictor, current_time, frame_count):
        """
        Memproses frame saat ini untuk gestur kedipan.
        
        Mengembalikan:
            str: Sinyal aksi ("ACTION_TRIGGER", "MODE_SWITCH") atau None.
            dict: Data untuk ditampilkan di UI.
        """
        
        action_triggered = None
        
        # --- 1. Prediksi Landmark (di-cache) ---
        if frame_count % self.config.LANDMARK_SKIP_FRAMES == 0:
            landmarks = dlib_predictor(gray, rect)
            self.cached_shape = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            
            is_closed, ear, left_eye, right_eye = self._get_ear_status(self.cached_shape)
            
            # Update cache
            self.cached_is_closed = is_closed
            self.cached_ear = ear
            self.cached_left_eye = left_eye
            self.cached_right_eye = right_eye
        
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
                    
                    # Cek GANTI MODE (prioritas tertinggi)
                    if closed_duration >= self.config.MODE_SWITCH_BLINK_DURATION_MS:
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
            # Kalibrasi progress bar ke durasi GANTI MODE
            progress_percentage = min(100, int(closed_duration / self.config.MODE_SWITCH_BLINK_DURATION_MS * 100))
        
        display_data = {
            "is_closed": self.cached_is_closed,
            "ear": self.cached_ear,
            "left_eye": self.cached_left_eye,
            "right_eye": self.cached_right_eye,
            "closed_duration": closed_duration,
            "progress_percentage": progress_percentage
        }
        
        return action_triggered, display_data, self.cached_shape