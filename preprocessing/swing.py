# INFO: Modul untuk mendeteksi gestur gerakan kepala (Head Pose)
# Refactored: Logika Cooldown diperbaiki agar tidak memblokir pemrosesan data.

import numpy as np
from scipy.spatial import distance as dist

def _get_distance(p1, p2):
    """Helper untuk menghitung jarak Euclidean"""
    return dist.euclidean(p1, p2)

class HeadPoseProcessor:
    """
    Class ini mengelola state dan logika untuk deteksi gestur kepala.
    """
    
    def __init__(self, config):
        self.config = config
        
        # State Internal
        self.is_calibrated = False
        self.neutral_h_ratio = 0.5
        self.neutral_v_ratio = 0.5
        
        self.current_gesture = "NEUTRAL"
        self.last_trigger_time = 0
        
        print("✅ Modul HeadPoseProcessor (Kontrol TV) diinisialisasi.")

    def _get_ratios(self, shape):
        """Menghitung rasio horizontal dan vertikal dari landmark."""
        nose_tip = shape[30]
        chin = shape[8]
        left_face_edge = shape[0]
        right_face_edge = shape[16]
        mid_eyes_top = shape[27]
        
        dist_left = _get_distance(nose_tip, left_face_edge)
        dist_right = _get_distance(nose_tip, right_face_edge)
        total_h_dist = dist_left + dist_right
        h_ratio = dist_left / total_h_dist if total_h_dist > 0 else 0.5

        dist_up = _get_distance(mid_eyes_top, nose_tip)
        dist_down = _get_distance(nose_tip, chin)
        total_v_dist = dist_up + dist_down
        v_ratio = dist_up / total_v_dist if total_v_dist > 0 else 0.5
        
        return h_ratio, v_ratio

    def calibrate(self, shape):
        """Mengatur posisi kepala saat ini sebagai Netral."""
        self.neutral_h_ratio, self.neutral_v_ratio = self._get_ratios(shape)
        self.is_calibrated = True
        print("✅ Kalibrasi Gestur Kepala Selesai.")

    def process_frame(self, shape, current_time):
        """
        Memproses landmark untuk mendeteksi gestur.
        Logika Cooldown diterapkan HANYA saat memicu aksi, bukan memblokir kalkulasi.
        """
        
        # --- 1. Hitung Rasio Real-time ---
        h_ratio, v_ratio = self._get_ratios(shape)
        
        # Siapkan data display default (selalu update agar UI responsif)
        display_data = {
            "h_ratio": h_ratio,
            "v_ratio": v_ratio,
            "gesture": self.current_gesture
        }
        
        if not self.is_calibrated:
            return "NEEDS_CALIBRATION", display_data

        # Hitung selisih dari netral
        relative_h = h_ratio - self.neutral_h_ratio
        relative_v = v_ratio - self.neutral_v_ratio
        
        # --- 2. Tentukan Kandidat Gestur Baru ---
        candidate_gesture = "NEUTRAL"
        potential_action = None
        
        # Cek Vertikal (Pitch) - Logika Arah yang Benar (Up=Negatif)
        if relative_v < self.config.PITCH_LOOK_UP_THRESHOLD: 
            candidate_gesture = "UP"
            potential_action = "CH_PLUS"
        elif relative_v > self.config.PITCH_LOOK_DOWN_THRESHOLD:
            candidate_gesture = "DOWN"
            potential_action = "CH_MINUS"
        # Cek Horizontal (Yaw)
        elif relative_h > self.config.YAW_RIGHT_THRESHOLD:
            candidate_gesture = "RIGHT"
            potential_action = "VOL_PLUS"
        elif relative_h < self.config.YAW_LEFT_THRESHOLD:
            candidate_gesture = "LEFT"
            potential_action = "VOL_MINUS"
            
        # --- 3. Logika State Machine & Cooldown ---
        final_action_to_return = None
        
        # A. Transisi dari NEUTRAL ke AKSI
        if self.current_gesture == "NEUTRAL" and candidate_gesture != "NEUTRAL":
            # >>> CEK COOLDOWN DI SINI <<<
            if (current_time - self.last_trigger_time) > self.config.HEAD_GESTURE_COOLDOWN_MS:
                # Cooldown selesai, IZINKAN aksi
                self.current_gesture = candidate_gesture
                self.last_trigger_time = current_time
                final_action_to_return = potential_action
                
                print(f"🎵 Gestur TV Triggered: {candidate_gesture} -> {potential_action}")
            else:
                # Masih cooldown, ABAIKAN aksi tapi jangan blokir program
                # Kita biarkan current_gesture tetap NEUTRAL sampai user mencoba lagi nanti
                pass

        # B. Transisi dari AKSI kembali ke NEUTRAL (Reset)
        elif self.current_gesture != "NEUTRAL" and candidate_gesture == "NEUTRAL":
            # Selalu izinkan reset ke netral tanpa cooldown
            self.current_gesture = "NEUTRAL"
            # print("...Reset ke Netral")

        # C. Ganti Aksi Langsung (opsional, misal dari Kiri langsung ke Kanan)
        elif self.current_gesture != "NEUTRAL" and candidate_gesture != self.current_gesture:
             if (current_time - self.last_trigger_time) > self.config.HEAD_GESTURE_COOLDOWN_MS:
                self.current_gesture = candidate_gesture
                self.last_trigger_time = current_time
                final_action_to_return = potential_action
                print(f"🎵 Gestur Switch: {candidate_gesture} -> {potential_action}")

        # Update data display dengan status gesture yang sebenarnya
        display_data["gesture"] = self.current_gesture
        
        return final_action_to_return, display_data