# INFO: Modul Deteksi Gerakan Kepala (Head Pose) untuk Kontrol Lampu & SOS
# Logic Baru:
# 1. TENGOK KANAN (Tahan 2 detik) -> Nyalakan Lampu
# 2. TENGOK KIRI (Tahan 2 detik) -> Matikan Lampu
# 3. GELENG KEPALA (Shake) -> SOS

import numpy as np
from scipy.spatial import distance as dist
from collections import deque

# MediaPipe Indices
MP_NOSE_TIP = 1
MP_FACE_LEFT = 234
MP_FACE_RIGHT = 454

class HeadPoseProcessor:
    def __init__(self, config):
        self.config = config
        
        # State Internal untuk HOLD (Tahan Posisi)
        self.current_pose = "CENTER" # CENTER, LEFT, RIGHT
        self.pose_start_time = 0
        self.last_action_time = 0
        
        # State Internal untuk SHAKE (Geleng-geleng SOS)
        # Kita simpan urutan pose terakhir: ['LEFT', 'CENTER', 'RIGHT', 'CENTER', ...]
        self.pose_history = deque(maxlen=10) 
        self.last_shake_check_time = 0
        
        # Konstanta Threshold (Sesuaikan jika perlu)
        self.RATIO_LEFT_THRESH = 0.35 # < 0.35 dianggap Kiri
        self.RATIO_RIGHT_THRESH = 0.65 # > 0.65 dianggap Kanan
        
        # Durasi
        self.HOLD_DURATION_MS = 2000 # 2 Detik
        self.COOLDOWN_MS = 3000      # Jeda antar aksi
        
        print("✅ HeadPoseProcessor (Lampu & SOS) siap.")

    def _get_horizontal_ratio(self, shape):
        """
        Menghitung rasio posisi hidung relatif horizontal.
        0.0 (Kiri Absolut) <---> 0.5 (Tengah) <---> 1.0 (Kanan Absolut)
        """
        try:
            nose = shape[MP_NOSE_TIP]
            left = shape[MP_FACE_LEFT]
            right = shape[MP_FACE_RIGHT]
            
            # Jarak hidung ke kiri
            d_left = dist.euclidean(nose, left)
            # Jarak total (lebar wajah)
            d_total = dist.euclidean(left, right)
            
            if d_total == 0: return 0.5
            
            ratio = d_left / d_total
            return ratio
        except IndexError:
            return 0.5

    def calibrate(self, shape):
        # Tidak perlu kalibrasi rumit untuk logika ini
        pass

    def process_frame(self, shape, current_time):
        """
        Return: (ActionSignal, DisplayData)
        ActionSignal: "LIGHT_ON", "LIGHT_OFF", "SOS", None
        """
        
        # 1. Tentukan Pose Saat Ini
        ratio = self._get_horizontal_ratio(shape)
        
        detected_pose = "CENTER"
        if ratio < self.RATIO_LEFT_THRESH:
            detected_pose = "LEFT"
        elif ratio > self.RATIO_RIGHT_THRESH:
            detected_pose = "RIGHT"
            
        # 2. Update History untuk Deteksi SHAKE (SOS)
        # Hanya catat perubahan pose (e.g., L -> C -> R)
        if not self.pose_history or self.pose_history[-1]["pose"] != detected_pose:
            self.pose_history.append({"pose": detected_pose, "time": current_time})
            
        # --- LOGIKA DETEKSI SOS (GELENG KEPALA) ---
        # Pola: Left -> Right -> Left (atau sebaliknya) dalam waktu singkat
        # Filter history: ambil urutan L, R, L, R... abaikan CENTER
        non_center_history = [item for item in self.pose_history if item["pose"] != "CENTER"]
        
        if len(non_center_history) >= 4:
            # Cek pola selang-seling (L, R, L, R)
            p1 = non_center_history[-1]
            p2 = non_center_history[-2]
            p3 = non_center_history[-3]
            p4 = non_center_history[-4]
            
            # Cek waktu total (harus cepat, misal semua terjadi dalam 3 detik)
            duration = p1["time"] - p4["time"]
            
            if duration < 3000: # 3 detik
                # Cek apakah arahnya bolak-balik
                if p1["pose"] != p2["pose"] and p2["pose"] != p3["pose"]:
                    # Validasi SOS!
                    # Bersihkan history agar tidak trigger ganda
                    self.pose_history.clear() 
                    return "SOS", {"pose": "SHAKE!!!", "progress": 100}

        # --- LOGIKA DETEKSI HOLD (LAMPU) ---
        
        trigger_action = None
        progress = 0
        
        # State Machine untuk Hold
        if detected_pose == self.current_pose:
            # Pose masih sama, hitung durasi
            elapsed = current_time - self.pose_start_time
            
            # Hitung progress bar (0-100%)
            progress = min(100, int((elapsed / self.HOLD_DURATION_MS) * 100))
            
            if elapsed >= self.HOLD_DURATION_MS:
                # Cek cooldown global
                if (current_time - self.last_action_time) > self.COOLDOWN_MS:
                    if detected_pose == "RIGHT":
                        trigger_action = "LIGHT_ON"
                        self.last_action_time = current_time
                    elif detected_pose == "LEFT":
                        trigger_action = "LIGHT_OFF"
                        self.last_action_time = current_time
        else:
            # Pose berubah, reset timer
            self.current_pose = detected_pose
            self.pose_start_time = current_time
            progress = 0
            
        display_data = {
            "pose": detected_pose,
            "ratio": ratio,
            "progress": progress
        }
            
        return trigger_action, display_data