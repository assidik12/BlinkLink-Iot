from scipy.spatial import distance as dist
import numpy as np

class BlinkDetector:
    """
    Sebuah kelas untuk menangani semua logika deteksi kedipan berdasarkan
    Eye Aspect Ratio (EAR).
    """
    def __init__(self, ear_thresh=0.23, consec_frames=3):
        """
        Inisialisasi detektor dengan threshold yang ditentukan.
        """
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.blink_counter = 0

    @staticmethod
    def calculate_ear(eye_landmarks):
        """Menghitung EAR. Dibuat statis karena tidak bergantung pada state internal."""
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, eye_landmarks_shape):
        """
        Memproses landmark mata untuk mendeteksi apakah sebuah kedipan baru saja terjadi.
        Mengembalikan True jika ada kedipan, False jika tidak.
        """
        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)

        left_eye = eye_landmarks_shape[lStart:lEnd]
        right_eye = eye_landmarks_shape[rStart:rEnd]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        blink_detected = False
        if ear < self.ear_thresh:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.consec_frames:
                blink_detected = True # Kedipan terkonfirmasi!
            self.blink_counter = 0 # Reset counter
            
        return blink_detected, ear, left_eye, right_eye
