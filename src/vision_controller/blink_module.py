from scipy.spatial import distance as dist
import numpy as np

class BlinkDetector:
    """
    Versi modul yang disederhanakan.
    Fungsi utamanya adalah menghitung EAR dan status mata (terbuka/tertutup).
    Logika durasi/counter dipindahkan ke main loop agar lebih bersih.
    """
    def __init__(self, ear_thresh=0.23):
        """
        Inisialisasi detektor dengan threshold EAR.
        """
        self.ear_thresh = ear_thresh

    @staticmethod
    def calculate_ear(eye_landmarks):
        """Menghitung EAR. Dibuat statis karena tidak bergantung pada state internal."""
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def get_ear_status(self, eye_landmarks_shape):
        """
        Menghitung EAR dan mengembalikan status mata.

        Mengembalikan:
        - is_closed (bool): True jika mata tertutup (di bawah threshold)
        - ear (float): Nilai EAR yang dihitung
        - left_eye (ndarray): Koordinat mata kiri
        - right_eye (ndarray): Koordinat mata kanan
        """
        # Indeks landmark mata dlib
        (lStart, lEnd) = (42, 48)
        (rStart, rEnd) = (36, 42)

        left_eye = eye_landmarks_shape[lStart:lEnd]
        right_eye = eye_landmarks_shape[rStart:rEnd]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Rata-rata EAR dari kedua mata
        ear = (left_ear + right_ear) / 2.0
        
        # Tentukan status mata berdasarkan threshold
        is_closed = ear < self.ear_thresh
            
        return is_closed, ear, left_eye, right_eye