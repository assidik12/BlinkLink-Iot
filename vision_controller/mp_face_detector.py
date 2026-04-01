import cv2
import mediapipe as mp
import numpy as np

class DlibRectLike:
    """
    Kelas pembungkus sederhana untuk meniru objek dlib.rectangle.
    Ini diperlukan agar tidak perlu merombak face_auth.py.
    """
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def left(self): return self.x
    def top(self): return self.y
    def width(self): return self.w
    def height(self): return self.h
    def right(self): return self.x + self.w
    def bottom(self): return self.y + self.h

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True, # Penting untuk akurasi mata (iris)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image):
        """
        Mendeteksi wajah dan mengembalikan bounding box serta landmark.
        
        Args:
            image: Gambar BGR atau RGB (numpy array)
            
        Returns:
            results: List of dict {'rect': DlibRectLike, 'landmarks': np.array}
        """
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        detected_faces = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Konversi Landmarks ke koordinat piksel (Numpy Array)
                # MediaPipe mengembalikan 478 landmarks (468 mesh + 10 iris) jika refine_landmarks=True
                landmarks_np = np.array([
                    (int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark
                ])

                # 2. Hitung Bounding Box dari landmarks
                # Kita ambil min/max x dan y untuk membuat kotak wajah
                x_min = np.min(landmarks_np[:, 0])
                y_min = np.min(landmarks_np[:, 1])
                x_max = np.max(landmarks_np[:, 0])
                y_max = np.max(landmarks_np[:, 1])
                
                # Tambahkan sedikit padding agar tidak terlalu ketat
                padding_w = int((x_max - x_min) * 0.1)
                padding_h = int((y_max - y_min) * 0.1)
                
                x = max(0, x_min - padding_w)
                y = max(0, y_min - padding_h)
                rect_w = min(w, (x_max + padding_w) - x)
                rect_h = min(h, (y_max + padding_h) - y)

                rect = DlibRectLike(x, y, rect_w, rect_h)

                detected_faces.append({
                    'rect': rect,
                    'landmarks': landmarks_np
                })

        return detected_faces
