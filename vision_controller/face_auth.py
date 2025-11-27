import numpy as np
import pickle
import cv2
from keras_facenet import FaceNet # <-- IMPORT KUNCI

class FaceAuthenticator:
    
    def __init__(self, embeddings_path, model_input_size=(160, 160), tolerance=1.0):
        """
        Inisialisasi authenticator.
        
        Args:
            embeddings_path (str): Path ke file .pkl berisi embeddings.
            model_input_size (tuple): Ukuran input model (default FaceNet 160x160).
            tolerance (float): Jarak (distance) maksimum untuk dianggap sebagai 
                               orang yang sama. Nilai yang lebih rendah lebih ketat.
                               Untuk FaceNet, nilai antara 0.7 - 1.1 adalah umum.
        """
        print("🔄 Menginisialisasi modul otentikasi wajah (FaceNet)...")
        self.input_size = model_input_size
        self.tolerance = tolerance 
        print(f"⚙️ Tolerance diset ke: {self.tolerance}")
        
        # --- Muat Model FaceNet Asli ---
        # Ini adalah model yang SAMA PERSIS dengan yang digunakan di 'encoded_face.py'
        try:
            self.embedder = FaceNet()
            print("✅ Model FaceNet pre-trained berhasil dimuat.")
            # Model keras-facenet default menghasilkan embedding 512-dimensi
        except Exception as e:
            print(f"❌ Gagal memuat model FaceNet: {e}")
            exit()
        
        # --- Muat Database Wajah ---
        try:
            with open(embeddings_path, 'rb') as f:
                (self.known_encodings, self.known_names) = pickle.load(f)
            print(f"✅ Database wajah dimuat. Terdapat {len(self.known_names)} wajah terdaftar.")
            # Cek dimensi embedding pertama untuk memastikan
            if len(self.known_encodings) > 0:
                print(f"   Dimensi embedding database: {self.known_encodings[0].shape}")
        except FileNotFoundError:
            print(f"❌ File embedding '{embeddings_path}' tidak ditemukan. Jalankan 'encoded_face.py' terlebih dahulu.")
            exit()

    def recognize_face(self, frame, face_rect):
        """
        Mengenali wajah dari region of interest (ROI) yang diberikan.
        """
        (x, y, w, h) = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        
        # Validasi bounds
        frame_h, frame_w = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        
        if w <= 0 or h <= 0:
            return "Unknown", float('inf')
        
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return "Unknown", float('inf')
        
        try:
            # Pra-pemrosesan untuk FaceNet
            image_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image_rgb, self.input_size)
            face_array = np.asarray(resized_image)
            
            # Dapatkan embedding dari wajah yang terdeteksi
            # Ini sekarang akan menghasilkan embedding 512-dimensi
            current_embedding = self.embedder.embeddings([face_array])[0]
            
            # Bandingkan dengan database
            return self._find_closest_match(current_embedding)
        except Exception as e:
            # Menangkap error jika wajah terlalu kecil atau di luar frame
            # print(f"⚠️ Error processing face: {e}") # Aktifkan untuk debug
            return "Unknown", float('inf')

    def _find_closest_match(self, face_embedding):
        """Mencari nama yang paling cocok dari database wajah."""
        if not self.known_encodings:
            return "Unknown", float('inf')
            
        # Hitung jarak L2 (Euclidean)
        # Ini sekarang akan membandingkan (N, 512) dengan (512,) -> VALID!
        distances = np.linalg.norm(np.array(self.known_encodings) - face_embedding, axis=1)
        
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
              
        # Logika keputusan yang sekarang berfungsi
        if min_distance <= self.tolerance:
            # Jaraknya cukup dekat -> ini adalah orang yang kita kenal
            return self.known_names[min_distance_index], min_distance
        else:
            # Jaraknya terlalu jauh -> ini adalah orang asing
            return "Unknown", min_distance