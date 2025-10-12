import numpy as np
import pickle
import cv2
from tensorflow import keras

class FaceAuthenticator:
    """
    Sebuah kelas untuk menangani semua logika terkait pengenalan wajah.
    Ini memuat model, database wajah, dan melakukan perbandingan.
    """
    def __init__(self, embeddings_path, model_path="face_embedding_model.h5", model_input_size=(160, 160), tolerance=1.5):
        """
        Inisialisasi authenticator dengan memuat semua aset yang diperlukan.
        """
        print("🔄 Menginisialisasi modul otentikasi wajah...")
        self.input_size = model_input_size
        self.tolerance = tolerance
        self.model_path = model_path
        print(f"⚙️ Tolerance diset ke: {self.tolerance}")
        
        # Muat model TensorFlow yang sudah di-save (bukan buat baru!)
        self.embedding_model = self._load_embedding_model()
        
        # Muat database wajah yang sudah di-train
        try:
            with open(embeddings_path, 'rb') as f:
                (self.known_encodings, self.known_names) = pickle.load(f)
            print(f"✅ Database wajah dimuat. Terdapat {len(self.known_names)} wajah terdaftar.")
        except FileNotFoundError:
            print(f"❌ File embedding '{embeddings_path}' tidak ditemukan. Jalankan 'encoded_face.py' terlebih dahulu.")
            exit()

    def _load_embedding_model(self):
        """Load model yang sudah di-save (PENTING: gunakan model yang sama dengan saat training!)"""
        import os
        if os.path.exists(self.model_path):
            print(f"✅ Loading model dari {self.model_path}")
            model = keras.models.load_model(self.model_path)
            return model
        else:
            print(f"⚠️ Model file '{self.model_path}' tidak ditemukan!")
            print(f"   Membuat model baru (WARNING: ini akan menghasilkan embedding yang berbeda!)")
            return self._create_embedding_model()

    def _create_embedding_model(self):
        """Membangun struktur model Keras untuk inferensi embedding."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size[0], self.input_size[1], 3)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'), keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'), keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(), keras.layers.Dense(128, activation=None)
        ])
        print("✅ Model TensorFlow (embedding) berhasil dibangun.")
        return model

    def recognize_face(self, frame, face_rect):
        """
        Mengenali wajah dari region of interest (ROI) yang diberikan dalam sebuah frame.
        """
        (x, y, w, h) = (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height())
        
        # Validasi bounds - pastikan ROI tidak keluar dari frame
        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        # Pastikan ROI valid (tidak kosong)
        if w <= 0 or h <= 0:
            return "Unknown", float('inf')
        
        # Ekstrak ROI dan lakukan pra-pemrosesan
        face_roi = frame[y:y+h, x:x+w]
        
        # Double check ROI tidak kosong
        if face_roi.size == 0:
            return "Unknown", float('inf')
        
        try:
            face_for_embedding = cv2.resize(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB), self.input_size)
            face_for_embedding = np.expand_dims(face_for_embedding.astype('float32') / 255.0, axis=0)
            
            # Dapatkan embedding dari wajah yang terdeteksi
            current_embedding = self.embedding_model.predict(face_for_embedding, verbose=0)[0]
            
            # Bandingkan dengan database
            return self._find_closest_match(current_embedding)
        except Exception as e:
            print(f"⚠️ Error processing face: {e}")
            return "Unknown", float('inf')

    def _find_closest_match(self, face_embedding):
        """Mencari nama yang paling cocok dari database wajah."""
        if not self.known_encodings:
            return "Unknown", float('inf')
            
        distances = np.linalg.norm(np.array(self.known_encodings) - face_embedding, axis=1)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
              
        if min_distance <= self.tolerance:
            return self.known_names[min_distance_index], min_distance
        else:
            return "Unknown", min_distance
