import cv2
import os
import pickle
import numpy as np
from keras_facenet import FaceNet # <-- IMPORT BARU

# --- Konfigurasi ---
dataset_dir = "datasets"
embeddings_file = "face_embeddings_tf.pkl"
model_input_size = (160, 160) # FaceNet menggunakan input 160x160

# --- Muat Model FaceNet Asli ---
# Ini akan mengunduh model pre-trained secara otomatis saat pertama kali dijalankan
try:
    embedder = FaceNet()
    print("✅ Model FaceNet pre-trained berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model FaceNet: {e}")
    print("Pastikan Anda terhubung ke internet untuk mengunduh model.")
    exit()

known_face_encodings = []
known_face_names = []

print("Memulai proses pembuatan embeddings (FaceNet)...")

# Iterasi melalui setiap subfolder (nama orang) di dalam folder 'datasets'
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    if os.path.isdir(person_dir):
        print(f"Memproses wajah untuk: {person_name}")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  - Peringatan: Gagal memuat {filename}.")
                        continue

                    # Konversi BGR (OpenCV) ke RGB (FaceNet)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Ubah ukuran gambar agar sesuai dengan input model
                    resized_image = cv2.resize(image_rgb, model_input_size)
                    
                    # FaceNet tidak memerlukan normalisasi /255.0
                    # Ia membutuhkan input sebagai array numpy
                    face_array = np.asarray(resized_image)

                    # Dapatkan embedding
                    # Model FaceNet membutuhkan 'batch' (daftar wajah)
                    # Kita beri [face_array] untuk membuat batch berisi 1 wajah
                    embedding = embedder.embeddings([face_array])[0] # Ambil embedding pertama

                    known_face_encodings.append(embedding)
                    known_face_names.append(person_name)
                    print(f"  - Berhasil meng-encode {filename}")

                except Exception as e:
                    print(f"  - Error memproses {filename}: {e}")

# --- PENTING ---
# Hapus file embedding lama jika ada, karena formatnya sekarang berbeda
if os.path.exists(embeddings_file):
    os.remove(embeddings_file)
    print(f"File embedding lama '{embeddings_file}' dihapus.")

# Simpan embeddings dan nama ke dalam file
with open(embeddings_file, 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"\nProses pembuatan embeddings selesai!")
print(f"Total {len(known_face_encodings)} embeddings disimpan di {embeddings_file}")