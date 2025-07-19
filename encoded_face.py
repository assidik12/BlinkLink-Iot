import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Fungsi untuk Membuat Model Face Embedding (Placeholder) ---
# Dalam proyek nyata, Anda akan memuat model pre-trained seperti FaceNet.
# Contoh: model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
# Atau: model = tf.keras.models.load_model('path/to/your/facenet_model.h5')
# Model FaceNet biasanya membutuhkan input 160x160x3 dan menghasilkan embedding 128 dimensi.
def create_dummy_face_embedding_model():
    """
    Membuat model Keras sederhana sebagai placeholder untuk model face embedding.
    Ini BUKAN model FaceNet sungguhan, hanya untuk ilustrasi konsep.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(160, 160, 3)), # Ukuran input yang umum untuk model wajah
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=None) # Output adalah embedding 128 dimensi
    ])
    print("Model dummy face embedding dimuat.")
    return model

# Muat model face embedding
face_embedding_model = create_dummy_face_embedding_model()

# --- Konfigurasi ---
dataset_dir = "datasets"
embeddings_file = "face_embeddings_tf.pkl" # Nama file untuk menyimpan embeddings

known_face_encodings = []
known_face_names = []

print("Memulai proses pembuatan embeddings wajah dengan TensorFlow...")

# Iterasi melalui setiap subfolder (nama orang) di dalam folder 'datasets'
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)

    # Pastikan itu adalah direktori
    if os.path.isdir(person_dir):
        print(f"Memproses wajah untuk: {person_name}")
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, filename)
                try:
                    # Muat gambar menggunakan OpenCV
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  - Peringatan: Gagal memuat gambar {filename}. Dilewati.")
                        continue

                    # Konversi dari BGR ke RGB (TensorFlow/Keras biasanya butuh RGB)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Pra-pemrosesan: Ubah ukuran gambar agar sesuai dengan input model (160x160)
                    # Ini adalah ukuran umum untuk model FaceNet
                    resized_image = cv2.resize(image_rgb, (160, 160))
                    
                    # Normalisasi piksel ke rentang 0-1 (jika model membutuhkan)
                    normalized_image = resized_image.astype('float32') / 255.0

                    # Tambahkan dimensi batch (model membutuhkan input [batch_size, height, width, channels])
                    input_image = np.expand_dims(normalized_image, axis=0)

                    # Dapatkan embedding dari model
                    face_embedding = face_embedding_model.predict(input_image)[0] # Ambil embedding dari batch

                    known_face_encodings.append(face_embedding)
                    known_face_names.append(person_name)
                    print(f"  - Berhasil meng-encode {filename}")

                except Exception as e:
                    print(f"  - Error memproses {filename}: {e}. Dilewati.")

# Simpan embeddings dan nama ke dalam file menggunakan pickle
with open(embeddings_file, 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"\nProses pembuatan embeddings selesai!")
print(f"Total {len(known_face_encodings)} embeddings disimpan di {embeddings_file}")
print("Sekarang Anda bisa menjalankan script pengenalan wajah.")
