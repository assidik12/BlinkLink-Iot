import cv2
import os
import time
import cv2.data
import numpy as np

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Haar Cascade untuk deteksi wajah berhasil dimuat.")
except Exception as e:
    print(f"Error: Gagal memuat Haar Cascade untuk wajah. Pastikan file XML ada di tempat yang benar.")
    print(f"Detail Error: {e}")
    exit()

# --- Konfigurasi ---
# Tentukan nama orang yang fotonya akan dikumpulkan
person_name = input("Masukkan nama orang (contoh: Andi, Budi): ").strip()
if not person_name:
    print("Nama tidak boleh kosong. Keluar.")
    exit()


# Buat direktori untuk menyimpan foto jika belum ada
dataset_dir = os.path.join("datasets", f"{person_name}")
os.makedirs(dataset_dir, exist_ok=True)

print(f"\nSiap mengumpulkan foto untuk: {person_name}")
print(f"Foto akan disimpan di: {dataset_dir}")
print("Pastikan wajahmu terlihat jelas di kamera.")
print("Tekan 's' untuk menyimpan foto, 'q' untuk keluar.")

# Inisialisasi webcam
cap = cv2.VideoCapture(0) # 0 untuk webcam default

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Pastikan kamera terhubung.")
    exit()

count = 0
last_capture_time = time.time()
capture_interval = 0.5 # Tangkap foto setiap 0.5 detik saat tombol 's' ditekan

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Balik frame secara horizontal (opsional, agar terlihat seperti cermin)
    frame = cv2.flip(frame, 1)

    # Konversi ke grayscale untuk deteksi wajah (lebih cepat)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haar Cascade
    # scaleFactor: Seberapa banyak ukuran gambar dikurangi pada setiap skala gambar.
    # minNeighbors: Berapa banyak tetangga (deteksi tumpang tindih) yang harus dimiliki setiap kandidat persegi panjang untuk mempertahankannya.
    # minSize: Ukuran objek minimum yang mungkin. Objek yang lebih kecil akan diabaikan.
    faces_detected_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Iterasi melalui wajah yang terdeteksi dan gambar kotak
    for (x, y, w, h) in faces_detected_in_frame:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Kotak hijau

    # Tampilkan jumlah foto yang sudah diambil
    cv2.putText(frame, f"Foto diambil: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Tekan 's' untuk simpan, 'q' untuk keluar", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan frame
    cv2.imshow('Kumpulkan Foto Wajah (Haar Cascade)', frame)

    key = cv2.waitKey(1) & 0xFF

    # Jika tombol 's' ditekan dan ada wajah terdeteksi
    if key == ord('s') and len(faces_detected_in_frame) > 0:
        # Batasi pengambilan foto agar tidak terlalu cepat
        if time.time() - last_capture_time > capture_interval:
            for (x, y, w, h) in faces_detected_in_frame:
                # Ambil hanya area wajah
                face_roi = frame[y:y+h, x:x+w]
                # Simpan foto wajah
                filename = os.path.join(dataset_dir, f"{person_name}_{count}.jpg")
                cv2.imwrite(filename, face_roi)
                print(f"Menyimpan {filename}")
                count += 1
                last_capture_time = time.time() # Reset timer

    # Jika tombol 'q' ditekan, keluar dari loop
    elif key == ord('q'):
        break

print(f"\nSelesai mengumpulkan {count} foto untuk {person_name}.")

# Lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
