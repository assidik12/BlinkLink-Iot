import cv2
import os
import glob

# Ganti dengan path yang sesuai jika berbeda
INPUT_DIR = 'unnormalize_face'
OUTPUT_DIR = 'datasets'

# Pastikan file haarcascade_frontalface_default.xml ada di environment Anda
# Biasanya bawaan dari instalasi opencv-python, tapi pastikan pathnya benar
# Bisa juga didownload dari: https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error: Gagal memuat Haar cascade dari {cascade_path}")
    print("Pastikan OpenCV terinstal dengan benar dan file cascade tersedia.")
    exit()

def process_images():
    print(f"Memulai pemrosesan gambar dari '{INPUT_DIR}'...")
    
    # Memeriksa apakah folder input ada
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Folder input '{INPUT_DIR}' tidak ditemukan.")
        return

    # Membuat folder output utama jika belum ada
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Membuat folder utama dataset: '{OUTPUT_DIR}'")

    # Mendapatkan daftar folder di dalam INPUT_DIR (masing-masing folder mewakili satu orang)
    person_folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    
    total_processed = 0
    total_saved = 0
    
    for person_name in person_folders:
        person_input_dir = os.path.join(INPUT_DIR, person_name)
        person_output_dir = os.path.join(OUTPUT_DIR, person_name)
        
        # Membuat folder untuk orang tersebut di direktori output jika belum ada
        if not os.path.exists(person_output_dir):
            os.makedirs(person_output_dir)
            
        print(f"\nMemproses untuk: {person_name}")
        
        # Mengambil semua file gambar (jpg, jpeg, png) di folder orang tersebut
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_files.extend(glob.glob(os.path.join(person_input_dir, ext)))
            
        if not image_files:
            print(f"  Tidak ada gambar ditemukan di {person_input_dir}")
            continue
            
        count = 1
        for img_path in image_files:
            total_processed += 1
            filename = os.path.basename(img_path)
            
            # Membaca gambar
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Gagal membaca gambar: {filename}")
                continue
                
            # Mengkonversi ke grayscale untuk deteksi yang lebih baik dan cepat
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Mendeteksi wajah menggunakan parameter yang diminta
            # scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            faces_detected_in_frame = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            # Jika wajah terdeteksi
            if len(faces_detected_in_frame) > 0:
                # Kita ambil wajah pertama yang terdeteksi (asumsi satu gambar satu wajah utama)
                (x, y, w, h) = faces_detected_in_frame[0]
                
                # Memotong (crop) area wajah dari gambar asli (berwarna)
                face_roi = img[y:y+h, x:x+w]
                
                # Mengubah ukuran (resize) wajah yang dipotong menjadi 100x100
                face_resized = cv2.resize(face_roi, (100, 100))
                
                # Menentukan nama file output
                output_filename = f"{person_name}_{count}.jpg"
                output_filepath = os.path.join(person_output_dir, output_filename)
                
                # Menyimpan gambar wajah yang sudah diproses
                cv2.imwrite(output_filepath, face_resized)
                # print(f"  Berhasil menyimpan: {output_filepath}")
                
                count += 1
                total_saved += 1
            else:
                print(f"  Wajah tidak terdeteksi pada gambar: {filename}")

if __name__ == "__main__":
    process_images()
    print("\n--- Pemrosesan Selesai ---")
    print(f"Total gambar yang diperiksa : ") # Variabel total_processed akan diprint nanti
    # Di script asli saya tidak return total_processed, jadi saya print manual di fungsinya saja