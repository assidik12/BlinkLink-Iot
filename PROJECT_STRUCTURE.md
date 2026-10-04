# 📁 BlinkLink-IoT Project Structure

Struktur project yang terorganisir untuk memudahkan maintenance dan navigasi.

## 📂 Direktori Utama

### `src/` - Source Code
Semua kode utama aplikasi berada di sini:
- `main.py` - Entry point utama aplikasi
- `vision_controller/` - Modul untuk kontrol visi dan deteksi wajah
- `preprocessing/` - Preprocessing data dan blink detection
- `iot_devices/` - Kode untuk komunikasi dengan ESP32 via MQTT
- `helper/` - Fungsi helper dan konfigurasi global
- `ui/` - User interface dengan pygame

### `models/` - Model & Data ML
File-file model machine learning dan embedding:
- `shape_predictor_68_face_landmarks.dat` - Dlib 68-point landmark predictor
- `face_embeddings_tf.pkl` - Face embedding database (pickle format)
- `face_embedding_model.h5` - Optional: TensorFlow face embedding model

### `data/` - Dataset
Dataset dan data training/testing:
- `datasets/` - Folder untuk dataset gambar wajah

### `scripts/` - Utility Scripts
Script untuk testing, benchmarking, dan utility lainnya:
- `benchmark_models.py` - Script untuk benchmark performa model
- `evaluation_report.md` - Laporan evaluasi sistem

### `docs/` - Documentation
Dokumentasi project dan hasil benchmark:
- `README.md` - Dokumentasi utama project
- `benchmark_results.json` - Hasil benchmark dalam format JSON
- `architecture.png` - Diagram arsitektur sistem

### `assets/` - Static Assets
Asset statis (logo, gambar, dll)

## 🚀 Cara Menjalankan

### Menjalankan Aplikasi Utama
```bash
# Dari root directory
python run.py

# Atau langsung dari src
python src/main.py
```

### Menjalankan Benchmark
```bash
python scripts/benchmark_models.py
```

## 📝 Catatan

- Path ke model files di-resolve secara otomatis dari `src/helper/config.py`
- Entry point `run.py` menangani path `src/` secara otomatis
- Semua path relatif dalam script menggunakan `os.path` untuk cross-platform compatibility
