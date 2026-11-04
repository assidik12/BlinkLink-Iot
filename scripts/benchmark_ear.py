"""
BlinkLink-IoT: EAR Performance Benchmark Suite
================================================
Komparasi Performa Komputasi Dlib 68-Landmarks vs MediaPipe Face Mesh
dalam Ekstraksi Fitur Eye Aspect Ratio (EAR).

Metrik yang diukur:
  - Latensi Inferensi per frame (ms)
  - Throughput (Frames Per Second / FPS)
  - Penggunaan Memori RAM (Peak RSS dalam MB) via psutil
  - Rata-rata Utilisasi CPU (%) via psutil
  - Nilai Eye Aspect Ratio (EAR)

Mendukung input:
  - Webcam live stream (e.g. --source 0)
  - File rekaman video (.mp4, .avi, dsb.)
  - Synthetic frame generator (jika kamera/video tidak tersedia)
"""

import os
import sys
import time
import bz2
import urllib.request
import argparse
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import psutil
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# KONSTANTA & INDEKS LANDMARK MATA
# ==============================================================================

# DLIB 68 Facial Landmarks (1-based pada paper, 0-based pada Python array):
# Left Eye : 36..41 -> [36, 37, 38, 39, 40, 41]
# Right Eye: 42..47 -> [42, 43, 44, 45, 46, 47]
DLIB_LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
DLIB_RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# MEDIAPIPE Face Mesh (468/478 points):
# Urutan 6 titik: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom2, p6_bottom1]
# Left Eye  (sudut pandang wajah): [33, 160, 158, 133, 153, 144]
# Right Eye (sudut pandang wajah): [362, 385, 387, 263, 373, 380]
MP_LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


# ==============================================================================
# 1. ALGORITMA PERHITUNGAN EYE ASPECT RATIO (EAR)
# ==============================================================================

def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Menghitung Eye Aspect Ratio (EAR) berdasarkan rumus Euclidean standar:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Args:
        eye_points: numpy array berukuran (6, 2) koordinat (x, y) 6 titik mata.
                    Index 0: p1 (corner 1)
                    Index 1: p2 (top 1)
                    Index 2: p3 (top 2)
                    Index 3: p4 (corner 2)
                    Index 4: p5 (bottom 2)
                    Index 5: p6 (bottom 1)
    
    Returns:
        float: Nilai Eye Aspect Ratio (EAR)
    """
    if eye_points is None or len(eye_points) < 6:
        return 0.0

    # Jarak vertikal 1: ||p2 - p6||
    dist_vert1 = np.linalg.norm(eye_points[1] - eye_points[5])
    # Jarak vertikal 2: ||p3 - p5||
    dist_vert2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Jarak horizontal : ||p1 - p4||
    dist_horiz = np.linalg.norm(eye_points[0] - eye_points[3])

    if dist_horiz <= 1e-6:
        return 0.0

    ear = (dist_vert1 + dist_vert2) / (2.0 * dist_horiz)
    return float(ear)


# ==============================================================================
# 2. HELPER UNDUH MODEL DLIB OTOMATIS
# ==============================================================================

def ensure_dlib_model(model_path: str, auto_download: bool = True) -> str:
    """
    Memverifikasi keberadaan model Dlib 68. Jika tidak ditemukan, opsi mengunduh
    dan mengekstrak secara otomatis dari dlib.net.
    """
    if os.path.exists(model_path):
        return model_path

    # Cari di folder models/ jika model_path relatif
    alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shape_predictor_68_face_landmarks.dat")
    if os.path.exists(alt_path):
        return alt_path

    if not auto_download:
        raise FileNotFoundError(
            f"Model Dlib tidak ditemukan di '{model_path}'.\n"
            f"Silakan unduh dari: {DLIB_MODEL_URL}\n"
            f"Ekstrak dan simpan ke: {model_path}"
        )

    print(f"\n[INFO] File model Dlib tidak ditemukan. Memulai proses unduh otomatis...")
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bz2_path = model_path + ".bz2"

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  -> Mengunduh: {percent}% [{count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DLIB_MODEL_URL, bz2_path, reporthook=_progress)
        print("\n  -> Selesai mengunduh. Mengekstrak file bz2...")

        with bz2.BZ2File(bz2_path, "rb") as source, open(model_path, "wb") as dest:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(chunk)

        if os.path.exists(bz2_path):
            os.remove(bz2_path)

        print(f"  -> Ekstraksi berhasil: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
        return model_path
    except Exception as e:
        if os.path.exists(bz2_path):
            os.remove(bz2_path)
        raise RuntimeError(f"Gagal mengunduh model Dlib otomatis: {e}\nSilakan unduh manual dari {DLIB_MODEL_URL}")


# ==============================================================================
# 3. RUNNER DLIB & MEDIAPIPE BENCHMARK
# ==============================================================================

class DlibEARBenchmark:
    """Wrapper runner untuk benchmark pipeline Dlib HOG + 68 Landmark."""

    def __init__(self, model_path: str):
        try:
            import dlib
        except ImportError as e:
            raise ImportError(
                "Pustaka 'dlib' belum terpasang. Pasang dengan 'pip install dlib-bin' "
                "atau 'pip install dlib'."
            ) from e

        print(f"[Dlib] Memuat model detektor HOG & shape predictor dari: {model_path}")
        self.dlib = dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[float], bool]:
        """
        Memproses 1 frame dengan Dlib: Deteksi Wajah -> Ekstraksi Landmark -> Hitung EAR.
        
        Returns:
            latency_ms (float): Waktu eksekusi pipeline dalam milidetik.
            ear_value (float or None): Nilai rata-rata EAR mata kiri dan kanan.
            face_detected (bool): True jika wajah terdeteksi.
        """
        start_time = time.perf_counter()

        # Konversi grayscale untuk Dlib
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Deteksi Wajah HOG
        faces = self.detector(gray, 0)
        face_detected = len(faces) > 0
        avg_ear = None

        # 2. Ekstraksi Landmark & EAR jika wajah ditemukan
        if face_detected:
            rect = faces[0]  # Ambil wajah utama
            shape = self.predictor(gray, rect)
            coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

            left_eye_pts = coords[DLIB_LEFT_EYE_INDICES]
            right_eye_pts = coords[DLIB_RIGHT_EYE_INDICES]

            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return latency_ms, avg_ear, face_detected


class MediaPipeEARBenchmark:
    """Wrapper runner untuk benchmark pipeline MediaPipe Face Mesh."""

    def __init__(self, refine_landmarks: bool = True):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("Pustaka 'mediapipe' belum terpasang.") from e

        print("[MediaPipe] Menginisialisasi Face Mesh model...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[float], bool]:
        """
        Memproses 1 frame dengan MediaPipe: Deteksi & Face Mesh -> Hitung EAR.
        
        Returns:
            latency_ms (float): Waktu eksekusi pipeline dalam milidetik.
            ear_value (float or None): Nilai rata-rata EAR mata kiri dan kanan.
            face_detected (bool): True jika wajah terdeteksi.
        """
        start_time = time.perf_counter()

        # Konversi BGR ke RGB (rekomendasi MediaPipe)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # 1. Eksekusi Face Mesh
        results = self.face_mesh.process(rgb)
        face_detected = bool(results.multi_face_landmarks)
        avg_ear = None

        # 2. Ekstraksi Landmark & EAR jika wajah ditemukan
        if face_detected:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Konversi normalized coords ke pixel coords (numpy array)
            # Karena rumus rasio EAR invarian terhadap skala, koordinat piksel atau float bernilai proporsional
            all_pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

            left_eye_pts = all_pts[MP_LEFT_EYE_INDICES]
            right_eye_pts = all_pts[MP_RIGHT_EYE_INDICES]

            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return latency_ms, avg_ear, face_detected

    def close(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# ==============================================================================
# 4. MONITOR SUMBER DAYA SISTEM (PSUTIL)
# ==============================================================================

class SystemResourceMonitor:
    """Memantau penggunaan memori RAM (MB) dan CPU (%) untuk proses ini."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # Panggilan pertama cpu_percent menginisialisasi titik referensi
        self.process.cpu_percent(interval=None)

    def get_stats(self) -> Tuple[float, float]:
        """
        Returns:
            ram_mb (float): Resident Set Size (RSS) memori dalam MB.
            cpu_percent (float): Persentase penggunaan CPU sejak pembacaan terakhir.
        """
        ram_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
        cpu_pct = self.process.cpu_percent(interval=None)
        return round(ram_mb, 2), round(cpu_pct, 2)


# ==============================================================================
# 5. GENERATOR FRAME SINTETIS (FALLBACK JIKA TANPA KAMERA/VIDEO)
# ==============================================================================

def create_synthetic_face_frame(width: int = 640, height: int = 480, frame_idx: int = 0) -> np.ndarray:
    """
    Membuat frame wajah sintetis dengan variasi bukaan mata berkala
    agar pengujian dapat berjalan otomatis tanpa file video maupun kamera fisik.
    """
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    center_x, center_y = width // 2, height // 2

    # Gambar kontur wajah oval
    cv2.ellipse(frame, (center_x, center_y), (140, 190), 0, 0, 360, (190, 205, 235), -1)
    cv2.ellipse(frame, (center_x, center_y), (140, 190), 0, 0, 360, (70, 70, 70), 3)

    # Variasi bukaan mata berosilasi (simulasi kedipan sinusoidal)
    cycle = (frame_idx % 30) / 30.0
    eye_open_scale = 1.0 if cycle > 0.3 else max(0.1, cycle / 0.3)
    eye_h = int(18 * eye_open_scale)

    # Mata Kiri & Kanan
    left_eye_center = (center_x - 55, center_y - 30)
    right_eye_center = (center_x + 55, center_y - 30)
    
    cv2.ellipse(frame, left_eye_center, (28, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, left_eye_center, (28, max(2, eye_h)), 0, 0, 360, (30, 30, 30), 2)
    cv2.circle(frame, left_eye_center, min(8, max(2, eye_h)), (60, 40, 20), -1)

    cv2.ellipse(frame, right_eye_center, (28, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, right_eye_center, (28, max(2, eye_h)), 0, 0, 360, (30, 30, 30), 2)
    cv2.circle(frame, right_eye_center, min(8, max(2, eye_h)), (60, 40, 20), -1)

    # Hidung & Mulut
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 35), (100, 100, 100), 3)
    cv2.ellipse(frame, (center_x, center_y + 80), (45, 12), 0, 0, 180, (50, 50, 180), 3)

    return frame


# ==============================================================================
# 6. ENGINE PENGUJIAN (INTERLEAVED VS SEQUENTIAL)
# ==============================================================================

def open_capture_source(source_arg: str):
    """Membuka cv2.VideoCapture atau mendeteksi mode sintetis."""
    if source_arg.lower() in ("synthetic", "mock", "test"):
        return None, True
    
    # Periksa apakah argumen berupa angka (webcam index)
    if source_arg.isdigit():
        cap = cv2.VideoCapture(int(source_arg))
    else:
        if not os.path.exists(source_arg):
            print(f"[PERINGATAN] File video '{source_arg}' tidak ditemukan. Beralih ke generator sintetis.")
            return None, True
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print(f"[PERINGATAN] Gagal membuka sumber '{source_arg}'. Beralih ke generator sintetis.")
        return None, True
    return cap, False


def run_warmup(dlib_bench: DlibEARBenchmark, mp_bench: MediaPipeEARBenchmark, warmup_frames: int):
    """Menjalankan beberapa iterasi pemanasan agar JIT/cache stabil sebelum pengukuran."""
    print(f"\n[WARMUP] Menjalankan {warmup_frames} frame pemanasan...")
    sample_frame = create_synthetic_face_frame(640, 480, 0)
    for _ in range(warmup_frames):
        dlib_bench.process_frame(sample_frame)
        mp_bench.process_frame(sample_frame)
    print("[WARMUP] Selesai.\n")


def run_interleaved_benchmark(
    source_arg: str,
    dlib_bench: DlibEARBenchmark,
    mp_bench: MediaPipeEARBenchmark,
    total_frames: int,
    warmup_frames: int,
    monitor: SystemResourceMonitor
) -> List[Dict]:
    """
    Mode Head-to-Head Adil (Opsi B):
    Memproses frame yang persis sama secara bergantian pada setiap iterasi frame.
    Menghilangkan bias variasi pencahayaan atau pose wajah antara kedua framework.
    """
    run_warmup(dlib_bench, mp_bench, warmup_frames)
    cap, is_synthetic = open_capture_source(source_arg)

    records = []
    print(f"[RUNNING] Memulai Benchmark Interleaved (Target: {total_frames} frame)...")
    progress_step = max(1, total_frames // 10)

    try:
        for f_idx in range(total_frames):
            if is_synthetic:
                frame = create_synthetic_face_frame(640, 480, f_idx)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"  -> Sumber video selesai pada frame ke-{f_idx}.")
                    break

            # 1. Eksekusi Dlib pada salinan frame
            dlib_latency, dlib_ear, dlib_detected = dlib_bench.process_frame(frame.copy())
            ram_mb, cpu_pct = monitor.get_stats()
            records.append({
                "framework": "Dlib 68",
                "frame_id": f_idx + 1,
                "latency_ms": dlib_latency,
                "fps_instant": 1000.0 / dlib_latency if dlib_latency > 0 else 0,
                "ear_value": dlib_ear if dlib_ear is not None else np.nan,
                "face_detected": int(dlib_detected),
                "ram_mb": ram_mb,
                "cpu_percent": cpu_pct
            })

            # 2. Eksekusi MediaPipe pada salinan frame
            mp_latency, mp_ear, mp_detected = mp_bench.process_frame(frame.copy())
            ram_mb, cpu_pct = monitor.get_stats()
            records.append({
                "framework": "MediaPipe FaceMesh",
                "frame_id": f_idx + 1,
                "latency_ms": mp_latency,
                "fps_instant": 1000.0 / mp_latency if mp_latency > 0 else 0,
                "ear_value": mp_ear if mp_ear is not None else np.nan,
                "face_detected": int(mp_detected),
                "ram_mb": ram_mb,
                "cpu_percent": cpu_pct
            })

            if (f_idx + 1) % progress_step == 0 or (f_idx + 1) == total_frames:
                sys.stdout.write(f"\r  -> Progres: {f_idx + 1}/{total_frames} frame diproses...")
                sys.stdout.flush()

        print("\n  -> Pengujian interleaved selesai.")
    finally:
        if cap is not None:
            cap.release()

    return records


def run_sequential_benchmark(
    source_arg: str,
    dlib_bench: DlibEARBenchmark,
    mp_bench: MediaPipeEARBenchmark,
    total_frames: int,
    warmup_frames: int,
    monitor: SystemResourceMonitor
) -> List[Dict]:
    """
    Mode Sequential (Opsi A):
    Menjalankan N frame berturut-turut untuk Dlib, lalu me-reset video/sumber
    dan menjalankan N frame untuk MediaPipe.
    """
    run_warmup(dlib_bench, mp_bench, warmup_frames)
    records = []

    # --- PASS 1: DLIB ---
    print(f"\n[RUNNING PASS 1/2] Menjalankan Dlib ({total_frames} frame)...")
    cap, is_synthetic = open_capture_source(source_arg)
    try:
        for f_idx in range(total_frames):
            if is_synthetic:
                frame = create_synthetic_face_frame(640, 480, f_idx)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            
            dlib_latency, dlib_ear, dlib_detected = dlib_bench.process_frame(frame)
            ram_mb, cpu_pct = monitor.get_stats()
            records.append({
                "framework": "Dlib 68",
                "frame_id": f_idx + 1,
                "latency_ms": dlib_latency,
                "fps_instant": 1000.0 / dlib_latency if dlib_latency > 0 else 0,
                "ear_value": dlib_ear if dlib_ear is not None else np.nan,
                "face_detected": int(dlib_detected),
                "ram_mb": ram_mb,
                "cpu_percent": cpu_pct
            })
    finally:
        if cap is not None:
            cap.release()

    # --- PASS 2: MEDIAPIPE ---
    print(f"[RUNNING PASS 2/2] Menjalankan MediaPipe ({total_frames} frame)...")
    cap, is_synthetic = open_capture_source(source_arg)
    try:
        for f_idx in range(total_frames):
            if is_synthetic:
                frame = create_synthetic_face_frame(640, 480, f_idx)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            
            mp_latency, mp_ear, mp_detected = mp_bench.process_frame(frame)
            ram_mb, cpu_pct = monitor.get_stats()
            records.append({
                "framework": "MediaPipe FaceMesh",
                "frame_id": f_idx + 1,
                "latency_ms": mp_latency,
                "fps_instant": 1000.0 / mp_latency if mp_latency > 0 else 0,
                "ear_value": mp_ear if mp_ear is not None else np.nan,
                "face_detected": int(mp_detected),
                "ram_mb": ram_mb,
                "cpu_percent": cpu_pct
            })
    finally:
        if cap is not None:
            cap.release()

    print("  -> Pengujian sequential selesai.")
    return records


# ==============================================================================
# 7. FORMAT TABEL ASCII LAPORAN
# ==============================================================================

def print_ascii_summary_table(df: pd.DataFrame):
    """Menghitung dan mencetak tabel perbandingan performa dalam format ASCII yang rapi."""
    grouped = df.groupby("framework")
    
    summary_data = []
    for name, group in grouped:
        lat = group["latency_ms"]
        mean_lat = lat.mean()
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        ear_valid = group["ear_value"].dropna()

        summary_data.append({
            "Framework": name,
            "Frames": len(group),
            "Face Det (%)": f"{(group['face_detected'].mean() * 100):.1f}%",
            "Avg Latency (ms)": f"{mean_lat:.2f}",
            "Min (ms)": f"{lat.min():.2f}",
            "Max (ms)": f"{lat.max():.2f}",
            "P95 (ms)": f"{lat.quantile(0.95):.2f}",
            "Throughput (FPS)": f"{fps:.1f}",
            "Peak RAM (MB)": f"{group['ram_mb'].max():.1f}",
            "Avg CPU (%)": f"{group['cpu_percent'].mean():.1f}%",
            "Avg EAR": f"{ear_valid.mean():.4f}" if len(ear_valid) > 0 else "N/A"
        })

    summary_df = pd.DataFrame(summary_data)

    print("\n" + "=" * 115)
    print("                 HASIL BENCHMARK KOMPARASI: DLIB 68 vs MEDIAPIPE FACE MESH")
    print("=" * 115)
    
    headers = list(summary_df.columns)
    col_widths = [max(len(h), max(len(str(val)) for val in summary_df[h])) + 2 for h in headers]

    # Format header
    header_str = "| " + " | ".join(h.center(w) for h, w in zip(headers, col_widths)) + " |"
    divider = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print(divider)
    print(header_str)
    print(divider)

    for _, row in summary_df.iterrows():
        row_str = "| " + " | ".join(str(row[h]).center(w) for h, w in zip(headers, col_widths)) + " |"
        print(row_str)

    print(divider)
    print("=" * 115)


# ==============================================================================
# 8. VISUALISASI GRAFIK KOMPARASI (MATPLOTLIB)
# ==============================================================================

def generate_comparison_plots(df: pd.DataFrame, output_image_path: str):
    """
    Membuat grafik analisis komparasi multi-panel dan menyimpannya ke file PNG.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    
    dlib_df = df[df["framework"] == "Dlib 68"].sort_values("frame_id")
    mp_df = df[df["framework"] == "MediaPipe FaceMesh"].sort_values("frame_id")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={"hspace": 0.35})
    fig.suptitle("Komparasi Performa Komputasi: Dlib 68 vs MediaPipe Face Mesh", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Latensi Inferensi per Frame
    ax1 = axes[0]
    ax1.plot(dlib_df["frame_id"], dlib_df["latency_ms"], label="Dlib 68 (ms)", color="#d9534f", alpha=0.6, linewidth=1.2)
    ax1.plot(mp_df["frame_id"], mp_df["latency_ms"], label="MediaPipe Face Mesh (ms)", color="#0275d8", alpha=0.75, linewidth=1.2)
    
    # Tambah garis rata-rata (dashed line)
    dlib_mean = dlib_df["latency_ms"].mean()
    mp_mean = mp_df["latency_ms"].mean()
    ax1.axhline(dlib_mean, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Dlib Rata-rata: {dlib_mean:.1f} ms")
    ax1.axhline(mp_mean, color="#0275d8", linestyle="--", linewidth=1.5, label=f"MediaPipe Rata-rata: {mp_mean:.1f} ms")

    ax1.set_title("1. Latensi Inferensi per Frame (Waktu Proses)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("Latensi (ms)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Nilai Eye Aspect Ratio (EAR) per Frame
    ax2 = axes[1]
    ax2.plot(dlib_df["frame_id"], dlib_df["ear_value"], label="Dlib EAR", color="#d9534f", linewidth=1.5, alpha=0.8)
    ax2.plot(mp_df["frame_id"], mp_df["ear_value"], label="MediaPipe EAR", color="#5cb85c", linewidth=1.5, alpha=0.8)
    ax2.set_title("2. Nilai Eye Aspect Ratio (EAR) pada Frame Pengujian", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frame ID")
    ax2.set_ylabel("Nilai EAR")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Ringkasan Metrik Kunci (Bar Chart)
    ax3 = axes[2]
    metrics = ["Avg Latency (ms)", "Throughput (FPS)", "Avg CPU (%)", "Peak RAM (MB)"]
    dlib_vals = [
        dlib_mean,
        1000.0 / dlib_mean if dlib_mean > 0 else 0,
        dlib_df["cpu_percent"].mean(),
        dlib_df["ram_mb"].max()
    ]
    mp_vals = [
        mp_mean,
        1000.0 / mp_mean if mp_mean > 0 else 0,
        mp_df["cpu_percent"].mean(),
        mp_df["ram_mb"].max()
    ]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax3.bar(x - width/2, dlib_vals, width, label="Dlib 68", color="#d9534f", alpha=0.85)
    rects2 = ax3.bar(x + width/2, mp_vals, width, label="MediaPipe Face Mesh", color="#0275d8", alpha=0.85)

    ax3.set_title("3. Ringkasan Metrik Kunci Komparatif", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f"{height:.1f}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.96, hspace=0.35)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OUTPUT] Grafik komparasi berhasil disimpan ke: {os.path.abspath(output_image_path)}")


# ==============================================================================
# 9. MAIN CLI ARGUMENT PARSER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Komparasi Performa Komputasi: Dlib 68 vs MediaPipe Face Mesh untuk EAR."
    )
    
    # Input source
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Sumber input: '0' untuk webcam, path file video (.mp4/.avi), atau 'synthetic' untuk pengujian otomatis (default: synthetic)."
    )
    
    # Jumlah frame & pemanasan
    parser.add_argument(
        "--frames",
        type=int,
        default=150,
        help="Jumlah total frame yang diuji (default: 150 frame)."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Jumlah frame pemanasan (warmup) sebelum pencatatan metrik (default: 10)."
    )
    
    # Mode benchmark
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interleaved", "sequential"],
        default="interleaved",
        help="Metode pengujian: 'interleaved' (frame sama bergantian, head-to-head adil) atau 'sequential' (Dlib N frame lalu MediaPipe N frame)."
    )
    
    # Path model Dlib
    parser.add_argument(
        "--dlib-model",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shape_predictor_68_face_landmarks.dat"),
        help="Path ke file shape_predictor_68_face_landmarks.dat (default: models/shape_predictor_68_face_landmarks.dat)."
    )
    
    # Output file
    parser.add_argument(
        "--csv-output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results.csv"),
        help="Path file output CSV untuk metrik per-frame (default: benchmark_results.csv)."
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_comparison.png"),
        help="Path file output gambar grafik perbandingan (default: benchmark_comparison.png)."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Nonaktifkan pembuatan grafik matplotlib."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Nonaktifkan unduh otomatis model Dlib jika file belum ada."
    )

    args = parser.parse_args()

    print("=" * 75)
    print("   BLINKLINK-IOT: BENCHMARK KOMPARASI DLIB 68 VS MEDIAPIPE FACE MESH")
    print("=" * 75)
    print(f" Sumber Input    : {args.source}")
    print(f" Mode Pengujian  : {args.mode.upper()}")
    print(f" Total Frame     : {args.frames}")
    print(f" Warmup Frames   : {args.warmup}")
    print(f" Model Dlib Path : {args.dlib-model if hasattr(args, 'dlib-model') else args.dlib_model}")
    print(f" CSV Output      : {args.csv_output}")
    print(f" Plot Output     : {args.plot_output if not args.no_plot else 'DISABLED'}")
    print("=" * 75)

    # 1. Pastikan model Dlib tersedia
    dlib_path = ensure_dlib_model(args.dlib_model, auto_download=not args.no_download)

    # 2. Inisialisasi runner
    monitor = SystemResourceMonitor()
    dlib_bench = DlibEARBenchmark(dlib_path)
    mp_bench = MediaPipeEARBenchmark()

    # 3. Jalankan benchmark
    if args.mode == "interleaved":
        records = run_interleaved_benchmark(
            source_arg=args.source,
            dlib_bench=dlib_bench,
            mp_bench=mp_bench,
            total_frames=args.frames,
            warmup_frames=args.warmup,
            monitor=monitor
        )
    else:
        records = run_sequential_benchmark(
            source_arg=args.source,
            dlib_bench=dlib_bench,
            mp_bench=mp_bench,
            total_frames=args.frames,
            warmup_frames=args.warmup,
            monitor=monitor
        )

    mp_bench.close()

    if not records:
        print("[ERROR] Tidak ada data metrik yang terkumpul. Periksa kembali sumber input video/kamera.")
        sys.exit(1)

    df_results = pd.DataFrame(records)

    # 4. Tampilkan Tabel Ringkasan ASCII di Konsol
    print_ascii_summary_table(df_results)

    # 5. Ekspor ke file CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_output)), exist_ok=True)
    df_results.to_csv(args.csv_output, index=False)
    print(f"\n[OUTPUT] Log metrik per-frame berhasil diekspor ke: {os.path.abspath(args.csv_output)}")

    # 6. Pembuatan Grafik Plotting (Matplotlib)
    if not args.no_plot:
        generate_comparison_plots(df_results, args.plot_output)

    print("\n[SELESAI] Seluruh pengujian benchmark performa EAR berhasil dilaksanakan.")


if __name__ == "__main__":
    main()
