"""
BlinkLink-IoT: EAR Performance & Accuracy Benchmark Suite
=========================================================
Komparasi Performa Komputasi & Evaluasi Akurasi Dlib 68 vs MediaPipe Face Mesh
dalam Ekstraksi Fitur Eye Aspect Ratio (EAR) dan Deteksi Kedipan Mata.

Fitur & Metrik:
  1. Performa Komputasi:
     - Latensi Inferensi per frame (ms) & P95 (ms)
     - Throughput (Frames Per Second / FPS)
     - Penggunaan Memori RAM (Peak RSS MB) & Utilisasi CPU (%) via psutil
  2. Evaluasi Akurasi vs Ground Truth (Bab 2.3 Skenario Kacamata & Cahaya):
     - Tingkat Keberhasilan Ekstraksi Landmark / Deteksi Wajah (%)
     - Klasifikasi Kondisi Mata: Terbuka (0) vs Tertutup (1)
     - Confusion Matrix: True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN)
     - Metrik Klasifikasi: Accuracy, Precision, Recall, F1-Score
     - Deteksi Jumlah Kedipan Aktual (GT Blinks) vs Terdeteksi (Pred Blinks)
  3. Format Input & Skenario:
     - Mendukung webcam live (`--source 0`), video rekaman (`.mp4`), atau generator sintetis (`synthetic`)
     - Metadata skenario: `--scenario-lighting` dan `--scenario-glasses`
     - Ground Truth via file CSV atau rentang frame tertutup `--gt-closed-ranges`
     - Ekspor CSV per-frame dan CSV rekapitulasi skenario `--summary-csv`
"""

import os
import sys
import time
import bz2
import datetime
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
DLIB_LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
DLIB_RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# MEDIAPIPE Face Mesh (468/478 points):
# Urutan 6 titik: [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom2, p6_bottom1]
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
        start_time = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        face_detected = len(faces) > 0
        avg_ear = None

        if face_detected:
            rect = faces[0]
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
        start_time = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self.face_mesh.process(rgb)
        face_detected = bool(results.multi_face_landmarks)
        avg_ear = None

        if face_detected:
            landmarks = results.multi_face_landmarks[0].landmark
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
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)

    def get_stats(self) -> Tuple[float, float]:
        ram_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
        cpu_pct = self.process.cpu_percent(interval=None)
        return round(ram_mb, 2), round(cpu_pct, 2)


# ==============================================================================
# 5. GENERATOR FRAME SINTETIS (FALLBACK JIKA TANPA KAMERA/VIDEO)
# ==============================================================================

def create_synthetic_face_frame(width: int = 640, height: int = 480, frame_idx: int = 0) -> np.ndarray:
    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    center_x, center_y = width // 2, height // 2

    # Gambar kontur wajah oval
    cv2.ellipse(frame, (center_x, center_y), (140, 190), 0, 0, 360, (190, 205, 235), -1)
    cv2.ellipse(frame, (center_x, center_y), (140, 190), 0, 0, 360, (70, 70, 70), 3)

    # Variasi bukaan mata berkala (kedip pada frame 1-9, 31-39, dst dalam siklus 30 frame)
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
# 6. ENGINE GROUND TRUTH & EVALUASI AKURASI
# ==============================================================================

def parse_ground_truth(
    total_frames: int,
    gt_file: Optional[str] = None,
    gt_ranges: Optional[str] = None
) -> Tuple[Optional[np.ndarray], int, List[Tuple[int, int]]]:
    """
    Memuat atau mengonversi anotasi Ground Truth ke dalam array biner per-frame (1-based: 1=tertutup, 0=terbuka).
    Mendukung:
      - String rentang frame, misal: "45-52,120-128,210-219"
      - File CSV rentang: kolom 'start_frame,end_frame'
      - File CSV per-frame: kolom 'frame_id,eye_closed' (atau sejenisnya)
    
    Returns:
        gt_array (np.ndarray of shape (total_frames,) or None): Label biner per frame (index 0 adalah frame 1).
        gt_blinks (int): Jumlah event kedipan pada Ground Truth.
        intervals (List[Tuple[int, int]]): Rentang frame tertutup (start_frame, end_frame).
    """
    if not gt_file and not gt_ranges:
        return None, 0, []

    gt_array = np.zeros(total_frames, dtype=int)
    intervals: List[Tuple[int, int]] = []

    # 1. Parsing dari parameter string rentang jika tersedia
    if gt_ranges:
        print(f"[GROUND TRUTH] Mem-parsing rentang frame tertutup dari CLI: '{gt_ranges}'")
        parts = [p.strip() for p in gt_ranges.split(",") if p.strip()]
        for p in parts:
            if "-" in p:
                start_s, end_s = p.split("-", 1)
                try:
                    s_idx = max(1, int(start_s.strip()))
                    e_idx = min(total_frames, int(end_s.strip()))
                    if s_idx <= e_idx:
                        intervals.append((s_idx, e_idx))
                        gt_array[s_idx - 1:e_idx] = 1
                except ValueError:
                    print(f"[PERINGATAN] Format rentang '{p}' tidak valid.")
            else:
                try:
                    single = int(p.strip())
                    if 1 <= single <= total_frames:
                        intervals.append((single, single))
                        gt_array[single - 1] = 1
                except ValueError:
                    pass

    # 2. Parsing dari file CSV jika tersedia
    elif gt_file:
        if not os.path.exists(gt_file):
            print(f"[PERINGATAN] File Ground Truth '{gt_file}' tidak ditemukan.")
            return None, 0, []
        
        print(f"[GROUND TRUTH] Membaca file anotasi: {gt_file}")
        df_gt = pd.read_csv(gt_file)
        cols_lower = [c.lower() for c in df_gt.columns]

        # Skenario A: Format Rentang (start_frame, end_frame)
        if any("start" in c for c in cols_lower) and any("end" in c for c in cols_lower):
            start_col = [c for c in df_gt.columns if "start" in c.lower()][0]
            end_col = [c for c in df_gt.columns if "end" in c.lower()][0]
            for _, row in df_gt.iterrows():
                try:
                    s_idx = max(1, int(row[start_col]))
                    e_idx = min(total_frames, int(row[end_col]))
                    if s_idx <= e_idx:
                        intervals.append((s_idx, e_idx))
                        gt_array[s_idx - 1:e_idx] = 1
                except (ValueError, TypeError):
                    continue

        # Skenario B: Format Per-Frame (frame_id / frame, eye_closed / state / closed)
        else:
            frame_col = None
            label_col = None
            for c in df_gt.columns:
                c_low = c.lower()
                if "frame" in c_low and frame_col is None:
                    frame_col = c
                if any(w in c_low for w in ["closed", "state", "label", "blink", "gt"]) and label_col is None:
                    label_col = c

            if frame_col and label_col:
                for _, row in df_gt.iterrows():
                    try:
                        f_idx = int(row[frame_col])
                        val = str(row[label_col]).strip().lower()
                        is_closed = 1 if val in ("1", "true", "closed", "yes") else 0
                        if 1 <= f_idx <= total_frames:
                            gt_array[f_idx - 1] = is_closed
                    except (ValueError, TypeError):
                        continue

                # Rekonstruksi interval dari array per-frame
                in_closed = False
                cur_start = 0
                for i in range(total_frames):
                    if gt_array[i] == 1 and not in_closed:
                        in_closed = True
                        cur_start = i + 1
                    elif gt_array[i] == 0 and in_closed:
                        in_closed = False
                        intervals.append((cur_start, i))
                if in_closed:
                    intervals.append((cur_start, total_frames))

    gt_blinks = len(intervals)
    print(f"[GROUND TRUTH] Ditemukan {len(intervals)} interval kedipan (total frame tertutup: {int(np.sum(gt_array))}).")
    return gt_array, gt_blinks, intervals


def evaluate_framework_results(
    df_framework: pd.DataFrame,
    threshold: float,
    min_blink_frames: int,
    gt_array: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Mengklasifikasikan kondisi mata per frame, menghitung kedipan (blink event),
    dan jika Ground Truth tersedia, menghitung Confusion Matrix (TP, TN, FP, FN)
    serta Accuracy, Precision, Recall, dan F1-Score.
    """
    df_eval = df_framework.copy().sort_values("frame_id").reset_index(drop=True)
    total_frames = len(df_eval)

    pred_closed_list = []
    blink_counts_list = []
    eval_outcomes_list = []

    blink_counter = 0
    consecutive_closed = 0

    for idx, row in df_eval.iterrows():
        f_idx = int(row["frame_id"])
        face_det = int(row["face_detected"])
        ear_val = row["ear_value"]

        # Klasifikasi: Jika wajah gagal terdeteksi atau EAR NaN, dianggap tidak tertutup (0)
        if face_det == 0 or pd.isna(ear_val):
            is_closed = 0
        else:
            is_closed = 1 if float(ear_val) <= threshold else 0

        pred_closed_list.append(is_closed)

        # Logika Deteksi Event Kedipan (transisi state)
        if is_closed == 1:
            consecutive_closed += 1
        else:
            if consecutive_closed >= min_blink_frames:
                blink_counter += 1
            consecutive_closed = 0

        blink_counts_list.append(blink_counter)

        # Evaluasi terhadap Ground Truth
        if gt_array is not None and (f_idx - 1) < len(gt_array):
            gt_val = int(gt_array[f_idx - 1])
            if gt_val == 1 and is_closed == 1:
                outcome = "TP"
            elif gt_val == 0 and is_closed == 0:
                outcome = "TN"
            elif gt_val == 0 and is_closed == 1:
                outcome = "FP"
            else:
                outcome = "FN"
            eval_outcomes_list.append(outcome)
        else:
            eval_outcomes_list.append("N/A")

    # Tangani jika frame video berakhir saat mata masih tertutup
    if consecutive_closed >= min_blink_frames:
        blink_counter += 1
        blink_counts_list[-1] = blink_counter

    df_eval["ear_threshold"] = threshold
    df_eval["pred_closed"] = pred_closed_list
    df_eval["blink_count_cum"] = blink_counts_list
    df_eval["eval_outcome"] = eval_outcomes_list

    if gt_array is not None:
        gt_slice = [int(gt_array[int(r["frame_id"]) - 1]) if (int(r["frame_id"]) - 1) < len(gt_array) else 0 for _, r in df_eval.iterrows()]
        df_eval["gt_closed"] = gt_slice
    else:
        df_eval["gt_closed"] = np.nan

    # Ringkasan Metrik
    framework_name = df_eval["framework"].iloc[0]
    face_det_pct = (df_eval["face_detected"].mean()) * 100.0
    mean_lat = df_eval["latency_ms"].mean()
    fps_val = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    summary = {
        "framework": framework_name,
        "total_frames": total_frames,
        "face_det_pct": face_det_pct,
        "avg_latency_ms": mean_lat,
        "p95_latency_ms": df_eval["latency_ms"].quantile(0.95),
        "throughput_fps": fps_val,
        "peak_ram_mb": df_eval["ram_mb"].max(),
        "avg_cpu_pct": df_eval["cpu_percent"].mean(),
        "avg_ear": df_eval["ear_value"].dropna().mean() if len(df_eval["ear_value"].dropna()) > 0 else np.nan,
        "ear_threshold": threshold,
        "pred_blinks": blink_counter,
    }

    if gt_array is not None:
        tp = int((df_eval["eval_outcome"] == "TP").sum())
        tn = int((df_eval["eval_outcome"] == "TN").sum())
        fp = int((df_eval["eval_outcome"] == "FP").sum())
        fn = int((df_eval["eval_outcome"] == "FN").sum())

        acc = ((tp + tn) / total_frames * 100.0) if total_frames > 0 else 0.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        summary.update({
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy_pct": acc,
            "precision_pct": prec,
            "recall_pct": rec,
            "f1_score_pct": f1,
        })
    else:
        summary.update({
            "tp": None, "tn": None, "fp": None, "fn": None,
            "accuracy_pct": None, "precision_pct": None, "recall_pct": None, "f1_score_pct": None,
        })

    return df_eval, summary


# ==============================================================================
# 7. ENGINE PENGUJIAN (INTERLEAVED VS SEQUENTIAL)
# ==============================================================================

def open_capture_source(source_arg: str):
    if source_arg.lower() in ("synthetic", "mock", "test"):
        return None, True
    
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
    run_warmup(dlib_bench, mp_bench, warmup_frames)
    records = []

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
# 8. FORMAT TABEL ASCII LAPORAN
# ==============================================================================

def print_ascii_summary_tables(summaries: List[Dict], has_gt: bool, gt_blinks: int = 0):
    """Menampilkan tabel hasil komparasi performa komputasi dan evaluasi akurasi."""
    # --- TABEL 1: Performa Komputasi & Ekstraksi Landmark ---
    comp_rows = []
    for s in summaries:
        comp_rows.append({
            "Framework": s["framework"],
            "Frames": s["total_frames"],
            "Face Det (%)": f"{s['face_det_pct']:.1f}%",
            "Avg Lat (ms)": f"{s['avg_latency_ms']:.2f}",
            "P95 (ms)": f"{s['p95_latency_ms']:.2f}",
            "Throughput": f"{s['throughput_fps']:.1f} FPS",
            "Peak RAM": f"{s['peak_ram_mb']:.1f} MB",
            "Avg CPU": f"{s['avg_cpu_pct']:.1f}%",
            "Avg EAR": f"{s['avg_ear']:.4f}" if not pd.isna(s['avg_ear']) else "N/A"
        })

    df_comp = pd.DataFrame(comp_rows)
    print("\n" + "=" * 115)
    print("         TABEL 1: HASIL BENCHMARK PERFORMA KOMPUTASI & EKSTRAKSI LANDMARK")
    print("=" * 115)
    headers1 = list(df_comp.columns)
    col_w1 = [max(len(h), max(len(str(val)) for val in df_comp[h])) + 2 for h in headers1]
    header_str1 = "| " + " | ".join(h.center(w) for h, w in zip(headers1, col_w1)) + " |"
    div1 = "+-" + "-+-".join("-" * w for w in col_w1) + "-+"

    print(div1)
    print(header_str1)
    print(div1)
    for _, row in df_comp.iterrows():
        print("| " + " | ".join(str(row[h]).center(w) for h, w in zip(headers1, col_w1)) + " |")
    print(div1)

    # --- TABEL 2: Evaluasi Ground Truth & Deteksi Kedipan (Jika GT ada) ---
    if has_gt:
        acc_rows = []
        for s in summaries:
            acc_rows.append({
                "Framework": s["framework"],
                "Thresh": f"{s['ear_threshold']:.2f}",
                "TP": s["tp"],
                "TN": s["tn"],
                "FP": s["fp"],
                "FN": s["fn"],
                "Accuracy": f"{s['accuracy_pct']:.2f}%",
                "Precision": f"{s['precision_pct']:.2f}%",
                "Recall": f"{s['recall_pct']:.2f}%",
                "F1-Score": f"{s['f1_score_pct']:.2f}%",
                "GT Blinks": gt_blinks,
                "Pred Blinks": s["pred_blinks"]
            })

        df_acc = pd.DataFrame(acc_rows)
        print("\n" + "=" * 115)
        print("         TABEL 2: EVALUASI KLASIFIKASI KONDISI MATA & KEDIPAN (VS GROUND TRUTH)")
        print("=" * 115)
        headers2 = list(df_acc.columns)
        col_w2 = [max(len(h), max(len(str(val)) for val in df_acc[h])) + 2 for h in headers2]
        header_str2 = "| " + " | ".join(h.center(w) for h, w in zip(headers2, col_w2)) + " |"
        div2 = "+-" + "-+-".join("-" * w for w in col_w2) + "-+"

        print(div2)
        print(header_str2)
        print(div2)
        for _, row in df_acc.iterrows():
            print("| " + " | ".join(str(row[h]).center(w) for h, w in zip(headers2, col_w2)) + " |")
        print(div2)
        print("=" * 115)


# ==============================================================================
# 9. VISUALISASI GRAFIK KOMPARASI (MATPLOTLIB)
# ==============================================================================

def generate_comparison_plots(
    df: pd.DataFrame,
    output_image_path: str,
    summaries: List[Dict],
    has_gt: bool = False,
    gt_intervals: Optional[List[Tuple[int, int]]] = None
):
    """
    Membuat grafik analisis komparasi multi-panel (3 atau 4 panel) dan menyimpannya ke PNG.
    Jika Ground Truth tersedia: menambahkan shading GT dan visualisasi Confusion Matrix.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    
    dlib_df = df[df["framework"] == "Dlib 68"].sort_values("frame_id")
    mp_df = df[df["framework"] == "MediaPipe FaceMesh"].sort_values("frame_id")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    num_panels = 4 if has_gt else 3
    fig, axes = plt.subplots(num_panels, 1, figsize=(13, 5.0 * num_panels), gridspec_kw={"hspace": 0.38})
    fig.suptitle("Analisis Komparatif: Dlib 68 vs MediaPipe Face Mesh", fontsize=16, fontweight="bold", y=0.99)

    # Panel 1: Latensi Inferensi per Frame
    ax1 = axes[0]
    ax1.plot(dlib_df["frame_id"], dlib_df["latency_ms"], label="Dlib 68 (ms)", color="#d9534f", alpha=0.6, linewidth=1.2)
    ax1.plot(mp_df["frame_id"], mp_df["latency_ms"], label="MediaPipe Face Mesh (ms)", color="#0275d8", alpha=0.75, linewidth=1.2)
    
    dlib_mean = dlib_df["latency_ms"].mean()
    mp_mean = mp_df["latency_ms"].mean()
    ax1.axhline(dlib_mean, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Dlib Rata-rata: {dlib_mean:.1f} ms")
    ax1.axhline(mp_mean, color="#0275d8", linestyle="--", linewidth=1.5, label=f"MediaPipe Rata-rata: {mp_mean:.1f} ms")

    ax1.set_title("1. Latensi Inferensi per Frame (Waktu Proses)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("Latensi (ms)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Nilai Eye Aspect Ratio (EAR) per Frame & Shading Ground Truth
    ax2 = axes[1]
    ax2.plot(dlib_df["frame_id"], dlib_df["ear_value"], label="Dlib EAR", color="#d9534f", linewidth=1.5, alpha=0.8)
    ax2.plot(mp_df["frame_id"], mp_df["ear_value"], label="MediaPipe EAR", color="#0275d8", linewidth=1.5, alpha=0.8)
    
    # Threshold lines
    dlib_thresh = summaries[0]["ear_threshold"] if len(summaries) > 0 else 0.21
    mp_thresh = summaries[1]["ear_threshold"] if len(summaries) > 1 else 0.21
    if abs(dlib_thresh - mp_thresh) < 1e-4:
        ax2.axhline(dlib_thresh, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Threshold EAR ({dlib_thresh:.2f})")
    else:
        ax2.axhline(dlib_thresh, color="#d9534f", linestyle=":", linewidth=1.5, label=f"Dlib Thresh ({dlib_thresh:.2f})")
        ax2.axhline(mp_thresh, color="#0275d8", linestyle=":", linewidth=1.5, label=f"MediaPipe Thresh ({mp_thresh:.2f})")

    # Shading untuk Ground Truth Tertutup
    if has_gt and gt_intervals:
        for idx_iv, (start_f, end_f) in enumerate(gt_intervals):
            label_gt = "Ground Truth (Mata Tertutup)" if idx_iv == 0 else ""
            ax2.axvspan(start_f, end_f, color="#2ca02c", alpha=0.25, label=label_gt)

    ax2.set_title("2. Nilai Eye Aspect Ratio (EAR) & Rentang Kedipan", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frame ID")
    ax2.set_ylabel("Nilai EAR")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Metrik Evaluasi Komparatif (Bar Chart)
    ax3 = axes[2]
    if has_gt:
        metrics = ["Face Det (%)", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
        dlib_s = [s for s in summaries if s["framework"] == "Dlib 68"][0]
        mp_s = [s for s in summaries if s["framework"] == "MediaPipe FaceMesh"][0]

        dlib_vals = [dlib_s["face_det_pct"], dlib_s["accuracy_pct"], dlib_s["precision_pct"], dlib_s["recall_pct"], dlib_s["f1_score_pct"]]
        mp_vals = [mp_s["face_det_pct"], mp_s["accuracy_pct"], mp_s["precision_pct"], mp_s["recall_pct"], mp_s["f1_score_pct"]]
        ax3.set_title("3. Metrik Akurasi & Deteksi vs Ground Truth", fontsize=12, fontweight="bold")
    else:
        metrics = ["Avg Latency (ms)", "Throughput (FPS)", "Avg CPU (%)", "Peak RAM (MB)"]
        dlib_vals = [dlib_mean, 1000.0 / dlib_mean if dlib_mean > 0 else 0, dlib_df["cpu_percent"].mean(), dlib_df["ram_mb"].max()]
        mp_vals = [mp_mean, 1000.0 / mp_mean if mp_mean > 0 else 0, mp_df["cpu_percent"].mean(), mp_df["ram_mb"].max()]
        ax3.set_title("3. Ringkasan Metrik Kunci Komparatif", fontsize=12, fontweight="bold")

    x = np.arange(len(metrics))
    width = 0.35
    rects1 = ax3.bar(x - width/2, dlib_vals, width, label="Dlib 68", color="#d9534f", alpha=0.85)
    rects2 = ax3.bar(x + width/2, mp_vals, width, label="MediaPipe Face Mesh", color="#0275d8", alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontweight="bold")
    ax3.set_ylim(0, max(max(dlib_vals), max(mp_vals)) * 1.15)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f"{height:.1f}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    # Panel 4: Confusion Matrix Breakdown (Hanya jika GT tersedia)
    if has_gt:
        ax4 = axes[3]
        cm_labels = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
        dlib_cm = [dlib_s["tp"], dlib_s["tn"], dlib_s["fp"], dlib_s["fn"]]
        mp_cm = [mp_s["tp"], mp_s["tn"], mp_s["fp"], mp_s["fn"]]

        x_cm = np.arange(len(cm_labels))
        rects_cm1 = ax4.bar(x_cm - width/2, dlib_cm, width, label=f"Dlib 68 (Blinks: {dlib_s['pred_blinks']})", color="#d9534f", alpha=0.85)
        rects_cm2 = ax4.bar(x_cm + width/2, mp_cm, width, label=f"MediaPipe (Blinks: {mp_s['pred_blinks']})", color="#0275d8", alpha=0.85)

        ax4.set_title("4. Distribusi Confusion Matrix (Klasifikasi Frame)", fontsize=12, fontweight="bold")
        ax4.set_xticks(x_cm)
        ax4.set_xticklabels(cm_labels, fontweight="bold")
        ax4.legend(loc="upper right", frameon=True)
        ax4.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax4.set_ylabel("Jumlah Frame")

        for rect in rects_cm1:
            h = rect.get_height()
            ax4.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for rect in rects_cm2:
            h = rect.get_height()
            ax4.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.96, hspace=0.38)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OUTPUT] Grafik komparasi berhasil disimpan ke: {os.path.abspath(output_image_path)}")


# ==============================================================================
# 10. REKAPITULASI SKENARIO KE CSV
# ==============================================================================

def append_to_summary_csv(
    summary_csv_path: str,
    summaries: List[Dict],
    source_name: str,
    lighting: str,
    glasses: str,
    gt_blinks: int = 0
):
    """Menambahkan ringkasan eksekusi skenario ke file CSV rekapitulasi kumulatif."""
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for s in summaries:
        rows.append({
            "timestamp": now_str,
            "source_video": os.path.basename(source_name),
            "scenario_lighting": lighting,
            "scenario_glasses": glasses,
            "framework": s["framework"],
            "total_frames": s["total_frames"],
            "face_det_pct": round(s["face_det_pct"], 2),
            "avg_latency_ms": round(s["avg_latency_ms"], 2),
            "p95_latency_ms": round(s["p95_latency_ms"], 2),
            "throughput_fps": round(s["throughput_fps"], 2),
            "peak_ram_mb": round(s["peak_ram_mb"], 2),
            "avg_cpu_pct": round(s["avg_cpu_pct"], 2),
            "avg_ear": round(s["avg_ear"], 4) if not pd.isna(s["avg_ear"]) else "",
            "ear_threshold": s["ear_threshold"],
            "tp": s["tp"],
            "tn": s["tn"],
            "fp": s["fp"],
            "fn": s["fn"],
            "accuracy_pct": round(s["accuracy_pct"], 2) if s["accuracy_pct"] is not None else "",
            "precision_pct": round(s["precision_pct"], 2) if s["precision_pct"] is not None else "",
            "recall_pct": round(s["recall_pct"], 2) if s["recall_pct"] is not None else "",
            "f1_score_pct": round(s["f1_score_pct"], 2) if s["f1_score_pct"] is not None else "",
            "gt_blinks": gt_blinks,
            "pred_blinks": s["pred_blinks"]
        })

    df_new = pd.DataFrame(rows)
    file_exists = os.path.exists(summary_csv_path)
    df_new.to_csv(summary_csv_path, mode="a" if file_exists else "w", header=not file_exists, index=False)
    print(f"[OUTPUT] Rekapitulasi skenario berhasil disimpan ke: {os.path.abspath(summary_csv_path)}")


# ==============================================================================
# 11. MAIN CLI ARGUMENT PARSER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Komparasi Performa & Evaluasi Akurasi EAR: Dlib 68 vs MediaPipe Face Mesh."
    )
    
    # Input source & skenario
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Sumber input: '0' untuk webcam, path file video (.mp4/.avi), atau 'synthetic' (default: synthetic)."
    )
    parser.add_argument(
        "--scenario-lighting",
        type=str,
        default="normal",
        help="Label skenario pencahayaan (mis. '300-400 lux' atau '10-20 lux')."
    )
    parser.add_argument(
        "--scenario-glasses",
        type=str,
        default="tanpa_kacamata",
        help="Label skenario kacamata (mis. 'tanpa_kacamata', 'bingkai_tipis', 'bingkai_tebal', 'pantulan_cahaya')."
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
        help="Metode pengujian: 'interleaved' (frame sama bergantian) atau 'sequential' (Dlib lalu MediaPipe)."
    )

    # Threshold EAR & Ground Truth
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=None,
        help="Threshold EAR universal untuk Dlib & MediaPipe (jika dispesifikasi, override threshold spesifik)."
    )
    parser.add_argument(
        "--ear-thresh-dlib",
        type=float,
        default=0.21,
        help="Threshold EAR untuk Dlib 68 (default: 0.21)."
    )
    parser.add_argument(
        "--ear-thresh-mp",
        type=float,
        default=0.21,
        help="Threshold EAR untuk MediaPipe Face Mesh (default: 0.21)."
    )
    parser.add_argument(
        "--min-blink-frames",
        type=int,
        default=2,
        help="Jumlah frame tertutup berturut-turut minimum untuk dihitung sebagai 1 kedipan (default: 2 frame)."
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        default=None,
        help="Path ke file Ground Truth CSV (format per-frame atau format rentang start_frame,end_frame)."
    )
    parser.add_argument(
        "--gt-closed-ranges",
        type=str,
        default=None,
        help="Rentang frame saat mata tertutup, dipisahkan koma (contoh: '45-52,120-128,210-219')."
    )
    
    # Path model Dlib
    parser.add_argument(
        "--dlib-model",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shape_predictor_68_face_landmarks.dat"),
        help="Path ke file shape_predictor_68_face_landmarks.dat."
    )
    
    # Output file
    parser.add_argument(
        "--csv-output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_results.csv"),
        help="Path file output CSV log per-frame (default: benchmark_results.csv)."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path file CSV rekapitulasi skenario (mendukung append multi-skenario)."
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_comparison.png"),
        help="Path file output grafik perbandingan (default: benchmark_comparison.png)."
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

    # Tentukan threshold EAR
    thresh_dlib = args.ear_thresh if args.ear_thresh is not None else args.ear_thresh_dlib
    thresh_mp = args.ear_thresh if args.ear_thresh is not None else args.ear_thresh_mp

    print("=" * 78)
    print("   BLINKLINK-IOT: BENCHMARK KOMPARASI & EVALUASI AKURASI DLIB VS MEDIAPIPE")
    print("=" * 78)
    print(f" Sumber Input       : {args.source}")
    print(f" Skenario Cahaya    : {args.scenario_lighting}")
    print(f" Skenario Kacamata  : {args.scenario_glasses}")
    print(f" Mode Pengujian     : {args.mode.upper()}")
    print(f" Total Frame Target : {args.frames}")
    print(f" Warmup Frames      : {args.warmup}")
    print(f" Threshold EAR Dlib : {thresh_dlib}")
    print(f" Threshold EAR MP   : {thresh_mp}")
    print(f" Min Blink Frames   : {args.min_blink_frames}")
    print(f" Ground Truth Info  : {'CLI Ranges' if args.gt_closed_ranges else (args.gt_file if args.gt_file else 'Tidak Ada')}")
    print(f" CSV Output (Frame) : {args.csv_output}")
    print(f" CSV Summary Rekap  : {args.summary_csv if args.summary_csv else 'Disabled'}")
    print(f" Plot Output        : {args.plot_output if not args.no_plot else 'Disabled'}")
    print("=" * 78)

    # 1. Pastikan model Dlib tersedia
    dlib_path = ensure_dlib_model(args.dlib_model, auto_download=not args.no_download)

    # 2. Inisialisasi runner
    monitor = SystemResourceMonitor()
    dlib_bench = DlibEARBenchmark(dlib_path)
    mp_bench = MediaPipeEARBenchmark()

    # 3. Jalankan ekstraksi benchmark
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

    df_raw = pd.DataFrame(records)
    actual_frames = df_raw["frame_id"].max()

    # 4. Muat Ground Truth jika disediakan
    gt_array, gt_blinks, gt_intervals = parse_ground_truth(
        total_frames=actual_frames,
        gt_file=args.gt_file,
        gt_ranges=args.gt_closed_ranges
    )
    has_gt = gt_array is not None

    # 5. Evaluasi masing-masing framework
    dlib_raw = df_raw[df_raw["framework"] == "Dlib 68"]
    mp_raw = df_raw[df_raw["framework"] == "MediaPipe FaceMesh"]

    dlib_eval_df, dlib_summary = evaluate_framework_results(
        df_framework=dlib_raw,
        threshold=thresh_dlib,
        min_blink_frames=args.min_blink_frames,
        gt_array=gt_array
    )

    mp_eval_df, mp_summary = evaluate_framework_results(
        df_framework=mp_raw,
        threshold=thresh_mp,
        min_blink_frames=args.min_blink_frames,
        gt_array=gt_array
    )

    summaries = [dlib_summary, mp_summary]

    # Gabungkan kembali dataframe evaluasi
    df_results = pd.concat([dlib_eval_df, mp_eval_df], ignore_index=True)
    df_results["scenario_lighting"] = args.scenario_lighting
    df_results["scenario_glasses"] = args.scenario_glasses

    # Urutkan kolom agar informatif
    col_order = [
        "framework", "scenario_lighting", "scenario_glasses", "frame_id",
        "latency_ms", "fps_instant", "face_detected", "ear_value", "ear_threshold",
        "pred_closed", "gt_closed", "eval_outcome", "blink_count_cum", "ram_mb", "cpu_percent"
    ]
    df_results = df_results[[c for c in col_order if c in df_results.columns]]

    # 6. Tampilkan Tabel Ringkasan ASCII di Konsol
    print_ascii_summary_tables(summaries, has_gt=has_gt, gt_blinks=gt_blinks)

    # 7. Ekspor ke file CSV per-frame
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_output)), exist_ok=True)
    df_results.to_csv(args.csv_output, index=False)
    print(f"\n[OUTPUT] Log metrik per-frame berhasil diekspor ke: {os.path.abspath(args.csv_output)}")

    # 8. Ekspor ke file CSV rekapitulasi skenario jika diminta
    if args.summary_csv:
        append_to_summary_csv(
            summary_csv_path=args.summary_csv,
            summaries=summaries,
            source_name=args.source,
            lighting=args.scenario_lighting,
            glasses=args.scenario_glasses,
            gt_blinks=gt_blinks
        )

    # 9. Pembuatan Grafik Plotting (Matplotlib)
    if not args.no_plot:
        generate_comparison_plots(
            df=df_results,
            output_image_path=args.plot_output,
            summaries=summaries,
            has_gt=has_gt,
            gt_intervals=gt_intervals
        )

    print("\n[SELESAI] Seluruh pengujian benchmark & evaluasi akurasi berhasil dilaksanakan.")


if __name__ == "__main__":
    main()
