"""
BlinkLink-IoT Model Benchmarking Script v2
============================================
Writes results to a JSON file for reliable output capture.
"""

import sys
import os
import time
import traceback
import json

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "benchmark_results.json")

results = {}
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def measure_inference(func, iterations=100, warmup=10):
    """Mengukur rata-rata inference time dalam ms."""
    for _ in range(warmup):
        try:
            func()
        except:
            pass
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    times.sort()
    trim = max(1, len(times) // 20)
    trimmed = times[trim:-trim] if trim > 0 else times
    
    return {
        "mean_ms": round(sum(trimmed) / len(trimmed), 4),
        "median_ms": round(trimmed[len(trimmed) // 2], 4),
        "min_ms": round(min(trimmed), 4),
        "max_ms": round(max(trimmed), 4),
        "p95_ms": round(trimmed[int(len(trimmed) * 0.95)], 4),
        "p99_ms": round(trimmed[int(len(trimmed) * 0.99)], 4),
        "fps_equivalent": round(1000.0 / (sum(trimmed) / len(trimmed)), 2),
    }

def get_model_size_mb(path):
    if os.path.exists(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    return None

log("=" * 60)
log("BlinkLink-IoT Model Benchmark v2")
log("=" * 60)

import numpy as np
import cv2

# Create synthetic test image (simulate 800x600 webcam frame)
test_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
test_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
test_face_roi = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

# ============================================================
# 1. MediaPipe Face Mesh
# ============================================================
log("\n[1] Benchmarking MediaPipe Face Mesh...")
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    def mp_infer():
        rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return face_mesh.process(rgb)
    
    mp_result = measure_inference(mp_infer, iterations=200)
    
    mp_path = os.path.dirname(mp.__file__)
    mp_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(mp_path)
        for f in filenames
        if f.endswith(('.tflite', '.binarypb'))
    ) / (1024 * 1024)
    
    results["MediaPipe Face Mesh"] = {
        "inference": mp_result,
        "model_size_mb": round(mp_size, 2),
        "status": "OK"
    }
    log(f"   Mean: {mp_result['mean_ms']:.2f}ms ({mp_result['fps_equivalent']} FPS), Model: {mp_size:.2f}MB")
except Exception as e:
    log(f"   SKIP: {e}")
    results["MediaPipe Face Mesh"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 2. FaceNet (keras-facenet) 
# ============================================================
log("\n[2] Benchmarking FaceNet (keras-facenet)...")
try:
    from keras_facenet import FaceNet
    embedder = FaceNet()
    
    def facenet_infer():
        face_array = np.asarray(cv2.resize(
            cv2.cvtColor(test_face_roi, cv2.COLOR_BGR2RGB), (160, 160)
        ))
        return embedder.embeddings([face_array])
    
    fn_result = measure_inference(facenet_infer, iterations=100)
    
    # Try to find model cache
    import keras_facenet as kf
    home = os.path.expanduser("~")
    facenet_cache = os.path.join(home, ".keras-facenet")
    fn_size = 0
    if os.path.exists(facenet_cache):
        for dp, dn, filenames in os.walk(facenet_cache):
            for f in filenames:
                fn_size += os.path.getsize(os.path.join(dp, f))
    fn_size_mb = round(fn_size / (1024 * 1024), 2) if fn_size > 0 else 23.0
    
    results["FaceNet (keras-facenet)"] = {
        "inference": fn_result,
        "model_size_mb": fn_size_mb,
        "embedding_dim": 512,
        "input_size": "160x160",
        "status": "OK"
    }
    log(f"   Mean: {fn_result['mean_ms']:.2f}ms ({fn_result['fps_equivalent']} FPS), Model: {fn_size_mb:.2f}MB")
except Exception as e:
    log(f"   SKIP: {e}")
    traceback.print_exc()
    results["FaceNet (keras-facenet)"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 3. EAR Blink Detection
# ============================================================
log("\n[3] Benchmarking EAR Blink Detection...")
try:
    from scipy.spatial import distance as dist
    
    eye_points = np.array([
        [100, 200], [110, 190], [130, 190],
        [140, 200], [130, 210], [110, 210]
    ], dtype=np.float64)
    
    def ear_infer():
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)
    
    ear_result = measure_inference(ear_infer, iterations=1000)
    
    results["EAR Blink Detection"] = {
        "inference": ear_result,
        "model_size_mb": 0.0,
        "status": "OK"
    }
    log(f"   Mean: {ear_result['mean_ms']:.4f}ms ({ear_result['fps_equivalent']} FPS)")
except Exception as e:
    log(f"   SKIP: {e}")
    results["EAR Blink Detection"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 4. Dlib
# ============================================================
log("\n[4] Benchmarking Dlib 68-Landmark...")
dlib_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "shape_predictor_68_face_landmarks.dat")
try:
    import dlib
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_model_path)
    test_rect = dlib.rectangle(200, 150, 400, 400)
    
    def dlib_detect_infer():
        return detector(test_gray, 0)
    
    def dlib_landmark_infer():
        return predictor(test_gray, test_rect)
    
    dlib_det_result = measure_inference(dlib_detect_infer, iterations=50)
    dlib_lm_result = measure_inference(dlib_landmark_infer, iterations=100)
    
    dlib_model_size = get_model_size_mb(dlib_model_path)
    
    results["Dlib HOG Detector"] = {
        "inference": dlib_det_result,
        "model_size_mb": dlib_model_size or 0,
        "status": "OK"
    }
    results["Dlib 68-Landmark"] = {
        "inference": dlib_lm_result,
        "model_size_mb": dlib_model_size or 95.06,
        "status": "OK"
    }
    log(f"   HOG Detect Mean: {dlib_det_result['mean_ms']:.2f}ms ({dlib_det_result['fps_equivalent']} FPS)")
    log(f"   Landmark Mean: {dlib_lm_result['mean_ms']:.2f}ms ({dlib_lm_result['fps_equivalent']} FPS)")
    log(f"   Model: {dlib_model_size}MB")
except Exception as e:
    log(f"   SKIP (dlib not installed): {e}")
    results["Dlib HOG Detector"] = {"status": "SKIP", "error": str(e)}
    results["Dlib 68-Landmark"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 5. Haar Cascade
# ============================================================
log("\n[5] Benchmarking Haar Cascade...")
try:
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(haar_path)
    
    def haar_infer():
        return haar_cascade.detectMultiScale(test_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    haar_result = measure_inference(haar_infer, iterations=100)
    haar_size = get_model_size_mb(haar_path)
    
    results["Haar Cascade"] = {
        "inference": haar_result,
        "model_size_mb": haar_size or 0.9,
        "status": "OK"
    }
    log(f"   Mean: {haar_result['mean_ms']:.2f}ms ({haar_result['fps_equivalent']} FPS), Model: {haar_size}MB")
except Exception as e:
    log(f"   SKIP: {e}")
    results["Haar Cascade"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 6. CLAHE Enhancement
# ============================================================
log("\n[6] Benchmarking CLAHE Enhancement...")
try:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def clahe_infer():
        lab = cv2.cvtColor(test_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    clahe_result = measure_inference(clahe_infer, iterations=200)
    
    results["CLAHE Enhancement"] = {
        "inference": clahe_result,
        "model_size_mb": 0.0,
        "status": "OK"
    }
    log(f"   Mean: {clahe_result['mean_ms']:.2f}ms ({clahe_result['fps_equivalent']} FPS)")
except Exception as e:
    log(f"   SKIP: {e}")
    results["CLAHE Enhancement"] = {"status": "SKIP", "error": str(e)}

# ============================================================
# 7. Memory Usage
# ============================================================
log("\n[7] Measuring Memory Usage...")
try:
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    results["_system"] = {
        "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
    }
    log(f"   RSS: {results['_system']['rss_mb']}MB, VMS: {results['_system']['vms_mb']}MB")
except:
    log("   psutil not available, skipping memory measurement")

# ============================================================
# Save Results
# ============================================================
output = {
    "results": results,
    "log": log_lines,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "frame_size": "800x600",
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

log(f"\n✅ Results saved to: {RESULTS_FILE}")
log("Benchmark complete.")
