import sys
print(f"Python executable: {sys.executable}")
try:
    import mediapipe as mp
    print(f"MediaPipe location: {mp.__file__}")
    print(f"Solutions available: {dir(mp.solutions)}")
    mp_face_mesh = mp.solutions.face_mesh
    print("✅ MediaPipe Solutions berhasil dimuat!")
except AttributeError as e:
    print(f"❌ AttributeError: {e}")
    print("Kemungkinan instalasi MediaPipe rusak.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error lain: {e}")
