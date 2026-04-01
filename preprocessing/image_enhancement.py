import cv2
import numpy as np

class LowLightEnhancer:
    """
    Kelas untuk meningkatkan kualitas gambar dalam kondisi minim cahaya
    menggunakan teknik CLAHE (Contrast Limited Adaptive Histogram Equalization)
    dan Gamma Correction.
    
    Referensi Penelitian:
    - Pizer et al. "Adaptive histogram equalization and its variations." (1987)
    - Berbagai studi tentang "Low-light image enhancement for face detection".
    """
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.5):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.gamma = gamma
        # Buat lookup table untuk gamma correction
        self.gamma_lut = np.array([((i / 255.0) ** (1.0 / self.gamma)) * 255
                                   for i in np.arange(0, 256)]).astype("uint8")

    def enhance(self, image):
        """
        Menerapkan peningkatan citra pada gambar BGR.
        
        Metode:
        1. Konversi ke LAB Color Space.
        2. Terapkan CLAHE pada channel L (Lightness).
        3. Konversi kembali ke BGR.
        """
        try:
            # 1. Konversi ke LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 2. Terapkan CLAHE pada L-channel
            l_enhanced = self.clahe.apply(l)
            
            # 3. Gabungkan kembali
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            
            # 4. Konversi balik ke BGR
            enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced_image
        except Exception as e:
            print(f"Warning: Image enhancement failed: {e}")
            return image

    def apply_gamma(self, image):
        """
        Menerapkan Gamma Correction.
        Berguna untuk mencerahkan gambar secara global.
        """
        return cv2.LUT(image, self.gamma_lut)

    def is_low_light(self, image, threshold=70):
        """
        Mengecek apakah gambar tergolong gelap berdasarkan rata-rata intensitas pixel.
        """
        if image is None or image.size == 0:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold

    def is_roi_dark(self, image, x, y, w, h, threshold=80):
        """
        Mengecek apakah region tertentu (ROI) gelap.
        Threshold sedikit lebih tinggi karena wajah harusnya cukup terang.
        """
        # Pastikan koordinat valid
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        roi = image[y:y+h, x:x+w]
        return self.is_low_light(roi, threshold)
