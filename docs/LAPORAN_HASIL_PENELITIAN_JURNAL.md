# LAPORAN HASIL PENELITIAN & PEMBAHASAN
## ANALISIS KOMPARASI KINERJA MEDIAPIPE FACE MESH DAN DLIB 68-LANDMARKS DALAM DETEKSI KEDIPAN MATA BERBASIS EYE ASPECT RATIO (EAR) PADA SISTEM ASSISTIVE IOT

> **Format Standar Jurnal Nasional Terakreditasi (SINTA 4 / SINTA 5)**  
> *Target Struktur: Bagian III (Hasil dan Pembahasan) & Bagian IV (Kesimpulan dan Saran)*  
> *Sesuai Pembagian Kerja Tim Peneliti (Anggota 3: Accuracy & Obstruction Analyst, Anggota 4: Performance & Hardware Benchmark, Anggota 5: Synthesizer & Tabel Utama)*

---

### RINGKASAN EKSEKUTIF PENELITIAN

Penelitian ini melakukan benchmarking komparatif empiris head-to-head (*interleaved processing*) antara **MediaPipe Face Mesh (478 titik landmark)** dan **Dlib 68 Facial Landmarks (HOG + Ensemble of Regression Trees)** dalam mengekstraksi metrik *Eye Aspect Ratio* (EAR) untuk deteksi kedipan mata (*blink detection*). Pengujian dilakukan menggunakan matriks skenario $2 \times 2$ kondisi lingkungan nyata pada video Full HD (1920×1080 piksel) dengan total **2.932 frame uji**:
1. **Skenario 1**: Pencahayaan Normal (300–500 lux) + Tanpa Kacamata (`normal.mp4`, 644 frames).
2. **Skenario 2**: Pencahayaan Rendah / Low-Light (10–30 lux) + Tanpa Kacamata (`normal-low-light.mp4`, 974 frames).
3. **Skenario 3**: Pencahayaan Normal (300–500 lux) + Dengan Kacamata Bening (`kacamata.mp4`, 328 frames).
4. **Skenario 4**: Pencahayaan Rendah / Low-Light (10–30 lux) + Dengan Kacamata Bening (`kacamata-low-light.mp4`, 986 frames).

Setiap frame diuji menggunakan acuan *Ground Truth* (GT) interval penutupan mata nyata untuk menghasilkan matriks konfusi (*Confusion Matrix*) serta evaluasi komputasi perangkat keras secara objektif.

---

## III. HASIL DAN PEMBAHASAN

### 3.1. Analisis Akurasi dan Ketahanan terhadap Oklusi Lingkungan (Lingkup Anggota 3)

Evaluasi akurasi dilakukan dengan membandingkan prediksi kondisi mata per-frame terhadap data *Ground Truth* (GT). Formula metrik yang digunakan mengacu pada standar klasifikasi biner:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Dimana:
- **TP (True Positive)**: Frame mata tertutup yang berhasil diklasifikasikan dengan benar sebagai tertutup ($\text{EAR} < \text{Threshold}$).
- **TN (True Negative)**: Frame mata terbuka yang diklasifikasikan dengan benar sebagai terbuka ($\text{EAR} \ge \text{Threshold}$).
- **FP (False Positive)**: Frame mata terbuka namun salah terdeteksi tertutup (memicu kedipan semu/*flicker*).
- **FN (False Negative)**: Frame mata tertutup namun gagal terdeteksi tertutup (kedipan terlewat/*missed blink*).

Tabel 1 menyajikan perbandingan matriks konfusi dan metrik klasifikasi untuk seluruh skenario pengujian.

#### Tabel 1. Evaluasi Akurasi Klasifikasi Kondisi Mata dan Deteksi Kedipan vs Ground Truth
| Skenario Pengujian | Model / Framework | Ambang Batas (Thresh) | TP | TN | FP | FN | Akurasi (%) | Presisi (%) | Recall (%) | F1-Score (%) | GT Blinks | Pred Blinks |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skenario 1: Normal**<br>*(300–500 lux, Tanpa Kacamata)* | **MediaPipe**<br>Dlib 68 | 0.17<br>0.22 | 264<br>261 | 380<br>373 | 0<br>7 | 0<br>3 | **100.00**<br>98.45 | **100.00**<br>97.39 | **100.00**<br>98.86 | **100.00**<br>98.12 | 5<br>5 | **5**<br>7 |
| **Skenario 2: Low-Light**<br>*(10–30 lux, Tanpa Kacamata)* | **MediaPipe**<br>Dlib 68 | 0.17<br>0.22 | 396<br>225 | 578<br>543 | 0<br>35 | 0<br>171 | **100.00**<br>78.85 | **100.00**<br>86.54 | **100.00**<br>56.82 | **100.00**<br>68.60 | 5<br>5 | **5**<br>33 |
| **Skenario 3: Kacamata**<br>*(300–500 lux, Pakai Kacamata)* | **MediaPipe**<br>Dlib 68 | 0.20<br>0.24 | 132<br>123 | 196<br>195 | 0<br>1 | 0<br>9 | **100.00**<br>96.95 | **100.00**<br>99.19 | **100.00**<br>93.18 | **100.00**<br>96.09 | 5<br>5 | **5**<br>6 |
| **Skenario 4: Kacamata + Low-Light**<br>*(10–30 lux, Pakai Kacamata)* | **MediaPipe**<br>Dlib 68 | 0.20<br>0.24 | 386<br>64 | 599<br>600 | 1<br>0 | 0<br>322 | **99.90**<br>67.34 | **99.74**<br>100.00 | **100.00**<br>16.58 | **99.87**<br>28.44 | 5<br>5 | **5**<br>13 |

#### Analisis Pembahasan Akurasi & Ketahanan Lingkungan:
1. **Ketahanan terhadap Penurunan Intensitas Cahaya (Low-Light)**:
   Pada kondisi pencahayaan redup (Skenario 2), performa MediaPipe Face Mesh tidak mengalami degradasi sama sekali dengan perolehan **F1-Score 100.00%** dan tepat mendeteksi 5 dari 5 kedipan aktual. Sebaliknya, Dlib 68 mengalami penurunan akurasi signifikan menjadi **78.85%** dengan F1-Score **68.60%** (Recall anjlok ke 56.82%). Hal ini disebabkan oleh hilangnya kontur gradien tepi kelopak mata pada citra gelap, sehingga nilai EAR Dlib berfluktuasi secara tidak stabil di sekitar ambang batas dan menghasilkan **35 False Positive (FP)**, yang menyebabkan over-counting sebanyak **33 kedipan semu**.
2. **Pengaruh Oklusi Kacamata terhadap Pergeseran Nilai EAR**:
   Kehadiran bingkai kacamata (Skenario 3 & 4) menyebabkan pergeseran rata-rata nilai EAR ke atas ($\approx +0.024$ pada MediaPipe dan $\approx +0.021$ pada Dlib). Pembiasan lensa dan bayangan bingkai kacamata menyebabkan jarak vertikal kelopak mata atas-bawah tampak sedikit lebih renggang secara komputasi. MediaPipe mampu mengompensasi fenomena ini dengan menaikkan threshold dinamis ke $0.20$, mempertahankan **F1-Score 100.00%** pada Skenario 3 dan **99.87%** pada Skenario 4.
3. **Kegagalan Ekstrem Dlib pada Kondisi Gabungan (Skenario 4)**:
   Pada kondisi gabungan kacamata dan pencahayaan redup, performa Dlib 68 mengalami keruntuhan total (*catastrophic degradation*) dengan F1-Score hanya **28.44%** dan Recall **16.58%** (terdapat 322 False Negative). Dlib gagal mengenali saat mata terpejam karena titik landmark kelopak mata tertarik ke bingkai kacamata yang gelap.

---

### 3.2. Analisis Kinerja Komputasi dan Profiling Sumber Daya Perangkat Keras (Lingkup Anggota 4)

Pengujian kinerja komputasi dilakukan secara terisolasi pada mesin pengujian berspesifikasi prosesor multi-core dengan lingkungan Python 3.12, memanfaatkan pengukuran latensi inferensi per-frame via `time.perf_counter()` serta utilisasi memori (Peak RSS) dan CPU via modul `psutil`.

#### Tabel 2. Hasil Pengukuran Kinerja Komputasi dan Beban Perangkat Keras
| Skenario Pengujian | Framework / Model | Total Frame | Keberhasilan Deteksi Wajah (%) | Rata-rata Latensi (ms) | P95 Latency (ms) | Throughput (FPS) | Rata-rata EAR | Peak RAM (MB) | Rata-rata CPU (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skenario 1: Normal** | **MediaPipe**<br>Dlib 68 | 644<br>644 | **99.84%**<br>100.00% | **12.44**<br>306.41 | **14.20**<br>348.21 | **80.37 FPS**<br>3.26 FPS | 0.1698<br>0.2484 | 628.67<br>628.67 | 201.15%<br>109.10% |
| **Skenario 2: Low-Light** | **MediaPipe**<br>Dlib 68 | 974<br>974 | **99.90%**<br>100.00% | **12.27**<br>301.87 | **13.39**<br>319.63 | **81.47 FPS**<br>3.31 FPS | 0.1804<br>0.2437 | 629.76<br>629.76 | 202.63%<br>108.98% |
| **Skenario 3: Kacamata** | **MediaPipe**<br>Dlib 68 | 328<br>328 | **99.70%**<br>100.00% | **12.61**<br>304.12 | **13.58**<br>338.53 | **79.29 FPS**<br>3.29 FPS | 0.1936<br>0.2692 | 623.90<br>623.90 | 192.95%<br>110.51% |
| **Skenario 4: Kacamata + Low-Light** | **MediaPipe**<br>Dlib 68 | 986<br>986 | **99.90%**<br>90.47% | **12.43**<br>300.33 | **13.66**<br>326.83 | **80.45 FPS**<br>3.33 FPS | 0.2084<br>0.2674 | 628.64<br>628.63 | 201.10%<br>107.66% |

#### Analisis Pembahasan Performa & Sumber Daya:
1. **Kecepatan Inferensi dan Throughput Real-Time**:
   MediaPipe Face Mesh menunjukkan latensi inferensi yang sangat rendah dan stabil pada kisaran **12.27 ms hingga 12.61 ms per frame**, menghasilkan throughput rata-rata **~80.4 FPS**. Throughput ini melampaui batas standar interaksi manusia-komputer real-time (kamera 30 FPS membutuhkan waktu proses $\le 33.3$ ms). Sebaliknya, Dlib 68 membutuhkan waktu pemrosesan **300.33 ms hingga 306.41 ms per frame** (**~3.3 FPS**). Rendahnya FPS Dlib menjadikannya tidak layak (*infeasible*) untuk mendeteksi kedipan mata alami manusia yang berlangsung singkat (100–400 ms), karena terdapat risiko tinggi frame penutupan mata terlewatkan di antara interval inferensi.
2. **Keberhasilan Deteksi Wajah pada Citra Beresolusi Tinggi (1080p)**:
   MediaPipe mempertahankan tingkat deteksi wajah stabil di atas **99.70% – 99.90%** pada semua kondisi. Sebaliknya, detektor wajah HOG Dlib mengalami kegagalan deteksi wajah yang substansial pada Skenario 4, di mana deteksi wajah anjlok menjadi **90.47%** (kehilangan wajah pada 94 frame). Hal ini membuktikan bahwa arsitektur Single Shot Multibox Detector (SSD) berbasis convolutional neural network pada MediaPipe jauh lebih kokoh dibandingkan filter linear HOG klasik saat menghadapi degradasi kontras citra.
3. **Analisis Beban CPU dan Utilisasi Multi-threading**:
   Utilisasi CPU MediaPipe tercatat berada pada rata-rata **~192% – 202%** (memanfaatkan sekitar 2 thread prosesor secara efisien melalui akselerasi delegasi TFLite XNNPACK). Sedangkan Dlib mencatat utilisasi **~107% – 111%** (dominan single-thread CPU bound). Konsumsi memori kerja sistem (Peak RAM) kedua model berada pada angka yang identik dan efisien yaitu **~623–629 MB**, yang membuktikan bahwa kedua arsitektur ramah terhadap implementasi embedded PC / single-board computer (SBC).

---

### 3.3. Sintesis dan Matriks Perbandingan Komprehensif (Lingkup Anggota 5)

Sebagai sintesis keseluruhan, Tabel 3 merangkum kelebihan, keterbatasan, dan kelayakan implementasi kedua pendekatan ke dalam sistem antarmuka berbasis bantuan (*assistive IoT*).

#### Tabel 3. Matriks Perbandingan Komprehensif MediaPipe Face Mesh vs Dlib 68
| Dimensi Evaluasi | Dlib 68 Facial Landmarks | MediaPipe Face Mesh | Keunggulan Komparatif |
| :--- | :--- | :--- | :--- |
| **Arsitektur Dasar** | HOG Feature Detector + Kazemi-Sullivan Gradient Tree Boosting | BlazeFace SSD Detector + 3D Mesh Attention Regression | MediaPipe (Modern Deep Learning) |
| **Kerapatan Landmark** | 68 Titik Landmark 2D (6 titik per mata) | 478 Titik Dense Landmark 3D (termasuk iris mesh) | MediaPipe (Resolusi spasial 7x lebih rapat) |
| **Throughput Rata-rata** | 3.26 – 3.33 FPS (Sangat lambat pada 1080p) | 79.29 – 81.47 FPS (Ultra real-time) | **MediaPipe unggul 24.4x lebih cepat** |
| **Latensi Inferensi** | 300 – 306 ms / frame | 12.2 – 12.6 ms / frame | **MediaPipe unggul (margin waktu 290 ms)** |
| **F1-Score Cahaya Redup** | 68.60% (Rentan false positive kedip semu) | **100.00% (Sempurna)** | **MediaPipe unggul mutlak** |
| **F1-Score Oklusi Kacamata** | 28.44% pada kondisi redup (Gagal deteksi) | **99.87% (Tetap stabil)** | **MediaPipe unggul mutlak** |
| **Kebutuhan Dependensi** | Memerlukan C++ Build Tools & CMake saat instalasi | Pustaka siap pakai via PyPI (`pip install mediapipe`) | MediaPipe (Deployment lebih portabel) |
| **Sensitivitas Nilai EAR** | Nilai rata-rata mata terbuka: ~0.24 – 0.27 | Nilai rata-rata mata terbuka: ~0.17 – 0.20 | Keduanya memerlukan kalibrasi terpisah |
| **Kelayakan Assistive IoT** | **Tidak Layak** untuk kontrol saklar/relay kritis | **Sangat Layak & Direkomendasikan** | MediaPipe sebagai pipeline utama |

---

## IV. KESIMPULAN DAN SARAN

### 4.1. Kesimpulan
Berdasarkan hasil pengujian eksperimental 4 skenario kondisi lingkungan nyata dengan total 2.932 frame, disimpulkan bahwa:
1. **MediaPipe Face Mesh membuktikan keunggulan mutlak** dibandingkan Dlib 68 baik dari segi efisiensi komputasi maupun akurasi klasifikasi kedipan mata. MediaPipe menghasilkan throughput rata-rata **~80.4 FPS** (latensi **12.4 ms**), yaitu **24.4 kali lebih cepat** dibandingkan Dlib 68 yang hanya mencapai **~3.3 FPS** (latensi **302 ms**).
2. MediaPipe menunjukkan ketahanan (*robustness*) luar biasa terhadap variasi lingkungan, mempertahankan tingkat deteksi wajah **99.9%** dan **F1-Score antara 99.87% – 100.00%**, serta konsisten mendeteksi tepat 5 dari 5 kedipan aktual pada seluruh skenario.
3. Sebaliknya, Dlib 68 mengalami degradasi kinerja parah pada kondisi pencahayaan rendah dan oklusi kacamata (F1-score merosot hingga **28.44%** dengan over-counting kedipan semu akibat noise citra), sehingga Dlib 68 dinilai **tidak memenuhi syarat keandalan dan keselamatan** untuk antarmuka kendali perangkat listrik/relay disabilitas.

### 4.2. Saran untuk Pengembangan Jurnal / Penelitian Lanjutan
1. **Implementasi Ambang Batas Adaptif (Dynamic Calibration)**: Disarankan untuk menyematkan modul kalibrasi mandiri selama 3 detik pertama saat sistem dijalankan guna menentukan baseline EAR masing-masing pengguna (terutama pengguna berkacamata).
2. **Pengujian pada Dataset Publik Pembanding**: Untuk meningkatkan validitas akademis jurnal SINTA 4/5, penelitian selanjutnya dapat menambahkan evaluasi pembanding pada dataset publik teranotasi standar seperti *Eyeblink8* atau *RT-BENE*.
3. **Pengukuran Latensi End-to-End**: Melakukan pengujian keterlambatan waktu mulai dari fisik mata tertutup, transmisi paket MQTT, hingga kontak mekanik relay ESP32 aktif secara fisik.
