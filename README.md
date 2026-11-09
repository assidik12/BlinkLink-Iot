<div align="center">

# 👁️ BlinkLink

### Hands-Free IoT Control via Facial Recognition & Eye Blinks

**Empowering Independence Through Assistive Technology**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MQTT](https://img.shields.io/badge/MQTT-Protocol-blueviolet.svg)](https://mqtt.org/)

[Features](#-key-features) • [Demo](#-demo) • [How It Works](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 🎯 What is BlinkLink?

BlinkLink is an **innovative assistive technology system** designed to help individuals with severe motor disabilities control their environment using **only facial gestures**. By combining cutting-edge **Computer Vision**, **Machine Learning**, and **IoT technology**, BlinkLink transforms simple eye blinks into powerful commands that can control smart home devices, computers, and more.

### The Problem We're Solving

Millions of people worldwide live with conditions like ALS, cerebral palsy, or spinal cord injuries that severely limit their motor control. Traditional assistive devices often require complex setup, expensive hardware, or invasive procedures. **BlinkLink offers a non-invasive, affordable, and easy-to-use alternative** that only requires a standard webcam.

---

## 🌟 Key Features

<table>
<tr>
<td width="50%">

### 🔐 Biometric Security

- **Facial Recognition** ensures only authorized users can access the system
- Custom-trained TensorFlow model for accurate identification
- Privacy-focused: All processing happens locally on your device

</td>
<td width="50%">

### 👁️ Real-Time Gesture & Blink Detection

- High-FPS facial landmark detection using **MediaPipe Face Mesh (478 landmarks)** with fallback support for **Dlib 68-landmarks**
- **Eye Aspect Ratio (EAR) algorithm** with hysteresis filtering (anti-flicker)
- Intentional gesture recognition: distinguishes voluntary long blinks (Action trigger: ~2s, Mode switch: ~5s) from natural involuntary blinks
- Head pose estimation (Yaw/Pitch/Roll) for multi-device navigation

</td>
</tr>
<tr>
<td width="50%">

### 🌐 Wireless IoT Control

- Lightweight **MQTT protocol** for instant device communication
- Control virtually any electronic appliance via **ESP32**
- Expandable to multiple devices and rooms

</td>
<td width="50%">

### 🧩 Modular Architecture

- Clean, **object-oriented Python codebase**
- Easy to maintain, extend, and customize
- Separate modules for authentication and gesture detection

</td>
</tr>
</table>

---

## 🎬 Demo

<div align="center">

[![BlinkLink Demo Video](https://img.shields.io/badge/🎥_Watch_Demo_on-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/posts/ahmad-sofi-sidik_neurotech-bci-iot-activity-7392400198737559552-Arg2?utm_source=share&utm_medium=member_desktop&rcm=ACoAAENGFPgB4uy9rK2c06iPe3EkMmylkvzKKjk)

_Watch BlinkLink in action: The system identifies the user, verifies identity, and responds to eye blinks to control IoT devices._

</div>

### What You'll See:

1. **Face Detection**: System identifies and tracks your face
2. **Authentication**: Facial recognition verifies your identity
3. **Authorization**: Green box indicates you're cleared to control devices
4. **Blink Command**: Simple eye blink triggers IoT device (light/fan/etc.)

---

## 🔬 How It Works

<div align="center">

![System Architecture](docs/architecture.png)

</div>

### The Journey of a Blink

```
👤 User Blinks → 📹 Webcam Captures → 🧠 AI Processes → 📡 MQTT Publishes → 💡 Device Responds
```

1. **Vision Controller (Python/PC)** 🖥️
   - Menangkap stream kamera real-time.
   - Melakukan autentikasi pengguna lokal via FaceNet embeddings.
   - Mengekstraksi landmark mata dengan **MediaPipe Face Mesh** dan menghitung metrik EAR dengan hysteresis filter.
   - Mengirim perintah kontrol ke topik MQTT spesifik (iot/commands/lampu, iot/commands/tv, dll.).

2. **MQTT Broker (Cloud/Local)** ☁️
   - Menjadi jembatan komunikasi antara PC controller dan ESP32.

3. **IoT Device (ESP32 Hardware / Virtual Simulator)** 🔌
   - Menerima instruksi via MQTT.
   - **Kontrol Relay (GPIO 2)**: Menghidupkan/mematikan saklar listrik lampu.
   - **Kontrol IR Sender (GPIO 4)**: Memancarkan sinyal infra-merah NEC untuk mengontrol TV / STB (Volume Up/Down, Channel Up/Down).
   - Mode simulasi tersedia tanpa hardware fisik (python src/iot_devices/iot_device_virtual.py).

---

## 📊 Benchmark Results: Evaluasi Matriks Multi-Skenario & Akurasi

Evaluasi performa komputasi dan akurasi ekstraksi **Eye Aspect Ratio (EAR)** antara **MediaPipe Face Mesh (478 titik)** dan **Dlib 68-Landmarks (HOG + Shape Predictor)** diuji menggunakan matriks 2x2 kondisi nyata (Pencahayaan Normal vs Low-Light, serta Tanpa Kacamata vs Dengan Kacamata) pada resolusi 1080p (total **2.932 frames**).

Pengujian ini dilengkapi dengan **Ground Truth (GT)** interval kedipan nyata untuk mengukur metrik evaluasi klasifikasi secara objektif (*Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix*).

### 📈 Tabel 1: Hasil Evaluasi Akurasi Klasifikasi Kedipan (vs Ground Truth)

File rekapitulasi data lengkap tersimpan di [`docs/benchmarks/summary_comparison.csv`](docs/benchmarks/summary_comparison.csv).

| Skenario Pengujian | Model / Pipeline | Thresh | TP / TN / FP / FN | Accuracy | Precision | Recall | **F1-Score** | GT Blinks | **Pred Blinks** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Normal**<br>*(Terang, Tanpa Kacamata)* | **MediaPipe FaceMesh**<br>Dlib 68 | 0.17<br>0.22 | 264 / 380 / 0 / 0<br>261 / 373 / 7 / 3 | **100.0%**<br>98.5% | **100.0%**<br>97.4% | **100.0%**<br>98.9% | **100.0%**<br>98.1% | 5 kedipan<br>5 kedipan | **5 kedipan**<br>7 kedipan |
| **2. Low-Light**<br>*(Redup, Tanpa Kacamata)* | **MediaPipe FaceMesh**<br>Dlib 68 | 0.17<br>0.22 | 396 / 578 / 0 / 0<br>225 / 543 / 35 / 171 | **100.0%**<br>78.9% | **100.0%**<br>86.5% | **100.0%**<br>56.8% | **100.0%**<br>68.6% | 5 kedipan<br>5 kedipan | **5 kedipan**<br>33 kedipan *(Flicker)* |
| **3. Kacamata**<br>*(Terang, Pakai Kacamata)* | **MediaPipe FaceMesh**<br>Dlib 68 | 0.20<br>0.24 | 132 / 196 / 0 / 0<br>123 / 195 / 1 / 9 | **100.0%**<br>97.0% | **100.0%**<br>99.2% | **100.0%**<br>93.2% | **100.0%**<br>96.1% | 5 kedipan<br>5 kedipan | **5 kedipan**<br>6 kedipan |
| **4. Kacamata + Low-Light**<br>*(Paling Ekstrem)* | **MediaPipe FaceMesh**<br>Dlib 68 | 0.20<br>0.24 | 386 / 599 / 1 / 0<br>64 / 600 / 0 / 322 | **99.9%**<br>67.3% | **99.7%**<br>100.0% | **100.0%**<br>16.6% | **99.9%**<br>28.4% *(Anjlok)* | 5 kedipan<br>5 kedipan | **5 kedipan**<br>13 kedipan |

---

### ⚡ Tabel 2: Performa Komputasi & Resource Hardware

| Skenario Pengujian | Model / Pipeline | Deteksi Wajah (%) | Rata-rata Latensi | P95 Latency | Throughput (FPS) | Rata-rata EAR | Peak RAM | Rata-rata CPU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Normal** | **MediaPipe FaceMesh**<br>Dlib 68 | **99.8%**<br>100.0% | **12.44 ms**<br>306.41 ms | **14.20 ms**<br>348.21 ms | **80.4 FPS**<br>3.3 FPS | **0.1698**<br>0.2484 | 628.7 MB<br>628.7 MB | 201.2%<br>109.1% |
| **2. Low-Light** | **MediaPipe FaceMesh**<br>Dlib 68 | **99.9%**<br>100.0% | **12.27 ms**<br>301.87 ms | **13.39 ms**<br>319.63 ms | **81.5 FPS**<br>3.3 FPS | **0.1804**<br>0.2437 | 629.8 MB<br>629.8 MB | 202.6%<br>109.0% |
| **3. Kacamata** | **MediaPipe FaceMesh**<br>Dlib 68 | **99.7%**<br>100.0% | **12.61 ms**<br>304.12 ms | **13.58 ms**<br>338.53 ms | **79.3 FPS**<br>3.3 FPS | **0.1936**<br>0.2692 | 623.9 MB<br>623.9 MB | 193.0%<br>110.5% |
| **4. Kacamata + Low-Light** | **MediaPipe FaceMesh**<br>Dlib 68 | **99.9%**<br>90.5% *(Drop)* | **12.43 ms**<br>300.33 ms | **13.66 ms**<br>326.83 ms | **80.5 FPS**<br>3.3 FPS | **0.2084**<br>0.2674 | 628.6 MB<br>628.6 MB | 201.1%<br>107.7% |

### 🔍 Temuan & Analisis Ilmiah Utama

1. **Akurasi & Nilai F1-Score Sempurna**: MediaPipe Face Mesh memperoleh **F1-Score 99.9% - 100.0%** dan secara sempurna menangkap tepat **5 dari 5 kedipan nyata (GT)** di semua skenario. Sebaliknya, Dlib mengalami penurunan drastis saat redup (**F1: 68.6%**) dan anjlok parah pada skenario redup berkacamata (**F1: 28.4%**, Recall hanya 16.6%).
2. **Imunitas Terhadap Flicker & False Positive**: Pada kondisi redup, Dlib menghasilkan **33 kedipan** (akibat EAR berfluktuasi melewati ambang batas), sementara MediaPipe tetap stabil di 5 kedipan berkat *dense mesh landmarks* dan pelacakan iris.
3. **Ketahanan Deteksi Wajah**: MediaPipe mempertahankan tingkat deteksi wajah **99.9%** pada kondisi ekstrem. Dlib kehilangan wajah pada ~95 frame (deteksi drop ke **90.5%**) karena ketiadaan gradien tepi pada HOG saat gelap berkacamata.
4. **Kecepatan Komputasi Real-Time**: MediaPipe konsisten di **~80 FPS (latensi ~12.4 ms)**, yaitu **~24.4x lebih cepat** daripada Dlib (~3.3 FPS), menjamin respon kendali IoT tanpa hambatan.

> [!NOTE]
> - **Visualisasi Grafik Lengkap**: Grafik komparasi per skenario (kurva EAR vs GT, latensi, dan resource) tersedia di [`docs/benchmarks/plots/`](docs/benchmarks/plots/) (`plot_normal.png`, `plot_normal_low_light.png`, `plot_kacamata.png`, `plot_kacamata_low_light.png`).
> - **Data Mentah Per-Frame**: File CSV per frame tersimpan di [`docs/benchmarks/data/`](docs/benchmarks/data/).
> - **Hasil JSON Terstruktur**: Terdaftar lengkap pada [`docs/benchmark_results.json`](docs/benchmark_results.json).

---

### 🧪 Panduan Cara Menjalankan Benchmark Mandiri

File video pengujian tidak disertakan di repository Git (di-ignore otomatis via `.gitignore`). Anda dapat menjalankan benchmark menggunakan kamera langsung atau rekaman video Anda sendiri:

#### 1. Persiapan Dependensi
```bash
pip install -r requirements.txt
pip install psutil pandas matplotlib
```

#### 2. Menjalankan Uji Skenario Mandiri (dengan Ground Truth)

Letakkan video Anda di root folder proyek, lalu jalankan perintah sesuai skenario:

```bash
# Skenario 1: Video Normal (Terang, Tanpa Kacamata)
python benchmark_ear.py --source normal.mp4 --frames 644 \
  --scenario-lighting "normal" --scenario-glasses "tanpa_kacamata" \
  --ear-thresh-dlib 0.22 --ear-thresh-mp 0.17 \
  --gt-closed-ranges "27-97,151-203,325-390,454-502,559-583" \
  --csv-output docs/benchmarks/data/results_normal.csv \
  --plot-output docs/benchmarks/plots/plot_normal.png \
  --summary-csv docs/benchmarks/summary_comparison.csv

# Skenario 2: Video Low-Light (Redup, Tanpa Kacamata)
python benchmark_ear.py --source normal-low-light.mp4 --frames 974 \
  --scenario-lighting "low_light" --scenario-glasses "tanpa_kacamata" \
  --ear-thresh-dlib 0.22 --ear-thresh-mp 0.17 \
  --gt-closed-ranges "40-145,228-307,491-588,686-759,845-882" \
  --csv-output docs/benchmarks/data/results_normal_low_light.csv \
  --plot-output docs/benchmarks/plots/plot_normal_low_light.png \
  --summary-csv docs/benchmarks/summary_comparison.csv

# Skenario 3: Video dengan Kacamata (Terang)
python benchmark_ear.py --source kacamata.mp4 --frames 328 \
  --scenario-lighting "normal" --scenario-glasses "dengan_kacamata" \
  --ear-thresh-dlib 0.24 --ear-thresh-mp 0.20 \
  --gt-closed-ranges "24-52,91-123,167-189,224-255,286-300" \
  --csv-output docs/benchmarks/data/results_kacamata.csv \
  --plot-output docs/benchmarks/plots/plot_kacamata.png \
  --summary-csv docs/benchmarks/summary_comparison.csv

# Skenario 4: Video Kacamata + Low-Light (Redup)
python benchmark_ear.py --source kacamata-low-light.mp4 --frames 986 \
  --scenario-lighting "low_light" --scenario-glasses "dengan_kacamata" \
  --ear-thresh-dlib 0.24 --ear-thresh-mp 0.20 \
  --gt-closed-ranges "70-154,272-370,501-569,673-762,859-901" \
  --csv-output docs/benchmarks/data/results_kacamata_low_light.csv \
  --plot-output docs/benchmarks/plots/plot_kacamata_low_light.png \
  --summary-csv docs/benchmarks/summary_comparison.csv

# Menggunakan Live Webcam (Real-time kamera perangkat)
python benchmark_ear.py --source 0 --frames 300 --mode interleaved
```

> [!TIP]
> Parameter `--gt-closed-ranges` menerima daftar interval frame saat mata tertutup (mis. `"50-65,120-135"`). Argumen ini akan secara otomatis menghitung *Confusion Matrix (TP, TN, FP, FN)* serta metrik *Accuracy, Precision, Recall, dan F1-Score* tanpa ada kolom yang kosong.

---

## 🛠️ Technology Stack

<div align="center">

|      Category      | Technologies                                                                                                                                                                                                                                                                                                                                                    |
| :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **AI & Vision**   | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Dlib](https://img.shields.io/badge/Dlib-008000?style=flat) |
| **IoT & Hardware** | ![ESP32](https://img.shields.io/badge/ESP32-000000?style=flat&logo=espressif&logoColor=white)                                                                                                                                                                                                                                                                   |
| **Communication**  | ![MQTT](https://img.shields.io/badge/MQTT-660066?style=flat&logo=mqtt&logoColor=white) ![WiFi](https://img.shields.io/badge/WiFi-0099CC?style=flat)                                                                                                                                                                                                             |
|  **Development**   | ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=flat&logo=visual-studio-code&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) ![Arduino IDE](https://img.shields.io/badge/Arduino_IDE-00979D?style=flat&logo=arduino&logoColor=white)                                                          |

</div>

---

## 🚀 Installation

### Prerequisites

**Hardware:**

- 📹 Standard USB Webcam
- 🔧 ESP32 DevKit V1 Board
- ⚡ 5V Relay Module
- � IR LED (for TV/STB control)
- 🔌 USB Cable & Jumper Wires

### Hardware Wiring (ESP32)

| Component        | ESP32 Pin |
| :--------------- | :-------- |
| **Relay Module** | GPIO 2    |
| **IR LED**       | GPIO 4    |
| **VCC**          | 5V / VIN  |
| **GND**          | GND       |

**Software:**

- 🐍 Python 3.9 or higher
- 🔧 Arduino IDE 1.8.13+
- 💻 Windows: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) & `CMake` (Required for `dlib`)
- 💻 Linux: `build-essential`, `cmake`, `libopenblas-dev`, `liblapack-dev`

### Step 1: Clone the Repository

```bash
git clone https://github.com/assidik12/BlinkLink-Iot.git
cd BlinkLink-Iot
```

### Step 2: Set Up Vision Controller (Python)

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dlib's facial landmark model:**
   - [Download here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place `shape_predictor_68_face_landmarks.dat` in the `vision_controller/` directory.

3. **Collect and Encode Your Face Data:**
   - Run the following command and follow the on-screen instructions to capture and encode your face data:

   ```bash
   python src/vision_controller/collect_face.py
   python src/vision_controller/encoded_face.py
   ```

### Step 3: Set Up IoT Device (ESP32)

1. **Install ESP32 Board in Arduino IDE:**
   - Follow this [guide](https://randomnerdtutorials.com/installing-the-esp32-board-in-arduino-ide/) to add ESP32 support to your Arduino IDE.

2. **Install Required Libraries:**
   - Open Arduino IDE, go to `Sketch > Include Library > Manage Libraries...`
   - Search for and install the following libraries:
     - `PubSubClient` by Nick O'Leary
     - `WiFi` (usually pre-installed with ESP32 board package)

3. **Configure and Upload Code:**
   - Open `iot_device/iot_device.ino` in Arduino IDE.
   - Update Wi-Fi credentials in the code:
     ```cpp
     const char* ssid = "YOUR_SSID";
     const char* password = "YOUR_PASSWORD";
     ```
   - Connect your ESP32 board, select the correct board (`ESP32 Dev Module`) and port in Arduino IDE.
   - Upload the code to the ESP32.

---

## ▶️ Usage

1. **Start the Vision Controller:**

   ```bash
   # Dari root directory project
   python run.py
   
   # Atau langsung dari src
   python src/main.py
   ```

2. **Simulation Mode (Without Hardware):**
   If you don't have an ESP32, you can run the virtual device to test the system:

   ```bash
   python src/iot_devices/iot_device_virtual.py
   ```

3. **Monitor via Web Dashboard:**
   Open `src/ui/index.html` in your web browser to view the real-time status and logs of the system.

4. **Monitor Output:**
   - Watch the terminal for logs on face detection, authentication, and device control.

5. **Interact with IoT Devices:**
   - Use configured eye blink gestures to control connected IoT devices (e.g., turn on lights, fans).

---

## ⚙️ Configuration

You can customize the system behavior by editing `src/helper/config.py`:

- **MQTT Settings**: Change `MQTT_BROKER` and topics to match your setup.
- **UI Settings**: Adjust `CAM_WIDTH`, `SCREEN_HEIGHT`, and `TARGET_FPS`.
- **Thresholds**: Sensitivity for blink detection can be tuned in the detection module.

---

## 🤝 Contributing

We welcome contributions to BlinkLink! Please follow these steps:

1. **Fork the repository**
2. **Create a new branch**: `git checkout -b feature/YourFeature`
3. **Make your changes**
4. **Commit your changes**: `git commit -m 'Add some feature'`
5. **Push to the branch**: `git push origin feature/YourFeature`
6. **Open a Pull Request**

Please ensure your code follows the existing style and includes appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions or feedback, please reach out:

- **Ahmad Sofi Sidik** - [sofi.sidik12@gmail.com](mailto:sofi.sidik12@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/ahmad-sofi-sidik/](https://www.linkedin.com/in/ahmad-sofi-sidik/)

---

Thank you for your interest in BlinkLink! Together, let's make technology accessible to everyone.
