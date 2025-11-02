<div align="center">

# 👁️🎤 BlinkLink

### Multimodal Assistive Technology: Hands-Free & Voice-Free IoT Control

**Empowering Independence Through Facial Recognition, Eye Blinks & Voice Commands**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MQTT](https://img.shields.io/badge/MQTT-Protocol-blueviolet.svg)](https://mqtt.org/)
[![Speech Recognition](https://img.shields.io/badge/Speech-Recognition-red.svg)](https://pypi.org/project/SpeechRecognition/)

[Features](#-key-features) • [Demo](#-demo) • [How It Works](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 🎯 What is BlinkLink?

BlinkLink is an **innovative multimodal assistive technology system** designed to help individuals with varying degrees of motor disabilities control their environment using **multiple interaction methods**:

- 👁️ **Eye Blinks** - For users with severe motor limitations
- 🎤 **Voice Commands** - For users who can speak but have limited mobility
- 🔐 **Facial Recognition** - Biometric security for all users

By combining cutting-edge **Computer Vision**, **Speech Recognition**, **Machine Learning**, and **IoT technology**, BlinkLink offers flexible control options that adapt to each user's unique abilities.

### The Problem We're Solving

Millions of people worldwide live with conditions like ALS, cerebral palsy, stroke, or spinal cord injuries that affect their motor control and speech in different ways. Some can speak but cannot move, others can move slightly but cannot speak, and some have both limitations.

**BlinkLink's multimodal approach provides:**

- ✅ **Multiple input methods** to suit different abilities
- ✅ **Non-invasive technology** using only webcam and microphone
- ✅ **Affordable solution** with standard hardware
- ✅ **Privacy-focused** with local processing

---

## 🌟 Key Features

<table>
<tr>
<td width="33%">

### 🔐 Biometric Security

- **Facial Recognition** ensures only authorized users can access the system
- Custom-trained TensorFlow model for accurate identification
- Privacy-focused: All processing happens locally on your device
- Multi-user support with individual profiles

</td>
<td width="33%">

### 👁️ Eye Blink Detection

- Ultra-responsive detection using **Dlib's 68-point facial landmarks**
- **Eye Aspect Ratio (EAR) algorithm** for accurate blink recognition
- Configurable sensitivity and thresholds
- Ideal for users with severe motor limitations

</td>
<td width="33%">

### 🎤 Voice Command Control

- **Speech recognition** using Google Speech API
- Natural language processing for intuitive commands
- Works in noisy environments with noise filtering
- Support for multiple languages (expandable)

</td>
</tr>
<tr>
<td width="33%">

### 🌐 Wireless IoT Control

- Lightweight **MQTT protocol** for instant device communication
- Control virtually any electronic appliance via **ESP32**
- Expandable to multiple devices and rooms
- Real-time command execution

</td>
<td width="33%">

### 🧩 Multimodal Architecture

- **Adaptive input selection** based on user preference
- Seamless switching between blink and voice modes
- Parallel processing of vision and audio streams
- Thread-safe command queue management

</td>
<td width="33%">

### 🔧 Modular & Extensible

- Clean, **object-oriented Python codebase**
- Separate modules for each modality
- Easy to add new input methods (gestures, head tracking, etc.)
- Plugin-based architecture for future expansion

</td>
</tr>
</table>

---

## 🎬 Demo

#### **Vision Mode (Blink Control):**

1. 📹 **Face Detection**: System identifies and tracks your face
2. 🔐 **Authentication**: Facial recognition verifies your identity
3. ✅ **Authorization**: Green box indicates you're cleared
4. 👁️ **Blink Command**: Eye blink triggers device control

#### **Voice Mode (Speech Control):**

1. 🎤 **Voice Activation**: Say "turn on light" or "turn off fan"
2. 🧠 **Speech Processing**: System recognizes and parses command
3. 📡 **MQTT Publish**: Command sent to IoT device
4. 💡 **Device Response**: Appliance responds instantly

---

## 🔬 How It Works

<!-- <div align="center">

![Multimodal System Architecture](architecture_multimodal.png)

</div> -->

### The Multimodal Processing Pipeline

```
👤 User Input (Blink/Voice) → 🎥 Capture (Camera/Mic) → 🧠 AI Processing → 📡 MQTT → 💡 Device Control
```

### System Components

#### 1. **Vision Controller (Python/PC)** 🖥️

- **Face Authentication Module** ([`face_auth.py`](vision_controller/face_auth.py))

  - TensorFlow-based facial recognition
  - User verification and authorization
  - Multi-user profile support

- **Blink Detection Module** ([`blink_detector.py`](vision_controller/blink_detector.py))
  - Eye Aspect Ratio (EAR) calculation
  - Dlib 68-point facial landmarks
  - Real-time blink event detection

#### 2. **Voice Controller (Python/PC)** 🎤

- **Voice Command Module** ([`voice_command.py`](vision_controller/voice_command.py))
  - Speech-to-text using Google Speech Recognition
  - Natural language command parsing
  - Background noise filtering
  - Multi-threaded audio processing

#### 3. **Command Orchestrator** 🎯

- **Main Application** ([`main.py`](main.py))
  - Parallel processing of vision and audio streams
  - Thread-safe command queue management
  - State management (authorized/unauthorized)
  - Unified MQTT publishing

#### 4. **MQTT Broker (Cloud/Local)** ☁️

- Central message hub (HiveMQ, Mosquitto, etc.)
- Routes commands from multiple input modalities
- Supports multiple device subscriptions

#### 5. **IoT Device (ESP32)** 🔌

- MQTT subscriber for device control
- WiFi connectivity
- Relay control for appliances
- Serial debugging output

---

## 🛠️ Technology Stack

<div align="center">

|      Category      | Technologies                                                                                                                                                                                                                                                                                                                                                    |
| :----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **AI & Vision**   | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Dlib](https://img.shields.io/badge/Dlib-008000?style=flat) |
| **Speech & Audio** | ![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-FF0000?style=flat) ![PyAudio](https://img.shields.io/badge/PyAudio-blue?style=flat) ![Google Speech API](https://img.shields.io/badge/Google_Speech-4285F4?style=flat&logo=google&logoColor=white)                                                                                          |
| **IoT & Hardware** | ![ESP32](https://img.shields.io/badge/ESP32-000000?style=flat&logo=espressif&logoColor=white) ![Arduino](https://img.shields.io/badge/Arduino-00979D?style=flat&logo=arduino&logoColor=white)                                                                                                                                                                   |
| **Communication**  | ![MQTT](https://img.shields.io/badge/MQTT-660066?style=flat&logo=mqtt&logoColor=white) ![WiFi](https://img.shields.io/badge/WiFi-0099CC?style=flat)                                                                                                                                                                                                             |
|  **Development**   | ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=flat&logo=visual-studio-code&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)                                                                                                                                                                  |

</div>

---

## 🚀 Installation

### Prerequisites

**Hardware:**

- 📹 Standard USB Webcam (for vision control)
- 🎤 Microphone (for voice control)
- 🔧 ESP32 DevKit V1 Board
- ⚡ 5V Relay Module
- 🔌 USB Cable & Jumper Wires

**Software:**

- 🐍 Python 3.9 or higher
- 🔧 Arduino IDE 1.8.13+
- 💻 Windows/Linux/macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/BlinkLink.git
cd BlinkLink
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
   python vision_controller/collect_and_encode.py
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
   python main.py
   ```

2. **Monitor Output:**

   - Watch the terminal for logs on face detection, authentication, and device control.

3. **Interact with IoT Devices:**
   - Use configured eye blink gestures or voice commands to control connected IoT devices (e.g., turn on lights, fans).

---

## 🤝 Contributing

We welcome contributions to BlinkLink! Please follow these steps:

1. **Fork the repository**
2. **Create a new branch**: `git checkout -b feature/YourFeature`
3. **Make your changes**
4. **Commit your changes**: `git commit -m 'Add some feature'`
5. **Push to the branch**: `git push origin feature/YourFeature`
6. \*\*Open a Pull Request`

Please ensure your code follows the existing style and includes appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For questions or feedback, please reach out:

- **Your Name** - [sofi.sidik12@gmail.com](mailto:sofi.sidik12@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/ahmad-sofi-sidik/](https://www.linkedin.com/in/ahmad-sofi-sidik/)

---

Thank you for your interest in BlinkLink! Together, let's make technology accessible to everyone.
