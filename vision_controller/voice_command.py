import speech_recognition as sr
import threading
import time

# Daftar kata kunci pemicu untuk berbagai perintah
# Menggunakan set untuk pencarian yang lebih cepat
KEYWORDS_TOGGLE = {"lampu", "toggle"}
ACTIVATE_KEYWORDS = {"halo", "hai", "bro"}
KEYWORDS_ON = {"nyala", "nyalakan", "hidupkan"}
KEYWORDS_OFF = {"mati", "matikan", "padamkan"}

class VoiceCommandHandler:
    """
    Menangani pengenalan perintah suara secara asynchronous di thread terpisah.
    """
    
    def __init__(self, command_queue):
        """
        Inisialisasi recognizer, microphone, dan antrian perintah.
        
        Args:
            command_queue (queue.Queue): Antrian untuk mengirim perintah  yang terdeteksi ke thread utama.
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = command_queue
        self.stop_event = threading.Event()
        
        # Atur sensitivitas recognizer
        self.recognizer.energy_threshold = 4000  # Default: 300. Naikkan jika terlalu sensitif.
        self.recognizer.dynamic_energy_threshold = True

        print("🎤 Menginisialisasi modul suara...")
        # Lakukan kalibrasi sekali di awal
        with self.microphone as source:
            print("   Menyesuaikan dengan suara sekitar (1 detik)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Modul suara siap.")

    def _listen_in_background(self):
        """
        Fungsi privat yang berjalan di thread. 
        Mendengarkan secara terus-menerus.
        """
        while not self.stop_event.is_set():
            try:
                with self.microphone as source:
                    # Mendengarkan audio dari mikrofon
                    # timeout=5: Berhenti mendengarkan jika 5 detik tidak ada suara
                    # phrase_time_limit=4: Berhenti merekam setelah 4 detik suara
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)
                
                # Gunakan Google Speech Recognition untuk menerjemahkan audio
                # Kita set bahasa ke "id-ID" untuk Bahasa Indonesia
                text = self.recognizer.recognize_google(audio, language="id-ID")
                print(f"🎧 Perintah suara terdeteksi: '{text}'")
                
                # Proses teks yang dikenali untuk mencari perintah
                self._process_command(text.lower())

            except sr.WaitTimeoutError:
                # Ini normal, terjadi jika tidak ada suara dalam 5 detik.
                # Loop akan berlanjut.
                pass
            except sr.UnknownValueError:
                # Terjadi jika audio terdeteksi tapi tidak bisa dipahami.
                print("... (Audio tidak dapat dipahami)")
                pass
            except sr.RequestError as e:
                # Terjadi jika ada masalah dengan API Google Speech
                print(f"Error layanan Google Speech: {e}")
                time.sleep(2) # Beri jeda sebelum mencoba lagi
            except Exception as e:
                print(f"Error di thread suara: {e}")
                
    def _process_command(self, text):
        """
        Menganalisis teks dan memasukkan perintah yang valid ke dalam antrian.
        """
        words = set(text.split())
        
        # Cek perintah OFF
        if KEYWORDS_OFF.intersection(words):
            print("   Perintah 'LIGHT_OFF' terdeteksi.")
            self.command_queue.put("LIGHT_OFF")
        # Cek perintah ON
        elif KEYWORDS_ON.intersection(words):
            print("   Perintah 'LIGHT_ON' terdeteksi.")
            self.command_queue.put("LIGHT_ON")
        # Cek perintah TOGGLE
        elif KEYWORDS_TOGGLE.intersection(words):
            print("   Perintah 'LIGHT_TOGGLE' terdeteksi.")
            self.command_queue.put("LIGHT_TOGGLE")

    def start(self):
        """
        Memulai thread untuk mendengarkan di latar belakang.
        """
        # Buat thread baru yang menargetkan fungsi _listen_in_background
        self.listener_thread = threading.Thread(target=self._listen_in_background)
        # daemon=True berarti thread ini akan otomatis berhenti 
        # saat program utama (main.py) berhenti.
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def stop(self):
        """
        Mengirim sinyal untuk menghentikan thread.
        """
        print("🛑 Menghentikan thread perintah suara...")
        self.stop_event.set()