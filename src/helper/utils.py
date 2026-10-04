import cv2
import helper.config as config
import numpy as np
import pygame
import os
import threading
from gtts import gTTS

class SoundManager:
    """Manages sound feedback generation and playback (Beeps + Voice)."""
    
    def __init__(self):
        self.sample_rate = 44100
        # Initialize mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
        
        self.sounds = {}
        self.voice_path = "assets/sounds"
        if not os.path.exists(self.voice_path):
            os.makedirs(self.voice_path)

        self._generate_default_sounds()
        
        # Define voice assets mapping
        self.voice_assets = {
            # Modes
            "mode_lampu": "Mode Lampu Aktif",
            "mode_tv": "Mode TV Aktif",
            "mode_musik": "Mode Musik Aktif",
            # Actions
            "lamp_on": "Lampu Menyala",
            "lamp_off": "Lampu Mati",
            "music_play": "Musik Diputar",
            "music_pause": "Musik Dijeda",
            "sos_alert": "Bahaya! Sinyal SOS Terdeteksi",
            "tv_action": "Aksi TV",
            # System
            "auth_success": "Akses Diterima"
        }
        
        # Start generating voice assets in background to avoid blocking UI
        threading.Thread(target=self._ensure_voice_assets, daemon=True).start()

    def _generate_default_sounds(self):
        """Generates default beep sounds."""
        # Mode Switch: High pitch short beep
        self.sounds['mode_switch'] = self._create_beep(frequency=880, duration=0.1)
        # Action Trigger: Medium pitch beep
        self.sounds['action_trigger'] = self._create_beep(frequency=440, duration=0.2)
        # Progress Feedback: Low pitch very short beep (optional)
        self.sounds['progress_tick'] = self._create_beep(frequency=220, duration=0.05)

    def _ensure_voice_assets(self):
        """Checks and generates missing voice files using gTTS."""
        print("🔊 Memeriksa aset suara...")
        for key, text in self.voice_assets.items():
            file_path = os.path.join(self.voice_path, f"{key}.mp3")
            if not os.path.exists(file_path):
                try:
                    print(f"   ⏳ Membuat suara: '{text}'...")
                    tts = gTTS(text=text, lang='id', slow=False)
                    tts.save(file_path)
                    print(f"   ✅ Tersimpan: {file_path}")
                except Exception as e:
                    print(f"   ❌ Gagal membuat suara '{key}': {e}")
        print("🔊 Aset suara siap.")

    def _create_beep(self, frequency=440, duration=0.1):
        """Creates a beep sound as a pygame Sound object."""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        
        # Generate sine wave
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit signed integer (standard audio format)
        audio = (wave * 32767).astype(np.int16)
        
        # Adjust for mixer channels
        channels = pygame.mixer.get_init()[2] if pygame.mixer.get_init() else 1
        if channels == 2:
            # Duplicate mono signal for stereo
            audio = np.column_stack((audio, audio))
        
        return pygame.sndarray.make_sound(audio)

    def play(self, sound_name):
        """Plays a pre-loaded beep sound."""
        if sound_name in self.sounds:
            self.sounds[sound_name].play()

    def play_voice(self, asset_key):
        """Plays a voice file (MP3) using mixer.music."""
        file_path = os.path.join(self.voice_path, f"{asset_key}.mp3")
        if os.path.exists(file_path):
            try:
                # Use mixer.music for MP3 streaming
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Error playing voice {asset_key}: {e}")
        else:
            # Fallback to beep if voice not ready
            print(f"Voice file missing: {asset_key}")
            self.play('action_trigger')

    def play_frequency(self, frequency, duration=0.1):
        """Generates and plays a custom frequency on the fly."""
        sound = self._create_beep(frequency, duration)
        sound.play()


class Utils:
    """Utility helpers for CV + UI operations.

    Provides methods:
    - draw_text_center(surface, text, font, color, rect)
    - resize_frame_for_detection(frame, scale=None)
    - scale_rect(rect, scale)
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    def draw_text_center(self, surface, text, font, color, rect):
        """Render text centered inside a pygame rect on the given surface."""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)

    def resize_frame_for_detection(self, frame, scale=None):
        """Resize frame for faster detection. If scale is None use config.DETECTION_SCALE."""
        if scale is None:
            scale = getattr(self.cfg, "DETECTION_SCALE", 0.5)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return small, scale

    def scale_rect(self, rect, scale):
        """Scale a dlib rectangle from scaled detection back to original frame coordinates."""
        # Using a dummy object since dlib might not be imported here
        class DlibRectLike:
             def __init__(self, l, t, r, b): self.l, self.t, self.r, self.b = l, t, r, b
             def left(self): return self.l
             def top(self): return self.t
             def right(self): return self.r
             def bottom(self): return self.b
             def width(self): return self.r - self.l
             def height(self): return self.b - self.t
        
        return DlibRectLike(
            int(rect.left() / scale),
            int(rect.top() / scale),
            int(rect.right() / scale),
            int(rect.bottom() / scale)
        )


# Singleton instance for convenience
util = Utils(config)

# Backward-compatible function wrappers
def draw_text_center(surface, text, font, color, rect):
    return util.draw_text_center(surface, text, font, color, rect)


def resize_frame_for_detection(frame, scale=None):
    return util.resize_frame_for_detection(frame, scale)


def scale_rect(rect, scale):
    return util.scale_rect(rect, scale)
