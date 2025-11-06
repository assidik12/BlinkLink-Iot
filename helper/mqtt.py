"""
Async MQTT Publisher untuk non-blocking message publishing
"""
import paho.mqtt.client as mqtt
from threading import Thread
from queue import Queue
import time


class MQTTPublisher:
    """⚡ Handler MQTT non-blocking menggunakan queue & thread"""
    
    def __init__(self, broker, port):
        self.message_queue = Queue()
        self.client = None
        self.running = False
        self.broker = broker
        self.port = port
        
    def connect(self):
        """Setup MQTT client"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("✅ MQTT Broker terhubung!")
            else:
                print(f"❌ MQTT Gagal: {rc}")
        
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = on_connect
        
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"❌ MQTT Error: {e}")
            return False
    
    def start_publisher_thread(self):
        """⚡ Jalankan thread untuk publish async"""
        self.running = True
        self.thread = Thread(target=self._publish_worker, daemon=True)
        self.thread.start()
    
    def _publish_worker(self):
        """Worker thread untuk handle publish"""
        while self.running:
            if not self.message_queue.empty():
                topic, message = self.message_queue.get()
                if self.client:
                    try:
                        self.client.publish(topic, message)
                        print(f"📤 MQTT Published: {message} → {topic}")
                    except Exception as e:
                        print(f"⚠️ MQTT Publish Error: {e}")
            time.sleep(0.01)  # Prevent busy waiting
    
    def publish_async(self, topic, message):
        """⚡ Non-blocking publish"""
        self.message_queue.put((topic, message))
    
    def stop(self):
        """Cleanup"""
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()