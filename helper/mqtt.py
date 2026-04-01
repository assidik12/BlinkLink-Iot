"""
Async MQTT Handler (Publish & Subscribe)
"""
import paho.mqtt.client as mqtt
from threading import Thread
from queue import Queue
import time

class MQTTClientHandler:
    """⚡ Handler MQTT non-blocking (Publish & Subscribe)"""
    
    def __init__(self, broker, port, on_message_callback=None):
        self.message_queue = Queue()
        self.client = None
        self.running = False
        self.broker = broker
        self.port = port
        self.on_message_callback = on_message_callback
        
    def connect(self):
        """Setup MQTT client"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("✅ MQTT Broker terhubung!")
            else:
                print(f"❌ MQTT Gagal: {rc}")
        
        def on_message(client, userdata, msg):
            if self.on_message_callback:
                try:
                    payload = msg.payload.decode()
                    self.on_message_callback(msg.topic, payload)
                except Exception as e:
                    print(f"⚠️ Error processing MQTT message: {e}")

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"❌ MQTT Error: {e}")
            return False
    
    def subscribe(self, topic):
        """Subscribe to a topic"""
        if self.client:
            self.client.subscribe(topic)
            print(f"👂 Subscribed to: {topic}")

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
                        # print(f"📤 MQTT Published: {message} → {topic}") # Opsional: kurangi log
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