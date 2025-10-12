# ===================================================
# Fase 1: Setup & Impor Library
# ===================================================
import paho.mqtt.client as mqtt
import time

print("Library berhasil diimpor.")

def on_connect_device(client, userdata, flags, rc):
    """Callback koneksi untuk PERANGKAT."""
    if rc == 0:
        print("[DEVICE] Berhasil terhubung ke Broker MQTT!")
        client.subscribe("iot/commands/eyeblink")
        print("[DEVICE] Mendengarkan perintah di topik 'iot/commands/eyeblink'...")
    else:
        print(f"[DEVICE] Gagal terhubung, kode status: {rc}")

def on_message_device(client, userdata, msg):
    """Callback pesan untuk PERANGKAT."""
    command = msg.payload.decode()
    print(f"[DEVICE] Menerima Perintah: {command}")

    if command == "LIGHT/ON":
        print("💡 [DEVICE] Status Lampu: MENYALA")
    elif command == "LIGHT/OFF":
        print("🔌 [DEVICE] Status Lampu: MATI")

def run_iot_device():
    """Menjalankan perangkat IoT virtual."""
    print("\n--- Menjalankan Perangkat IoT Virtual ---")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "BCI_Device")
    client.on_connect = on_connect_device
    client.on_message = on_message_device
    broker_address = "broker.hivemq.com"
    client.connect(broker_address, 1883, 60)

    client.loop_start()
    try:
        # Biarkan loop berjalan di background selama 60 detik untuk simulasi
        time.sleep(80)
    except KeyboardInterrupt:
        print("[DEVICE] Simulasi perangkat dihentikan.")
    finally:
        # Hentikan loop dan putuskan koneksi saat selesai
        client.loop_stop()
        client.disconnect()
        print("[DEVICE] Koneksi diputus.")


run_iot_device()