def Connect_mqtt(mqtt, config):
    """Menyiapkan dan menghubungkan client MQTT."""
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ Berhasil terhubung ke MQTT Broker!")
        else:
            print(f"❌ Gagal terhubung ke MQTT Broker, return code {rc}")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    try:
        client.connect(config.MQTT_BROKER, config.MQTT_PORT)
        client.loop_start()
        return client
    except Exception as e:
        print(f"❌ Tidak bisa terhubung ke broker MQTT: {e}")
        return None
