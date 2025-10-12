#include <WiFi.h>
#include <PubSubClient.h>

// --- KONFIGURASI (UBAH SESUAI KEBUTUHAN ANDA) ---
const char* ssid = "BROH";         // Ganti dengan nama WiFi Anda
const char* password = "12345678";  // Ganti dengan password WiFi Anda

const char* mqtt_server = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_topic = "iot/commands/eyeblink";

// Pin untuk mengontrol relay. LED_BUILTIN (biasanya pin 2) digunakan untuk testing.
#define RELAY_PIN 2 // LED_BUILTIN

// --- Inisialisasi Klien ---
WiFiClient espClient;
PubSubClient client(espClient);
long lastMsg = 0;
char msg[50];
int value = 0;

// // Fungsi ini dipanggil setiap kali ada pesan masuk dari MQTT
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Pesan diterima di topik: ");
  Serial.println(topic);
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.print("Isi Pesan: ");
  Serial.println(message);

  // Jika pesan yang diterima adalah "TOGGLE", ubah status relay/LED
  if (message == "TOGGLE") {
    // Baca status pin saat ini (HIGH atau LOW)
    int current_status = digitalRead(RELAY_PIN);
    // Balik statusnya
    digitalWrite(RELAY_PIN, !current_status); 
    Serial.println("Aksi: Status Relay/LED diubah!");
  }
}

// Fungsi untuk menghubungkan ke WiFi
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Menghubungkan ke WiFi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("✅ WiFi terhubung!");
  Serial.print("Alamat IP: ");
  Serial.println(WiFi.localIP());
}

// Fungsi untuk re-koneksi jika koneksi MQTT terputus
void reconnect() {
  while (!client.connected()) {
    Serial.print("Mencoba koneksi ulang ke MQTT Broker...");
    // Coba untuk terhubung
    if (client.connect("ESP32_BlinkClient")) {
      Serial.println("✅ terhubung!");
      // Setelah terhubung, subscribe ke topik yang diinginkan
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("gagal, rc=");
      Serial.print(client.state());
      Serial.println(" Coba lagi dalam 5 detik");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Pastikan relay/LED mati saat awal
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback); // Mengatur fungsi 'callback' untuk menangani pesan
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop(); // Perintah ini penting untuk menjaga koneksi dan mendengarkan pesan
}