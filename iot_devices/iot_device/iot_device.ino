#include <WiFi.h>
#include <PubSubClient.h>
#include <IRremote.h>

// --- KONFIGURASI WIFI & MQTT ---
const char* ssid = "BROH";
const char* password = "12345678";
const char* mqtt_server = "broker.hivemq.com";
const int mqtt_port = 1883;

// --- BARU: Topik MQTT Terpisah ---
const char* TOPIC_LAMPU = "iot/commands/lampu";
const char* TOPIC_TV = "iot/commands/tv";

// NEC Address dari STB
#define STB_ADDRESS 0x7B85

// Mapping command tombol
#define CMD_CH_UP     0x81
#define CMD_CH_DOWN   0x88
#define CMD_VOL_UP    0x82
#define CMD_VOL_DOWN  0x90

// --- KONFIGURASI PIN ---
#define RELAY_PIN 2         // Pin untuk Relay (Lampu)
#define IR_SENDER_PIN 4     // Pin untuk LED IR (Sender)

// --- Inisialisasi Klien ---
WiFiClient espClient;
PubSubClient client(espClient);

// --- Fungsi Helper untuk Relay ---
void toggleRelay() {
  int current_status = digitalRead(RELAY_PIN);
  digitalWrite(RELAY_PIN, !current_status);
  Serial.println("Aksi: Status Relay (Lampu) diubah!");
}

// --- FUNGSI UNTUK MENGIRIM SINYAL IR (DIPERBAIKI) ---
// Perubahan: IRType diganti menjadi decode_type_t untuk kompatibilitas IRremote v3/v4
void sendIrCode(unsigned long command, decode_type_t protocol, uint8_t nbits = 0) {
  Serial.print("Mengirim Command IR: 0x");
  Serial.print(command, HEX);
  Serial.print(" ke Alamat: 0x");
  Serial.print(STB_ADDRESS, HEX);
  Serial.print(" (Protokol: ");
  Serial.print(getProtocolString(protocol));
  Serial.println(")");

  switch (protocol) {
    case NEC:
      // Format Baru Library v3+: sendNEC(Address, Command, Repeats)
      // Kita menggunakan STB_ADDRESS yang sudah didefinisikan di atas
      IrSender.sendNEC(STB_ADDRESS, (uint8_t)command, 0); 
      break;
      
    // Tambahkan case lain jika perlu (SAMSUNG, SONY, dll)
    default:
      Serial.println("Error: Protokol belum diimplementasikan di switch-case.");
      break;
  }
  delay(100);
}

// --- FUNGSI CALLBACK UTAMA ---
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.println("---");
  Serial.print("Pesan diterima di topik: ");
  Serial.println(topic);

  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.print("Isi Pesan: ");
  Serial.println(message);

  // --- LOGIKA UTAMA ---

  // 1. Logika untuk Topik LAMPU
  if (strcmp(topic, TOPIC_LAMPU) == 0) {
    if (message == "TOGGLE") {
      toggleRelay();
    }
  }
  
  // 2. Logika untuk Topik TV
  else if (strcmp(topic, TOPIC_TV) == 0) {
    // nbits tidak terlalu krusial di fungsi baru sendNEC, tapi kita biarkan 32
    if (message == "ch_up") {
      sendIrCode(CMD_CH_UP, NEC, 32);
    } else if (message == "ch_down") {
      sendIrCode(CMD_CH_DOWN, NEC, 32);
    } else if (message == "vol_plus") {
      sendIrCode(CMD_VOL_UP, NEC, 32);
    } else if (message == "vol_min") {
      sendIrCode(CMD_VOL_DOWN, NEC, 32);
    }
  }
  
  else {
    Serial.println("Topik tidak dikenal.");
  }
}

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
  Serial.println("\n✅ WiFi terhubung!");
  Serial.print("Alamat IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Mencoba koneksi ulang ke MQTT Broker...");
    if (client.connect("ESP32_IR_Blaster_UniqueId")) { // Client ID sebaiknya unik
      Serial.println("✅ terhubung!");
      
      client.subscribe(TOPIC_LAMPU);
      client.subscribe(TOPIC_TV);
      Serial.println("Subscribe ke topik lampu dan TV.");

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
  digitalWrite(RELAY_PIN, HIGH); // Default OFF untuk relay active-low (sesuaikan jika active-high)
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // Inisialisasi IR Sender
  IrSender.begin(IR_SENDER_PIN); 
  Serial.println("Pengirim IR siap.");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}