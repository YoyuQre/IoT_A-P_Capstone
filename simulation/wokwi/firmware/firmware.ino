/*
 * IoT Device Behaviour Fingerprinting - Wokwi ESP32 Firmware
 * 
 * Demonstrates:
 * - Device identity (unique ID per device category)
 * - Periodic sensor telemetry (simulated DHT22 readings)
 * - Normal communication behaviour (MQTT-like JSON payloads)
 * - Anomaly simulation via button press (traffic spike / destination expansion)
 * 
 * This is a DESIGN ARTIFACT for the Wokwi simulation layer.
 * The actual statistical pipeline runs in Python (see src/simulator.py).
 */

#include <Arduino.h>
#include <DHT.h>

// ============================================================
// DEVICE CONFIGURATION (modify per device instance)
// ============================================================
const char* DEVICE_ID = "bulb_01";        // Unique device ID
const char* DEVICE_TYPE = "Smart Bulb";   // Category: Smart Bulb, Smart Camera, Smart Lock, Smart Meter, Smart TV
const char* FIRMWARE_VERSION = "1.0.0";

// ============================================================
// HARDWARE PINS
// ============================================================
#define DHT_PIN 15
#define LED_STATUS_PIN 21
#define LED_ANOMALY_PIN 22
#define BTN_ANOMALY_PIN 23

// ============================================================
// TIMING CONFIGURATION
// ============================================================
const unsigned long TELEMETRY_INTERVAL_MS = 5000;  // 5 seconds (faster for demo)
const unsigned long ANOMALY_DURATION_MS = 30000;   // 30 seconds anomaly window

// ============================================================
// ANOMALY STATE
// ============================================================
enum AnomalyMode {
  MODE_NORMAL = 0,
  MODE_TRAFFIC_SPIKE = 1,
  MODE_DEST_EXPANSION = 2,
  MODE_COMBINED = 3
};

AnomalyMode currentMode = MODE_NORMAL;
unsigned long anomalyStartTime = 0;
bool anomalyActive = false;
bool lastButtonState = HIGH;

// ============================================================
// SENSOR
// ============================================================
DHT dht(DHT_PIN, DHT22);

// ============================================================
// SIMULATED TELEMETRY GENERATION
// ============================================================
struct TelemetryData {
  float temperature;
  float humidity;
  unsigned long packetCount;
  unsigned int destinationCount;
  unsigned int packetSize;
  const char* protocol;
};

TelemetryData generateTelemetry() {
  TelemetryData data;
  
  // Read actual sensor (or simulate if not available)
  data.temperature = dht.readTemperature();
  data.humidity = dht.readHumidity();
  
  // Fallback to simulated values if sensor fails
  if (isnan(data.temperature)) data.temperature = 22.0 + random(-50, 50) / 10.0;
  if (isnan(data.humidity)) data.humidity = 50.0 + random(-200, 200) / 10.0;
  
  // Base characteristics per device type (synthetic simulation parameters)
  if (strcmp(DEVICE_TYPE, "Smart Bulb") == 0) {
    data.packetSize = 100;
    data.destinationCount = 1;
    data.protocol = "MQTT";
    data.packetCount = anomalyActive ? random(50, 100) : random(1, 5);  // Spike: 10x
  } else if (strcmp(DEVICE_TYPE, "Smart Camera") == 0) {
    data.packetSize = 1200;
    data.destinationCount = 2;
    data.protocol = "RTSP";
    data.packetCount = anomalyActive ? random(200, 500) : random(10, 50);
  } else if (strcmp(DEVICE_TYPE, "Smart Lock") == 0) {
    data.packetSize = 64;
    data.destinationCount = 1;
    data.protocol = "CoAP";
    data.packetCount = anomalyActive ? random(10, 20) : random(0, 1);  // Very low rate
  } else if (strcmp(DEVICE_TYPE, "Smart Meter") == 0) {
    data.packetSize = 200;
    data.destinationCount = 1;
    data.protocol = "HTTP";
    data.packetCount = anomalyActive ? random(20, 50) : random(1, 5);
  } else if (strcmp(DEVICE_TYPE, "Smart TV") == 0) {
    data.packetSize = 1000;
    data.destinationCount = anomalyActive ? random(6, 12) : random(2, 6);  // Dest expansion
    data.protocol = "HTTPS";
    data.packetCount = anomalyActive ? random(200, 500) : random(10, 50);
  } else {
    data.packetSize = 100;
    data.destinationCount = 1;
    data.protocol = "MQTT";
    data.packetCount = 1;
  }
  
  // Combined anomaly affects both rate and destinations
  if (currentMode == MODE_COMBINED && anomalyActive) {
    data.packetCount = data.packetCount * 2;
    data.destinationCount = data.destinationCount * 2;
  }
  
  return data;
}

// ============================================================
// JSON PAYLOAD GENERATION (simulated MQTT message)
// ============================================================
String buildJsonPayload(const TelemetryData& data) {
  String json = "{";
  json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  json += "\"device_type\":\"" + String(DEVICE_TYPE) + "\",";
  json += "\"timestamp\":" + String(millis()) + ",";
  json += "\"temperature\":" + String(data.temperature, 1) + ",";
  json += "\"humidity\":" + String(data.humidity, 1) + ",";
  json += "\"packet_rate\":" + String(data.packetCount) + ",";
  json += "\"packet_size\":" + String(data.packetSize) + ",";
  json += "\"destination_count\":" + String(data.destinationCount) + ",";
  json += "\"protocol\":\"" + String(data.protocol) + "\",";
  json += "\"anomaly_mode\":" + String(currentMode);
  json += "}";
  return json;
}

// ============================================================
// ANOMALY TRIGGER (button press cycles through modes)
// ============================================================
void checkAnomalyButton() {
  bool buttonState = digitalRead(BTN_ANOMALY_PIN);
  
  // Detect falling edge (button press)
  if (lastButtonState == HIGH && buttonState == LOW) {
    if (!anomalyActive) {
      // Cycle through anomaly modes
      currentMode = static_cast<AnomalyMode>((currentMode + 1) % 4);
      anomalyActive = true;
      anomalyStartTime = millis();
      
      Serial.print("[ANOMALY TRIGGERED] Mode: ");
      switch (currentMode) {
        case MODE_TRAFFIC_SPIKE: Serial.println("TRAFFIC_SPIKE"); break;
        case MODE_DEST_EXPANSION: Serial.println("DESTINATION_EXPANSION"); break;
        case MODE_COMBINED: Serial.println("COMBINED"); break;
        default: Serial.println("NORMAL"); break;
      }
    }
  }
  lastButtonState = buttonState;
  
  // Auto-expire anomaly after duration
  if (anomalyActive && (millis() - anomalyStartTime > ANOMALY_DURATION_MS)) {
    anomalyActive = false;
    currentMode = MODE_NORMAL;
    Serial.println("[ANOMALY EXPIRED] Returned to NORMAL mode");
  }
}

// ============================================================
// LED INDICATORS
// ============================================================
void updateIndicators() {
  // Status LED: heartbeat
  digitalWrite(LED_STATUS_PIN, (millis() / 500) % 2);
  
  // Anomaly LED: solid when anomaly active
  digitalWrite(LED_ANOMALY_PIN, anomalyActive ? HIGH : LOW);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(100);
  
  // Initialize pins
  pinMode(LED_STATUS_PIN, OUTPUT);
  pinMode(LED_ANOMALY_PIN, OUTPUT);
  pinMode(BTN_ANOMALY_PIN, INPUT_PULLUP);
  
  // Initialize sensor
  dht.begin();
  
  // Print device identity
  Serial.println("========================================");
  Serial.println("IoT Device Behaviour Fingerprinting Demo");
  Serial.println("========================================");
  Serial.print("Device ID: "); Serial.println(DEVICE_ID);
  Serial.print("Device Type: "); Serial.println(DEVICE_TYPE);
  Serial.print("Firmware: "); Serial.println(FIRMWARE_VERSION);
  Serial.println("Press button to trigger anomaly modes:");
  Serial.println("  1) Traffic Spike");
  Serial.println("  2) Destination Expansion");
  Serial.println("  3) Combined");
  Serial.println("  4) Normal (auto-reset after 30s)");
  Serial.println("========================================");
  
  // Initial LED state
  digitalWrite(LED_STATUS_PIN, LOW);
  digitalWrite(LED_ANOMALY_PIN, LOW);
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  static unsigned long lastTelemetry = 0;
  
  checkAnomalyButton();
  updateIndicators();
  
  // Periodic telemetry generation
  if (millis() - lastTelemetry >= TELEMETRY_INTERVAL_MS) {
    lastTelemetry = millis();
    
    TelemetryData data = generateTelemetry();
    String payload = buildJsonPayload(data);
    
    // Output to serial (simulates MQTT publish)
    Serial.println(payload);
  }
  
  delay(10);
}