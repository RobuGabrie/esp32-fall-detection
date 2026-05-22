#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_SSD1306.h>
#include <Protocentral_MAX30205.h>
#include <SparkFun_Bio_Sensor_Hub_Library.h>
#include "esp_task_wdt.h"
#include "driver/gpio.h"

// BLE — built-in to ESP32 Arduino core (Bluedroid). Adds ~50KB RAM and ~80mA when active.
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "esp_bt.h"

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_variables.h"
#include "derived_metrics.h"

// --- Hardware & Config ---
#define I2C_SDA        21
#define I2C_SCL        22
#define SAMPLE_RATE_HZ 100
#define MAX_SAMPLES    500      // 5.0s ring buffer — must cover pre-impact window + settling delay
#define SENSOR_CHANNELS 7

// Fall evaluation timing — total time from trigger to verdict ≈ ML_WAIT_MS + SETTLE_WAIT_MS.
// Must stay safely under MAX_SAMPLES / SAMPLE_RATE_HZ seconds so the impact is still in the ring buffer.
#define ML_WAIT_MS         500   // wait this long after trigger to accumulate post-impact context for ML
#define SETTLE_WAIT_MS     3000  // additional wait before evaluating stillness/orientation gates
#define ML_POST_OFFSET     50    // samples of post-impact context to include in ML window (0.5s)

// --- Indicator Hardware ---
#define LED_GREEN      32
#define LED_YELLOW     33
#define LED_RED        4
#define LED_CH_GREEN   1        // LEDC channel for PWM dimming
#define LED_GREEN_DIM  13       // ~5% duty = idle indicator
#define LED_GREEN_FULL 255
#define BUZZER_PIN     23
#define BUZZER_CH      0
#define BUZZER_FREQ    1700
#define BUZZER_DUTY    128
#define FALL_ALERT_DURATION_MS 10000

// --- OLED & Button ---
#define OLED_ADDR          0x3D
#define OLED_WIDTH         128
#define OLED_HEIGHT        64
#define BUTTON_PIN         27
#define BIO_RESET_PIN      16
#define BIO_MFIO_PIN       17
#define VBAT_PIN           35       // ADC1_CH7
#define VBAT_DIVIDER       2.0f
// Battery runtime estimate. Measured avg draw ~65 mA, spikes ~120 mA (rare, ignored).
// Set BATTERY_CAPACITY_MAH to your pack's actual capacity.
#define BATTERY_CAPACITY_MAH    2000
#define BATTERY_AVG_CURRENT_MA  65
#define DISPLAY_PAGES      9  // 5=stress, 6=debug, 7=posture/sleep, 8=steps/hrv
#define BIO_READ_INTERVAL  1000
#define TEMP_READ_INTERVAL 60000
#define VBAT_READ_INTERVAL 60000
#define DISPLAY_TIMEOUT_MS 60000
#define BTN_DEBOUNCE_MS    200

#define DISPLAY_ALWAYS_ON  0
#define DISPLAY_AOD        1
#define DISPLAY_AOD_UPDATE_MS 5000

// --- BLE ---
// Custom UUIDs — replace with your own if deploying. Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#define BLE_DEVICE_NAME      "FallGuard"
#define BLE_SERVICE_UUID     "f00dbabe-0001-1000-8000-00805f9b34fb"
#define BLE_TELEMETRY_UUID   "f00dbabe-0002-1000-8000-00805f9b34fb"  // notify: 6B status packet
#define BLE_EVENT_UUID       "f00dbabe-0003-1000-8000-00805f9b34fb"  // notify: "FALL" string on confirmed fall
#define BLE_CMD_UUID         "f00dbabe-0005-1000-8000-00805f9b34fb"  // write: commands from phone

#define BLE_TELEMETRY_INTERVAL_MS 500   // 2Hz live stream — lower to avoid flooding the link

// --- Fall Detection Parameters ---
// EMA time constant: alpha=0.02 → ~50 samples (500ms) to track slow orientation drift
#define GRAVITY_ALPHA          0.02f

// Phase 1: Freefall (Path A)
// Bicep stays tucked near the torso during a fall — cleaner freefall signature than wrist
#define FREEFALL_THRESHOLD     4.0f    // m/s²
#define FREEFALL_MIN_SAMPLES   3       // 30ms minimum continuous freefall
#define FREEFALL_TO_IMPACT_MS  2000    // impact must follow within this window

// Phase 2a: Path A — freefall + impact
// Bicep sees lower impact G-forces than wrist (closer to center of mass)
#define IMPACT_ACCEL_MIN       22.0f   // m/s² (2.2g)
#define IMPACT_GYRO_MIN        1.0f    // rad/s

// Phase 2b: Path B — high-G stumble/collapse without freefall; higher threshold to reduce false positives
#define IMPACT_ONLY_ACCEL_MIN  35.0f   // m/s² (3.5g)
#define IMPACT_ONLY_GYRO_MIN   2.0f    // rad/s

// Phase 3: Post-fall posture gates
// Bicep advantage: standing/walking arm is ~vertical; lying-down arm is ~horizontal → large angle shift
#define STILLNESS_MIN          8.0f    // m/s² — wider band tolerates groaning/breathing motion
#define STILLNESS_MAX          11.0f
#define ORIENTATION_MIN_DEG    35.0f   // 45° rejected real sideways falls; 35° still rejects normal arm gestures

#define COOLDOWN_MS            4000

// --- Class Indices ---
// Internal convention (used throughout code, BLE, display, LEDs):
//   0=stationary, 1=walking, 2=running, 3=hand_motion
// Edge Impulse model returns labels ALPHABETICALLY:
//   0=hand_motion, 1=running, 2=stationary, 3=walking
// Remap EI output → internal via EI_TO_INTERNAL below.
#define HAND_MOTION_IDX 3
#define FALL_IDX        0
static const int EI_TO_INTERNAL[4] = { 3, 2, 0, 1 };

#define DEBUG 1

// --- Objects ---
Adafruit_MPU6050 mpu;
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);
MAX30205 tempSensor;
SparkFun_Bio_Sensor_Hub bioHub(BIO_RESET_PIN, BIO_MFIO_PIN);
bioData body;

// =============================================================================
// SHARED STATE — annotated with which task(s) own each field
// =============================================================================

// Ring buffer — Core 1 (sampling_task) writes; Core 0 (background_task) reads.
// Protected by ring_buf_mux for both head_index updates and gravity EMA.
float ring_buffer[MAX_SAMPLES][SENSOR_CHANNELS];
volatile int head_index = 0;
portMUX_TYPE ring_buf_mux = portMUX_INITIALIZER_UNLOCKED;

// I2C mutex — prevents sampling_task and background_task from accessing the
// shared Wire bus simultaneously (MPU6050, SSD1306, MAX30205, MAX32664 all on same bus).
SemaphoreHandle_t i2c_mutex = nullptr;

// Feature buffers — allocated once in setup(); never freed.
// Eliminates the repeated malloc/free that caused heap fragmentation in the original code.
float *activity_features_buf = nullptr;
float *fall_features_buf     = nullptr;

// EMA gravity vector — written by sampling_task under ring_buf_mux; read by background_task.
// Tracks the slow gravity component by filtering out dynamic acceleration.
volatile float gravity_x = 0.0f, gravity_y = 0.0f, gravity_z = 9.81f;

// Trigger handoff — sampling_task sets these atomically, background_task clears trigger_fired.
// trigger_gravity_* captures EMA gravity at the moment of impact for orientation comparison.
volatile bool  trigger_fired       = false;
volatile int   trigger_head_idx    = 0;
volatile float trigger_gravity_x   = 0.0f;
volatile float trigger_gravity_y   = 0.0f;
volatile float trigger_gravity_z   = 9.81f;
volatile float trigger_accMag      = 0.0f;   // accMag at impact (for BLE/log)
volatile float trigger_gyroMag     = 0.0f;   // gyroMag at impact
volatile unsigned long last_trigger_time = 0;

// ---- BLE state ----
BLEServer*         ble_server   = nullptr;
BLECharacteristic* ble_tel_char = nullptr;
BLECharacteristic* ble_evt_char = nullptr;
BLECharacteristic* ble_cmd_char = nullptr;
volatile bool      ble_connected = false;
unsigned long      last_ble_telemetry_send = 0;

// Versioned status packet. v4 layout (19 bytes):
//   [0]   version           = 4
//   [1]   activity          0=stationary 1=walking 2=running 3=hand-motion
//   [2]   fall_state        0=idle 1=fall confirmed (within alert window)
//   [3]   battery_pct
//   [4]   stress_pct
//   [5]   hr                bpm
//   [6]   spo2              %
//   [7]   posture           0=upright 1=reclined 2=lying 3=inverted
//   [8]   sleep_state       0=awake 1=resting 2=light_sleep 3=deep_sleep
//   [9-10] step_count       u16 LE
//   [11]  cadence_spm       steps per minute
//   [12]  hrv_rmssd         HRV (HR-based proxy)
//   [13]  resting_hr        bpm
//   [14]  temp_dc           (temp_C - 30.0) * 10  → decode: temp = byte/10 + 30
//   [15-16] battery_mv      u16 LE — battery voltage in millivolts (e.g. 3870 = 3.87V)
//   [17-18] time_left_min   u16 LE — estimated runtime remaining at typical draw
#define TELEMETRY_PACKET_VERSION 4
struct __attribute__((packed)) TelemetryPacket {
    uint8_t  version;
    uint8_t  activity;
    uint8_t  fall_state;
    uint8_t  battery_pct;
    uint8_t  stress_pct;
    uint8_t  hr;
    uint8_t  spo2;
    uint8_t  posture;
    uint8_t  sleep_state;
    uint16_t step_count;
    uint8_t  cadence_spm;
    uint8_t  hrv_rmssd;
    uint8_t  resting_hr;
    uint8_t  temp_dc;
    uint16_t battery_mv;
    uint16_t time_left_min;
};

// Activity state — written by background_task only
float current_activity_tag[4] = {1.0f, 0.0f, 0.0f, 0.0f};
volatile int current_activity_idx = 0;

// Application state
volatile bool gathering_post_buffer = false;
volatile unsigned long fall_alert_until = 0;

// Display state — background_task only
bool display_on         = false;
int  display_page       = 0;
unsigned long display_sleep_at    = 0;
unsigned long last_display_update = 0;
bool display_aod_active           = false;
unsigned long last_aod_update     = 0;

// Sensor display cache — sampling_task writes (volatile), background_task reads for render
volatile float disp_ax = 0, disp_ay = 0, disp_az = 0;
volatile float disp_gx = 0, disp_gy = 0, disp_gz = 0;
volatile float disp_accMag = 0;
volatile float disp_gyroMag = 0;

// Debug telemetry — shared across both tasks for the debug page (page 5)
volatile bool  dbg_freefall_active = false;
volatile float dbg_min_accMag      = 99.0f;   // lowest accMag seen since boot (freefall depth)
volatile float dbg_peak_accMag     = 0.0f;    // highest accMag seen since boot (impact peak)
volatile float dbg_peak_gyroMag    = 0.0f;
volatile char  dbg_last_path       = '-';
volatile unsigned long dbg_trigger_count = 0;
float dbg_last_conf[3]    = {0, 0, 0};        // background_task-only
float dbg_last_threshold  = 0;
float dbg_last_angle_deg  = 0;
bool  dbg_last_stillness  = false;
int   dbg_last_passing    = 0;                // windows that passed ML threshold
char  dbg_last_verdict[24] = "none";
float disp_temp_c  = 0;
float cached_vbat  = 0;
unsigned long last_temp_read = 0;
unsigned long last_vbat_read = 0;
int  disp_hr = 0, disp_spo2 = 0, disp_hr_conf = 0;
bool bio_ok         = false;
bool bio_has_reading = false;
unsigned long last_bio_read = 0;

// Stress level (0–100) — derived from HR delta vs resting baseline, activity context, SpO2, temp.
// Updated by update_stress_level() after each bio read.
volatile uint8_t stress_level = 0;
float hr_baseline = 0.0f;            // slow EMA of resting HR (only updates while stationary)
uint8_t hr_stress_pct  = 0;          // breakdown components — used by the stress OLED page
uint8_t spo2_stress_pct = 0;
uint8_t temp_stress_pct = 0;

// Button — ISR sets, background_task consumes
volatile bool btn_pressed = false;
unsigned long last_btn_press = 0;

// =============================================================================
// ISR
// =============================================================================
void IRAM_ATTR btn_isr() {
    btn_pressed = true;
}

// =============================================================================
// HELPERS
// =============================================================================

float read_battery_voltage() {
    float vout = analogRead(VBAT_PIN) / 4095.0f * 3.3f;
    return vout * VBAT_DIVIDER;
}

// Minutes of runtime left at typical current draw. Linear in remaining capacity.
int battery_minutes_remaining(int pct) {
    if (pct <= 0) return 0;
    return (int)(((long)pct * BATTERY_CAPACITY_MAH * 60L) / (100L * BATTERY_AVG_CURRENT_MA));
}

int battery_percent(float v) {
    static const float lut_v[]  = { 3.00f, 3.37f, 3.45f, 3.51f, 3.55f, 3.59f, 3.63f,
                                    3.67f, 3.71f, 3.75f, 3.79f, 3.83f, 3.87f, 3.91f,
                                    3.95f, 3.98f, 4.02f, 4.08f, 4.11f, 4.15f, 4.20f };
    static const int  lut_pct[] = {   0,    5,   10,   15,   20,   25,   30,
                                      35,   40,   45,   50,   55,   60,   65,
                                      70,   75,   80,   85,   90,   95,  100 };
    static const int  LUT_LEN = 21;
    if (v <= lut_v[0])           return 0;
    if (v >= lut_v[LUT_LEN - 1]) return 100;
    for (int i = 1; i < LUT_LEN; i++) {
        if (v <= lut_v[i]) {
            float t = (v - lut_v[i-1]) / (lut_v[i] - lut_v[i-1]);
            return (int)(lut_pct[i-1] + t * (lut_pct[i] - lut_pct[i-1]));
        }
    }
    return 100;
}

// Compute a 0–100 stress level from currently available physiological signals.
// Components (weighted):
//   • HR excess over resting baseline (60%) — discounted when expected activity rise
//   • SpO2 below 95% (25%) — oxygen saturation drop indicates physiological strain
//   • Body temperature elevation over 37.0°C (15%) — mild fever / sympathetic load
//
// Baseline HR is learned via slow EMA from stationary samples only — adapts to user
// over ~minutes of rest. If activity is running, HR rise is expected and doesn't count.
void update_stress_level() {
    if (!bio_has_reading) {
        stress_level = 0;
        hr_stress_pct = spo2_stress_pct = temp_stress_pct = 0;
        return;
    }

    // Maintain resting HR baseline — only update while stationary so exercise doesn't pollute it.
    // First-ever reading bootstraps the baseline immediately.
    if (current_activity_idx == 0) {
        if (hr_baseline < 30.0f) hr_baseline = (float)disp_hr;
        else                     hr_baseline = 0.97f * hr_baseline + 0.03f * (float)disp_hr;
    }
    if (hr_baseline < 30.0f) hr_baseline = 70.0f;  // sane default until first stationary read

    // --- HR component (max 60 pts) ---
    float hr_excess = (float)disp_hr - hr_baseline;
    if (hr_excess < 0) hr_excess = 0;
    float hr_norm = hr_excess / 40.0f;     // 40 bpm above baseline = max contribution
    if (hr_norm > 1.0f) hr_norm = 1.0f;
    float activity_factor;
    switch (current_activity_idx) {
        case 0:  activity_factor = 1.0f; break;   // stationary — full weight
        case 1:  activity_factor = 0.3f; break;   // walking — some rise expected
        case 2:  activity_factor = 0.0f; break;   // running — HR rise is normal, not stress
        case 3:  activity_factor = 0.7f; break;   // hand motion (fidgeting can be stress)
        default: activity_factor = 1.0f; break;
    }
    float hr_comp = hr_norm * activity_factor * 60.0f;

    // --- SpO2 component (max 25 pts) ---
    float spo2_comp = 0.0f;
    if (disp_spo2 > 0 && disp_spo2 < 95) {
        float dn = (95.0f - (float)disp_spo2) / 5.0f;  // 90% SpO2 = full contribution
        if (dn > 1.0f) dn = 1.0f;
        spo2_comp = dn * 25.0f;
    }

    // --- Temperature component (max 15 pts) ---
    float temp_comp = 0.0f;
    if (disp_temp_c > 37.0f) {
        float dt = (disp_temp_c - 37.0f) / 1.0f;       // 38°C = full contribution
        if (dt > 1.0f) dt = 1.0f;
        temp_comp = dt * 15.0f;
    }

    float total = hr_comp + spo2_comp + temp_comp;
    if (total < 0)   total = 0;
    if (total > 100) total = 100;
    stress_level    = (uint8_t) total;
    hr_stress_pct   = (uint8_t) hr_comp;
    spo2_stress_pct = (uint8_t) spo2_comp;
    temp_stress_pct = (uint8_t) temp_comp;
}

void update_activity_leds() {
    if (fall_alert_until || gathering_post_buffer) return;
    switch (current_activity_idx) {
        case 0:
            ledcWrite(LED_CH_GREEN, LED_GREEN_DIM);
            digitalWrite(LED_YELLOW, LOW);
            break;
        case 1:
        case 2:
            ledcWrite(LED_CH_GREEN, LED_GREEN_FULL);
            digitalWrite(LED_YELLOW, LOW);
            break;
        case 3:
            ledcWrite(LED_CH_GREEN, 0);
            digitalWrite(LED_YELLOW, HIGH);
            break;
    }
}

// Buzzer: uses vTaskDelay when the scheduler is running, delay() during setup()
void buzzer_beep(uint32_t freq, uint32_t duration_ms) {
    ledcSetup(BUZZER_CH, freq, 8);
    ledcWrite(BUZZER_CH, BUZZER_DUTY);
    if (xTaskGetSchedulerState() == taskSCHEDULER_RUNNING) {
        vTaskDelay(pdMS_TO_TICKS(duration_ms));
    } else {
        delay(duration_ms);
    }
    ledcWrite(BUZZER_CH, 0);
}

void alert_fall() {
    ledcWrite(LED_CH_GREEN, 0);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED,    HIGH);
    buzzer_beep(BUZZER_FREQ, 3000);
    fall_alert_until = millis() + FALL_ALERT_DURATION_MS;
}

// =============================================================================
// BLE — telemetry stream, event notifications, fall log
//
// Architecture: BLE callbacks run on the BT host task. Keep them tiny — only
// flag-setting allowed. All data assembly happens in background_task.
// =============================================================================

class BleConnCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override {
        ble_connected = true;
        Serial.println("[BLE] Client connected");
    }
    void onDisconnect(BLEServer* s) override {
        ble_connected = false;
        Serial.println("[BLE] Client disconnected — restarting advertising");
        s->getAdvertising()->start();
    }
};

class BleCmdCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* c) override {
        std::string v = c->getValue();
        if (v.empty()) return;
        Serial.printf("[BLE] cmd: %.*s\n", (int)v.size(), v.data());
        if (v == "alert_off") {
            fall_alert_until = 0;
            digitalWrite(LED_RED, LOW);
            ledcWrite(BUZZER_CH, 0);
        }
    }
};

void ble_init() {
    BLEDevice::init(BLE_DEVICE_NAME);
    // Request larger MTU so event/log strings can be sent without fragmentation
    BLEDevice::setMTU(247);

    ble_server = BLEDevice::createServer();
    ble_server->setCallbacks(new BleConnCallbacks());

    BLEService* svc = ble_server->createService(BLEUUID(BLE_SERVICE_UUID), 32, 0);

    ble_tel_char = svc->createCharacteristic(BLE_TELEMETRY_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    ble_tel_char->addDescriptor(new BLE2902());

    ble_evt_char = svc->createCharacteristic(BLE_EVENT_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    ble_evt_char->addDescriptor(new BLE2902());

    ble_cmd_char = svc->createCharacteristic(BLE_CMD_UUID,
                       BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
    ble_cmd_char->setCallbacks(new BleCmdCallbacks());

    svc->start();

    BLEAdvertising* adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(BLE_SERVICE_UUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);   // connection interval helper for iOS
    BLEDevice::startAdvertising();
    Serial.printf("[BLE] Advertising as \"%s\"\n", BLE_DEVICE_NAME);
}

// Send a 16-byte telemetry packet. Called every BLE_TELEMETRY_INTERVAL_MS from background_task.
void ble_send_telemetry() {
    if (!ble_connected || !ble_tel_char) return;
    TelemetryPacket p;
    p.version     = TELEMETRY_PACKET_VERSION;
    p.activity    = (uint8_t)(current_activity_idx & 0xFF);
    p.fall_state  = fall_alert_until ? 1 : 0;
    p.battery_pct = (uint8_t) battery_percent(cached_vbat);
    p.stress_pct  = stress_level;
    p.hr          = (uint8_t) (disp_hr   > 255 ? 255 : (disp_hr   < 0 ? 0 : disp_hr));
    p.spo2        = (uint8_t) (disp_spo2 > 100 ? 100 : (disp_spo2 < 0 ? 0 : disp_spo2));
    DerivedMetrics dm = dm_get();
    p.posture     = dm.posture;
    p.sleep_state = dm.sleep_state;
    p.step_count  = dm.step_count;
    p.cadence_spm = dm.cadence_spm;
    p.hrv_rmssd   = dm.hrv_rmssd;
    p.resting_hr  = dm.resting_hr;
    float t_enc = (disp_temp_c - 30.0f) * 10.0f;
    if (t_enc < 0)      t_enc = 0;
    else if (t_enc > 255) t_enc = 255;
    p.temp_dc     = (uint8_t)t_enc;
    int batt_pct  = battery_percent(cached_vbat);
    float mv = cached_vbat * 1000.0f;
    p.battery_mv  = (mv < 0) ? 0 : (mv > 65535.0f ? 65535 : (uint16_t)mv);
    int mins = battery_minutes_remaining(batt_pct);
    p.time_left_min = (mins < 0) ? 0 : (mins > 65535 ? 65535 : (uint16_t)mins);
    ble_tel_char->setValue((uint8_t*)&p, sizeof(p));
    ble_tel_char->notify();
}

// Send an event string. Phone receives full CSV line via notify.
void ble_send_event(const char* line) {
    if (!ble_connected || !ble_evt_char) return;
    ble_evt_char->setValue((uint8_t*)line, strlen(line));
    ble_evt_char->notify();
}


// =============================================================================
// DISPLAY — called from background_task while holding i2c_mutex
// =============================================================================

void render_aod() {
    const char* status_str = fall_alert_until ? "FALL!" :
                             gathering_post_buffer ? "TRIG" : "OK";
    const char* act_short[] = {"Stat", "Walk", "Run", "Hand"};
    int pct = battery_percent(cached_vbat);
    char buf[22];
    snprintf(buf, sizeof(buf), "%s | %s | %d%%", status_str, act_short[current_activity_idx], pct);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 28);
    display.print(buf);
    display.display();
    last_aod_update = millis();
}

void enter_aod() {
    display.ssd1306_command(SSD1306_SETCONTRAST);
    display.ssd1306_command(0x00);  // minimum contrast — minimal charge pump current
    display.ssd1306_command(SSD1306_DISPLAYON);
    display_on         = false;
    display_aod_active = true;
    render_aod();
}

// Caller must hold i2c_mutex before calling display_wake
void display_wake(unsigned long now) {
    cached_vbat    = read_battery_voltage();
    last_vbat_read = now;
    disp_temp_c    = tempSensor.getTemperature();  // I2C — caller holds mutex
    last_temp_read = now;
    display_aod_active = false;
    display.ssd1306_command(SSD1306_SETCONTRAST);
    display.ssd1306_command(0x8F);  // medium contrast
    display.ssd1306_command(SSD1306_DISPLAYON);
    display_on       = true;
    display_sleep_at = now + DISPLAY_TIMEOUT_MS;
}

void display_sleep_now() {
#if DISPLAY_AOD
    enter_aod();
#else
    display.ssd1306_command(SSD1306_DISPLAYOFF);
    display_on = false;
#endif
}

void render_display() {
    char buf[24];
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    if (display_page == 0) {
        const char* status_str = fall_alert_until ? "FALL!" :
                                 gathering_post_buffer ? "TRIGGER" : "OK";
        const char* activities[] = {"Stationary", "Walking", "Running", "Hand Motion"};
        display.setCursor(0, 0);  display.print("Status:   "); display.println(status_str);
        display.setCursor(0, 10); display.print("Activity: "); display.println(activities[current_activity_idx]);
        int pct = battery_percent(cached_vbat);
        snprintf(buf, sizeof(buf), "Batt: %.2fV %3d%%", cached_vbat, pct);
        display.setCursor(0, 22); display.print(buf);
        display.drawRect(0, 34, 80, 10, SSD1306_WHITE);
        display.fillRect(1, 35, (pct * 78) / 100, 8, SSD1306_WHITE);
        int mins = battery_minutes_remaining(pct);
        snprintf(buf, sizeof(buf), "Left: %dh %02dm", mins / 60, mins % 60);
        display.setCursor(0, 48); display.print(buf);
        display.setCursor(104, 56); display.print("1/9");

    } else if (display_page == 1) {
        display.setCursor(0, 0); display.println("Accel (m/s2)");
        snprintf(buf, sizeof(buf), " X:%7.2f Y:%6.2f", (float)disp_ax, (float)disp_ay);
        display.setCursor(0, 12); display.println(buf);
        snprintf(buf, sizeof(buf), " Z:%7.2f", (float)disp_az);
        display.setCursor(0, 22); display.println(buf);
        snprintf(buf, sizeof(buf), " Mag:%6.2f", (float)disp_accMag);
        display.setCursor(0, 32); display.println(buf);
        display.setCursor(104, 56); display.print("2/9");

    } else if (display_page == 2) {
        display.setCursor(0, 0); display.println("Gyro (rad/s)");
        snprintf(buf, sizeof(buf), " X:%7.3f Y:%6.3f", (float)disp_gx, (float)disp_gy);
        display.setCursor(0, 12); display.println(buf);
        snprintf(buf, sizeof(buf), " Z:%7.3f", (float)disp_gz);
        display.setCursor(0, 22); display.println(buf);
        display.setCursor(104, 56); display.print("3/9");

    } else if (display_page == 3) {
        float temp_f = disp_temp_c * 9.0f / 5.0f + 32.0f;
        display.setTextSize(1);
        display.setCursor(0, 0); display.println("Body Temperature");
        display.drawFastHLine(0, 10, 128, SSD1306_WHITE);
        display.setTextSize(2);
        snprintf(buf, sizeof(buf), "%.2f C", disp_temp_c);
        display.setCursor(10, 18); display.println(buf);
        snprintf(buf, sizeof(buf), "%.2f F", temp_f);
        display.setCursor(10, 38); display.println(buf);
        display.setTextSize(1);
        display.setCursor(104, 56); display.print("4/9");

    } else if (display_page == 4) {
        display.setTextSize(1);
        display.setCursor(0, 0); display.println("Heart Rate & SpO2");
        display.drawFastHLine(0, 10, 128, SSD1306_WHITE);
        if (!bio_ok) {
            display.setCursor(0, 28); display.print("Sensor not found");
        } else if (!bio_has_reading) {
            display.setCursor(0, 28); display.print("Acquiring...");
        } else {
            display.setTextSize(2);
            snprintf(buf, sizeof(buf), "%3d bpm", disp_hr);
            display.setCursor(10, 14); display.println(buf);
            snprintf(buf, sizeof(buf), "%3d%%SpO2", disp_spo2);
            display.setCursor(10, 34); display.println(buf);
            display.setTextSize(1);
            snprintf(buf, sizeof(buf), "Confidence: %d%%", disp_hr_conf);
            display.setCursor(0, 56); display.print(buf);
        }
        display.setTextSize(1);
        display.setCursor(104, 56); display.print("5/9");

    } else if (display_page == 5) {
        // Stress level — derived from HR vs baseline + activity + SpO2 + temperature
        display.setTextSize(1);
        display.setCursor(0, 0);  display.println("Stress Level");
        display.drawFastHLine(0, 9, 128, SSD1306_WHITE);

        const char* label;
        if      (stress_level >= 75) label = "HIGH";
        else if (stress_level >= 50) label = "ELEVATED";
        else if (stress_level >= 25) label = "MILD";
        else                         label = "LOW";

        display.setTextSize(2);
        snprintf(buf, sizeof(buf), "%3d", (int)stress_level);
        display.setCursor(0, 14); display.print(buf);
        display.setTextSize(1);
        display.setCursor(54, 14); display.print("/100");
        display.setCursor(54, 24); display.print(label);

        // Bar
        display.drawRect(0, 34, 128, 7, SSD1306_WHITE);
        display.fillRect(1, 35, (stress_level * 126) / 100, 5, SSD1306_WHITE);

        // Breakdown — which component contributes most
        snprintf(buf, sizeof(buf), "HR:%d Spo:%d T:%d",
                 hr_stress_pct, spo2_stress_pct, temp_stress_pct);
        display.setCursor(0, 44); display.print(buf);

        if (bio_has_reading && hr_baseline >= 30.0f) {
            snprintf(buf, sizeof(buf), "now:%d base:%d", disp_hr, (int)hr_baseline);
            display.setCursor(0, 54); display.print(buf);
        } else {
            display.setCursor(0, 54); display.print("calibrating...");
        }
        display.setCursor(110, 56); display.print("6/9");

    } else if (display_page == 6) {
        // Debug page — live trigger telemetry for tuning thresholds
        display.setTextSize(1);
        display.setCursor(0, 0);  display.print("Fall Debug");
        display.setCursor(90, 0); display.print(ble_connected ? "BLE+" : "BLE-");
        display.drawFastHLine(0, 9, 128, SSD1306_WHITE);

        // Live values + key threshold for comparison
        snprintf(buf, sizeof(buf), "a:%5.1f g:%5.2f", (float)disp_accMag, (float)disp_gyroMag);
        display.setCursor(0, 12); display.println(buf);

        // Min/peak accMag and peak gyro since boot — drop below FF_THRESH = freefall reached
        snprintf(buf, sizeof(buf), "min:%4.1f pk:%4.1f/%.1f",
                 (float)dbg_min_accMag, (float)dbg_peak_accMag, (float)dbg_peak_gyroMag);
        display.setCursor(0, 22); display.println(buf);

        // Freefall state + trigger count + last path
        snprintf(buf, sizeof(buf), "FF:%c trg:%lu p:%c",
                 dbg_freefall_active ? 'Y' : 'N',
                 (unsigned long)dbg_trigger_count,
                 (char)dbg_last_path);
        display.setCursor(0, 32); display.println(buf);

        // Last 3 window confidences vs threshold
        snprintf(buf, sizeof(buf), "c:%.2f %.2f %.2f t%.2f",
                 dbg_last_conf[0], dbg_last_conf[1], dbg_last_conf[2], dbg_last_threshold);
        display.setCursor(0, 42); display.println(buf);

        // Last verdict + orientation angle
        snprintf(buf, sizeof(buf), "%s a:%.0f", dbg_last_verdict, dbg_last_angle_deg);
        display.setCursor(0, 52); display.print(buf);

        display.setCursor(110, 56); display.print("7/9");

    } else if (display_page == 7) {
        // Posture & sleep state — derived from EMA gravity + activity + HR
        DerivedMetrics dm = dm_get();
        const char* posture_lbl[]  = {"UPRIGHT", "RECLINED", "LYING", "INVERTED"};
        const char* sleep_lbl[]    = {"AWAKE", "RESTING", "LIGHT SLP", "DEEP SLP"};
        display.setTextSize(1);
        display.setCursor(0, 0);  display.print("Posture / Sleep");
        display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
        display.setCursor(0, 14); display.print("Posture:");
        display.setTextSize(2);
        display.setCursor(0, 24); display.print(posture_lbl[dm.posture & 3]);
        display.setTextSize(1);
        display.setCursor(0, 44); display.print("Sleep: ");
        display.print(sleep_lbl[dm.sleep_state & 3]);
        display.setCursor(110, 56); display.print("8/9");

    } else if (display_page == 8) {
        // Steps, cadence, HRV proxy, resting HR
        DerivedMetrics dm = dm_get();
        display.setTextSize(1);
        display.setCursor(0, 0);  display.print("Activity Stats");
        display.drawFastHLine(0, 9, 128, SSD1306_WHITE);
        snprintf(buf, sizeof(buf), "Steps:  %5u", (unsigned)dm.step_count);
        display.setCursor(0, 14); display.print(buf);
        snprintf(buf, sizeof(buf), "Cadence: %3u spm", (unsigned)dm.cadence_spm);
        display.setCursor(0, 26); display.print(buf);
        snprintf(buf, sizeof(buf), "HRV:    %3u", (unsigned)dm.hrv_rmssd);
        display.setCursor(0, 38); display.print(buf);
        snprintf(buf, sizeof(buf), "Rest HR: %3u bpm", (unsigned)dm.resting_hr);
        display.setCursor(0, 50); display.print(buf);
        display.setCursor(110, 56); display.print("9/9");

    }
    display.display();
}

// =============================================================================
// ML + FEATURE EXTRACTION
// =============================================================================

// Extract samples from the ring buffer snapshot at head h.
// include_activity: if true, appends 4-float activity tag after each 7-float sample (Model B).
void extract_features(float* out, int samples_needed, bool include_activity, int h, int offset = 0) {
    int start = (h - offset - samples_needed + MAX_SAMPLES * 2) % MAX_SAMPLES;
    int out_idx = 0;
    for (int i = 0; i < samples_needed; i++) {
        int buf_idx = (start + i) % MAX_SAMPLES;
        for (int c = 0; c < 7; c++) out[out_idx++] = ring_buffer[buf_idx][c];
        if (include_activity) {
            for (int c = 0; c < 4; c++) out[out_idx++] = current_activity_tag[c];
        }
    }
}

int get_signal_data_cb(size_t offset, size_t length, float *out_ptr, float *features) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

// =============================================================================
// POST-FALL GATES
// =============================================================================

bool check_post_event_stillness() {
    int count_still = 0;
    for (int i = 0; i < 50; i++) {
        int idx = (head_index - 1 - i + MAX_SAMPLES) % MAX_SAMPLES;
        float mag = ring_buffer[idx][6];
        if (mag >= STILLNESS_MIN && mag <= STILLNESS_MAX) count_still++;
    }
    return (count_still >= 38);  // was 45/50 — 38/50 tolerates someone breathing heavily / trying to get up
}

// Compare EMA gravity at trigger time (pre-fall) vs now (post-fall) to detect body rotation.
// Uses EMA values rather than ring-buffer averages — immune to dynamic acceleration artifacts.
bool check_orientation_change() {
    float pre_ax = trigger_gravity_x, pre_ay = trigger_gravity_y, pre_az = trigger_gravity_z;

    // Snapshot current EMA gravity under spinlock to avoid torn read
    float post_ax, post_ay, post_az;
    portENTER_CRITICAL(&ring_buf_mux);
    post_ax = gravity_x; post_ay = gravity_y; post_az = gravity_z;
    portEXIT_CRITICAL(&ring_buf_mux);

    float pre_mag  = sqrtf(pre_ax*pre_ax   + pre_ay*pre_ay   + pre_az*pre_az);
    float post_mag = sqrtf(post_ax*post_ax  + post_ay*post_ay  + post_az*post_az);

    if (pre_mag < 1.0f || post_mag < 1.0f) return false;

    float dot = (pre_ax*post_ax + pre_ay*post_ay + pre_az*post_az) / (pre_mag * post_mag);
    dot = constrain(dot, -1.0f, 1.0f);
    float angle_deg = acosf(dot) * 57.2958f;
    dbg_last_angle_deg = angle_deg;

#if DEBUG
    Serial.printf("Orientation change: %.1f deg (needs >= %.1f)\n", angle_deg, ORIENTATION_MIN_DEG);
#endif

    return angle_deg >= ORIENTATION_MIN_DEG;
}

// =============================================================================
// SAMPLING TASK — Core 1, Priority 20
//
// Owns: ring buffer writes, EMA gravity updates, freefall/impact detection.
// Runs at exactly 100Hz using vTaskDelayUntil — unaffected by display or ML latency.
//
// I2C access: non-blocking mutex try (3ms timeout). On bus contention, uses last cached
// MPU values. At 100Hz, losing 1-2 samples while background_task holds the bus for a
// display refresh (~23ms) is negligible for the 200+ sample detection windows.
// =============================================================================
void sampling_task(void *pvParameters) {
    TickType_t last_wake = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ);

    float last_ax = 0, last_ay = 0, last_az = 0;
    float last_gx = 0, last_gy = 0, last_gz = 0;
    bool freefall_active      = false;
    int  freefall_sample_count = 0;
    unsigned long freefall_time_ms = 0;
    // Pre-freefall gravity snapshot — captured at the moment freefall begins (Path A),
    // before the EMA starts being pulled down by the low-G samples. Cleaner reference than EMA at impact.
    float freefall_start_gx = 0, freefall_start_gy = 0, freefall_start_gz = 9.81f;
    unsigned long dbg_last_print = 0;
    int sample_counter = 0;

    while (true) {
        vTaskDelayUntil(&last_wake, period);

        float ax, ay, az, gx, gy, gz;

        if (xSemaphoreTake(i2c_mutex, pdMS_TO_TICKS(3)) == pdTRUE) {
            sensors_event_t a, g, t;
            mpu.getEvent(&a, &g, &t);
            xSemaphoreGive(i2c_mutex);
            ax = a.acceleration.x; ay = a.acceleration.y; az = a.acceleration.z;
            gx = g.gyro.x;         gy = g.gyro.y;         gz = g.gyro.z;
            last_ax = ax; last_ay = ay; last_az = az;
            last_gx = gx; last_gy = gy; last_gz = gz;
        } else {
            ax = last_ax; ay = last_ay; az = last_az;
            gx = last_gx; gy = last_gy; gz = last_gz;
        }

        float accMag  = sqrtf(ax*ax + ay*ay + az*az);
        float gyroMag = sqrtf(gx*gx + gy*gy + gz*gz);

        // Update EMA gravity and write ring buffer under a single spinlock acquisition
        portENTER_CRITICAL(&ring_buf_mux);
        gravity_x = GRAVITY_ALPHA * ax + (1.0f - GRAVITY_ALPHA) * gravity_x;
        gravity_y = GRAVITY_ALPHA * ay + (1.0f - GRAVITY_ALPHA) * gravity_y;
        gravity_z = GRAVITY_ALPHA * az + (1.0f - GRAVITY_ALPHA) * gravity_z;
        ring_buffer[head_index][0] = ax;
        ring_buffer[head_index][1] = ay;
        ring_buffer[head_index][2] = az;
        ring_buffer[head_index][3] = gx;
        ring_buffer[head_index][4] = gy;
        ring_buffer[head_index][5] = gz;
        ring_buffer[head_index][6] = accMag;
        head_index = (head_index + 1) % MAX_SAMPLES;
        portEXIT_CRITICAL(&ring_buf_mux);

        // Derived metrics: cheap 100Hz update (step peak detection)
        dm_update_fast(accMag);

        // Update display cache — volatile ensures visibility across cores
        disp_ax = ax; disp_ay = ay; disp_az = az;
        disp_gx = gx; disp_gy = gy; disp_gz = gz;
        disp_accMag  = accMag;
        disp_gyroMag = gyroMag;

        // Track session-wide extremes for debugging — visible on debug page and serial
        if (accMag  < dbg_min_accMag)  dbg_min_accMag  = accMag;
        if (accMag  > dbg_peak_accMag) dbg_peak_accMag = accMag;
        if (gyroMag > dbg_peak_gyroMag) dbg_peak_gyroMag = gyroMag;

#if DEBUG
        // Periodic telemetry every 1s — print live values vs thresholds so you can see
        // whether your motion is even approaching the trigger gates.
        if (++sample_counter >= 100) {
            sample_counter = 0;
            unsigned long t = millis();
            if (t - dbg_last_print >= 1000) {
                dbg_last_print = t;
                Serial.printf("[live] acc=%5.1f (FF<%.1f, Imp>%.1f, Pb>%.1f) gyro=%5.2f (Imp>%.2f, Pb>%.2f) FF=%c trigs=%lu\n",
                              accMag, FREEFALL_THRESHOLD, IMPACT_ACCEL_MIN, IMPACT_ONLY_ACCEL_MIN,
                              gyroMag, IMPACT_GYRO_MIN, IMPACT_ONLY_GYRO_MIN,
                              freefall_active ? 'Y' : 'N',
                              (unsigned long)dbg_trigger_count);
            }
        }
#endif

        // Skip detection while background_task is evaluating or alert is active
        if (trigger_fired || gathering_post_buffer) {
            dbg_freefall_active = false;
            continue;
        }
        if ((unsigned long)(millis() - last_trigger_time) < COOLDOWN_MS) continue;

        // Phase 1: Freefall detection
        if (accMag < FREEFALL_THRESHOLD) {
            freefall_sample_count++;
            if (freefall_sample_count >= FREEFALL_MIN_SAMPLES && !freefall_active) {
                freefall_active   = true;
                freefall_time_ms  = millis();
                // Snapshot gravity BEFORE the EMA gets pulled toward zero by freefall samples.
                // For Path A this is the clean pre-fall body orientation.
                portENTER_CRITICAL(&ring_buf_mux);
                freefall_start_gx = gravity_x;
                freefall_start_gy = gravity_y;
                freefall_start_gz = gravity_z;
                portEXIT_CRITICAL(&ring_buf_mux);
                dbg_freefall_active = true;
#if DEBUG
                Serial.printf(">>> Freefall START accMag=%.2f\n", accMag);
#endif
            }
        } else {
            freefall_sample_count = 0;
        }

        if (freefall_active && (millis() - freefall_time_ms > FREEFALL_TO_IMPACT_MS)) {
            freefall_active = false;
            dbg_freefall_active = false;
#if DEBUG
            Serial.println(">>> Freefall expired — no impact within window");
#endif
        }

        // Phase 2a: Path A — freefall followed by impact
        bool path_a = freefall_active  && accMag > IMPACT_ACCEL_MIN      && gyroMag > IMPACT_GYRO_MIN;
        // Phase 2b: Path B — high-G stumble/collapse without freefall
        bool path_b = !freefall_active && accMag > IMPACT_ONLY_ACCEL_MIN && gyroMag > IMPACT_ONLY_GYRO_MIN;

        if (path_a || path_b) {
            // For Path A use the pre-freefall snapshot. For Path B the EMA is uncontaminated.
            float ref_gx = path_a ? freefall_start_gx : gravity_x;
            float ref_gy = path_a ? freefall_start_gy : gravity_y;
            float ref_gz = path_a ? freefall_start_gz : gravity_z;

            freefall_active       = false;
            freefall_sample_count = 0;
            dbg_freefall_active   = false;
            last_trigger_time     = millis();
            dbg_last_path         = path_a ? 'A' : 'B';
            dbg_trigger_count++;

            portENTER_CRITICAL(&ring_buf_mux);
            trigger_gravity_x = ref_gx;
            trigger_gravity_y = ref_gy;
            trigger_gravity_z = ref_gz;
            trigger_head_idx  = head_index;
            portEXIT_CRITICAL(&ring_buf_mux);
            trigger_accMag  = accMag;
            trigger_gyroMag = gyroMag;

            trigger_fired = true;

#if DEBUG
            Serial.printf(">>> TRIGGER Path %c — accMag=%.1f gyroMag=%.2f head=%d\n",
                          path_a ? 'A' : 'B', accMag, gyroMag, trigger_head_idx);
#endif
            ledcWrite(LED_CH_GREEN, 0);
            digitalWrite(LED_YELLOW, HIGH);
        }
    }
}

// =============================================================================
// BACKGROUND TASK — Core 0, Priority 3
//
// Handles display, ML inference, slow sensor reads, fall evaluation.
// ML inference (30–50ms) and display I2C transfers (~23ms) run here and no
// longer block the 100Hz sampling loop on Core 1.
// =============================================================================
void background_task(void *pvParameters) {
    unsigned long last_activity_check = 0;
    unsigned long last_dm_slow_tick   = 0;

    while (true) {
        unsigned long now = millis();

        // --- Button (interrupt-driven, debounced) ---
        if (btn_pressed) {
            btn_pressed = false;
            if (now - last_btn_press > BTN_DEBOUNCE_MS) {
                last_btn_press = now;
                if (xSemaphoreTake(i2c_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
#if DISPLAY_ALWAYS_ON
                    display_page = (display_page + 1) % DISPLAY_PAGES;
                    render_display();
#else
                    if (!display_on) {
                        display_page = 0;
                        display_wake(now);   // also reads temp/battery — holds mutex
                        render_display();
                    } else {
                        display_page     = (display_page + 1) % DISPLAY_PAGES;
                        display_sleep_at = now + DISPLAY_TIMEOUT_MS;
                        render_display();
                    }
#endif
                    xSemaphoreGive(i2c_mutex);
                }
            }
        }

        // --- Battery voltage (once per minute) ---
        if (now - last_vbat_read >= VBAT_READ_INTERVAL) {
            last_vbat_read = now;
            cached_vbat = read_battery_voltage();
        }

        // --- Body temperature (once per minute) ---
        if (now - last_temp_read >= TEMP_READ_INTERVAL) {
            last_temp_read = now;
            if (xSemaphoreTake(i2c_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
                disp_temp_c = tempSensor.getTemperature();
                xSemaphoreGive(i2c_mutex);
            }
        }

        // --- HR / SpO2 (~1Hz sensor output rate) ---
        if (bio_ok && now - last_bio_read >= BIO_READ_INTERVAL) {
            last_bio_read = now;
            if (xSemaphoreTake(i2c_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
                body = bioHub.readBpm();
                xSemaphoreGive(i2c_mutex);
                if (body.heartRate > 0 && body.confidence > 0) {
                    disp_hr          = body.heartRate;
                    disp_spo2        = body.oxygen;
                    disp_hr_conf     = body.confidence;
                    bio_has_reading  = true;
                }
            }
            update_stress_level();  // refresh stress after each bio read (or no-op if no reading yet)
        }

        // --- Display refresh / auto-sleep / AOD ---
        if (xSemaphoreTake(i2c_mutex, 0) == pdTRUE) {  // non-blocking — skip if sampler has bus
            if (display_on) {
#if !DISPLAY_ALWAYS_ON
                if (now > display_sleep_at) {
                    display_sleep_now();
                } else
#endif
                if (now - last_display_update >= 250) {
                    last_display_update = now;
                    render_display();
                }
            } else if (display_aod_active) {
                if (now - last_aod_update >= DISPLAY_AOD_UPDATE_MS) {
                    render_aod();
                }
            }
            xSemaphoreGive(i2c_mutex);
        }

        // --- Clear fall alert ---
        if (fall_alert_until && now > fall_alert_until) {
            fall_alert_until = 0;
            digitalWrite(LED_RED, LOW);
            update_activity_leds();
        }

        // --- Derived metrics slow tick (1Hz) ---
        if (now - last_dm_slow_tick >= 1000) {
            last_dm_slow_tick = now;
            float gx, gy, gz;
            portENTER_CRITICAL(&ring_buf_mux);
            gx = gravity_x; gy = gravity_y; gz = gravity_z;
            portEXIT_CRITICAL(&ring_buf_mux);
            int hr_now = bio_has_reading ? disp_hr : 0;
            int hr_base = (int)hr_baseline;
            dm_update_slow(current_activity_idx, hr_now, hr_base, gx, gy, gz);
        }

        // --- BLE: 10Hz telemetry stream ---
        if (ble_connected && now - last_ble_telemetry_send >= BLE_TELEMETRY_INTERVAL_MS) {
            last_ble_telemetry_send = now;
            ble_send_telemetry();
        }

        // --- Activity Classifier (Model A, 1s interval) ---
        if (!gathering_post_buffer && !trigger_fired && now - last_activity_check >= 1000) {
            last_activity_check = now;

            portENTER_CRITICAL(&ring_buf_mux);
            int h = head_index;
            portEXIT_CRITICAL(&ring_buf_mux);

            extract_features(activity_features_buf,
                             impulse_1003832_1.dsp_input_frame_size / 7, false, h);

            signal_t sigA;
            sigA.total_length = impulse_1003832_1.dsp_input_frame_size;
            sigA.get_data = [](size_t offset, size_t length, float *out) -> int {
                return get_signal_data_cb(offset, length, out, activity_features_buf);
            };

            ei_impulse_result_t resultA = { 0 };
            process_impulse(&impulse_handle_1003832_1, &sigA, &resultA, false);

            int winner = 0; float best = 0;
            for (uint16_t i = 0; i < impulse_1003832_1.label_count; i++) {
                if (resultA.classification[i].value > best) {
                    best = resultA.classification[i].value;
                    winner = i;
                }
            }
            if (best > 0.7f) {
                int winner_internal = EI_TO_INTERNAL[winner & 3];
                for (int i = 0; i < 4; i++) current_activity_tag[i] = 0.0f;
                // Tag fed into Model B must match training order (EI alphabetical), keep raw winner index
                current_activity_tag[winner] = 1.0f;
                current_activity_idx = winner_internal;
                update_activity_leds();
                if (display_aod_active && xSemaphoreTake(i2c_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                    render_aod();
                    xSemaphoreGive(i2c_mutex);
                }
            }
        }

        // --- Fall Evaluation (Model B, triggered by sampling_task) ---
        if (trigger_fired) {
            trigger_fired         = false;
            gathering_post_buffer = true;

            // CRITICAL: snapshot trigger_head_idx BEFORE waiting.
            // If we used the current head_index after a long delay, the impact would have been
            // overwritten in the ring buffer (5s buffer vs 3.5s+ of evaluation delay).
            int trigger_h = trigger_head_idx;

            // Phase 1: brief wait so ~50 post-impact samples (0.5s) land in the buffer.
            vTaskDelay(pdMS_TO_TICKS(ML_WAIT_MS));

            float threshold = (current_activity_idx == HAND_MOTION_IDX) ? 0.90f : 0.85f;
            dbg_last_threshold = threshold;

            // Centre the extraction window on the impact: end the window ML_POST_OFFSET samples
            // AFTER trigger so the window contains pre-impact context + impact + brief post-impact.
            // This matches how Edge Impulse training windows are typically aligned around the event.
            int eval_h = (trigger_h + ML_POST_OFFSET + MAX_SAMPLES) % MAX_SAMPLES;

            // Wider offsets (0/25/50 samples = 0/250/500 ms) — the three windows now actually
            // probe different alignments of the event rather than nearly the same 200ms window.
            int offsets[3] = {0, 25, 50};
            int passing_windows = 0;

            for (int w = 0; w < 3; w++) {
                extract_features(fall_features_buf,
                                 impulse_1004073_1.dsp_input_frame_size / 11, true, eval_h, offsets[w]);

                signal_t sigB;
                sigB.total_length = impulse_1004073_1.dsp_input_frame_size;
                sigB.get_data = [](size_t offset, size_t length, float *out) -> int {
                    return get_signal_data_cb(offset, length, out, fall_features_buf);
                };

                ei_impulse_result_t resultB = { 0 };
                process_impulse(&impulse_handle_1004073_1, &sigB, &resultB, false);

                float conf = resultB.classification[FALL_IDX].value;
                dbg_last_conf[w] = conf;
                if (conf >= threshold) passing_windows++;
#if DEBUG
                Serial.printf("Window %d (offset=%d) conf=%.3f (needs >= %.2f) %s\n",
                              w, offsets[w], conf, threshold,
                              conf >= threshold ? "PASS" : "fail");
#endif
            }
            dbg_last_passing = passing_windows;

            // Majority voting (2 of 3) — tolerates one misaligned window rather than killing
            // the detection on a single jitter.
            bool ml_pass = passing_windows >= 2;
#if DEBUG
            Serial.printf("ML verdict: %d/3 windows passed → %s\n", passing_windows, ml_pass ? "PASS" : "FAIL");
#endif

            // Phase 2: wait for posture to settle (stillness + orientation gates need calm post-fall state).
            vTaskDelay(pdMS_TO_TICKS(SETTLE_WAIT_MS));

            bool stillness_ok = check_post_event_stillness();
            bool orient_ok    = check_orientation_change();
            dbg_last_stillness = stillness_ok;

            bool fall_ok = ml_pass && stillness_ok && orient_ok;
            if (fall_ok) {
                strncpy((char*)dbg_last_verdict, "FALL", sizeof(dbg_last_verdict));
                Serial.println("\n===========================");
                Serial.println("TRUE FALL DETECTED!");
                Serial.println("===========================\n");
                alert_fall();
            } else {
                if (!ml_pass)          strncpy((char*)dbg_last_verdict, "rej:ml",       sizeof(dbg_last_verdict));
                else if (!stillness_ok) strncpy((char*)dbg_last_verdict, "rej:moving",   sizeof(dbg_last_verdict));
                else                    strncpy((char*)dbg_last_verdict, "rej:no-tilt",  sizeof(dbg_last_verdict));
                Serial.printf("Rejected: ml=%s stillness=%s orient=%s\n",
                              ml_pass      ? "PASS" : "fail",
                              stillness_ok ? "PASS" : "fail",
                              orient_ok    ? "PASS" : "fail");
                digitalWrite(LED_YELLOW, LOW);
                update_activity_leds();
            }

            // BLE: notify phone only when a fall is actually confirmed.
            if (fall_ok) ble_send_event("FALL");

            gathering_post_buffer = false;
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);

    WiFi.mode(WIFI_OFF);
    // btStop() removed — BLE is now used for telemetry/calibration. Adds ~50KB RAM and ~80mA.

    ledcSetup(LED_CH_GREEN, 1000, 8);
    ledcAttachPin(LED_GREEN, LED_CH_GREEN);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_RED,    OUTPUT);

    ledcSetup(BUZZER_CH, 2000, 8);
    ledcAttachPin(BUZZER_PIN, BUZZER_CH);
    gpio_set_drive_capability((gpio_num_t)BUZZER_PIN, GPIO_DRIVE_CAP_3);

    // Startup test — scheduler not yet running, delay() is safe here
    ledcWrite(LED_CH_GREEN, 255);
    digitalWrite(LED_YELLOW, HIGH);
    digitalWrite(LED_RED,    HIGH);
    buzzer_beep(BUZZER_FREQ, 150); delay(80);
    buzzer_beep(BUZZER_FREQ, 150); delay(80);
    buzzer_beep(BUZZER_FREQ, 150);
    delay(200);
    ledcWrite(LED_CH_GREEN, 0);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED,    LOW);

    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);  // 400kHz fast mode — halves I2C transfer time vs 100kHz default

    pinMode(BUTTON_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), btn_isr, FALLING);
    gpio_wakeup_enable((gpio_num_t)BUTTON_PIN, GPIO_INTR_LOW_LEVEL);
    esp_sleep_enable_gpio_wakeup();

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        Serial.println("SSD1306 not found!");
    } else {
        display.clearDisplay();
        display.display();
#if DISPLAY_ALWAYS_ON
        display_wake(millis());
#elif DISPLAY_AOD
        enter_aod();
#else
        display.ssd1306_command(SSD1306_DISPLAYOFF);
#endif
    }

    tempSensor.sensorAddress = MAX30205_ADDRESS2;
    tempSensor.begin();
    if (tempSensor.scanAvailableSensors()) {
        Serial.println("MAX30205 Found!");
        disp_temp_c = tempSensor.getTemperature();
    } else {
        Serial.println("MAX30205 not found!");
    }
    last_temp_read = millis();

    cached_vbat = read_battery_voltage();
    last_vbat_read = millis();

    if (bioHub.begin() == 0 && bioHub.configBpm(MODE_ONE) == 0) {
        Serial.println("Bio Sensor Hub found! Waiting 4s for calibration...");
        delay(4000);
        bio_ok = true;
    } else {
        Serial.println("Bio Sensor Hub not found!");
    }

    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050! Check wiring.");
        while (1) { delay(10); }
    }
    Serial.println("MPU6050 Found!");
    mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
    mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // One-time allocation — no heap fragmentation from the original per-second malloc/free
    activity_features_buf = (float*)malloc(impulse_1003832_1.dsp_input_frame_size * sizeof(float));
    fall_features_buf     = (float*)malloc(impulse_1004073_1.dsp_input_frame_size * sizeof(float));
    if (!activity_features_buf || !fall_features_buf) {
        Serial.println("Feature buffer malloc failed!");
        while (1) { delay(10); }
    }

    i2c_mutex = xSemaphoreCreateMutex();

    dm_init();

    // Init BLE last so all sensor state is valid when the first connection arrives.
    // Free classic-BT controller memory first — we only use BLE, recovers ~30KB heap.
    esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
    ble_init();

    update_activity_leds();

    // sampling_task: Core 1, priority 20 — real-time 100Hz sensor loop
    xTaskCreatePinnedToCore(sampling_task,   "sampler",    8192,  nullptr, 20, nullptr, 1);
    // background_task: Core 0, priority 10 — bumped above default BLE host (5) so fall
    // evaluation isn't starved when BLE is busy advertising/notifying.
    xTaskCreatePinnedToCore(background_task, "background", 16384, nullptr, 10, nullptr, 0);
}

// All work is in the two FreeRTOS tasks above.
// FreeRTOS tickless idle handles low-power sleep automatically when both tasks are blocked.
void loop() {
    vTaskDelay(portMAX_DELAY);
}
