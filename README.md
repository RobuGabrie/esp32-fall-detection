<div align="center">

# 🛡️ FallGuard

### A wearable fall-detection & vitals-monitoring device powered by on-device Edge AI

[![Platform](https://img.shields.io/badge/platform-ESP32-000000?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Framework](https://img.shields.io/badge/framework-Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![Build](https://img.shields.io/badge/build-PlatformIO-FF7F00?style=for-the-badge&logo=platformio&logoColor=white)](https://platformio.org/)
[![ML](https://img.shields.io/badge/ML-Edge%20Impulse-3B47CE?style=for-the-badge)](https://edgeimpulse.com/)
[![BLE](https://img.shields.io/badge/wireless-BLE%204.2-0082FC?style=for-the-badge&logo=bluetooth&logoColor=white)](https://www.bluetooth.com/)

*Two TinyML models. Three independent gates. One reliable fall alert.*

</div>

---

## ✨ Overview

**FallGuard** is an upper-arm wearable that fuses an IMU, a pulse-ox bio-hub, and a body-temperature sensor with two on-device Edge Impulse neural networks to detect real-world falls — while rejecting the everyday motions that fool simple threshold detectors. Everything runs on a single ESP32: no cloud, no phone required for detection.

> 🎯 **Design goal:** zero false positives on sitting down, slamming a door, dropping the arm, or jumping — while catching genuine forward / backward / sideways falls in under 4 seconds.

---

## 🚀 Features

| | |
|---|---|
| 🧠 **Dual on-device ML** | Activity classifier (4-class) + fall classifier (binary), both trained on real wearable data |
| ⚡ **Real-time pipeline** | 100 Hz IMU sampling on a dedicated FreeRTOS core, hard-real-time trigger detection |
| 🩺 **Vitals fusion** | Heart rate, SpO₂, body temperature, HRV proxy, resting HR baseline, 0–100 stress score |
| 📡 **BLE telemetry** | 19-byte packet @ 2 Hz, plus `FALL` event notifications |
| 🔋 **Battery aware** | 21-point LUT, live mV, runtime-remaining estimate |
| 🖥️ **9-page OLED UI** | Status, sensors, vitals, stress, posture/sleep, step stats, ML debug |
| 🌙 **Always-on-display** | Low-contrast AOD mode for at-a-glance background visibility |
| 🦶 **Derived metrics** | Step counter, cadence, posture (upright/reclined/lying/inverted), sleep state |

---

## 🛠️ Hardware

<table>
<tr><th align="left">Part</th><th align="left">Role</th><th align="left">Bus / Pin</th></tr>
<tr><td>ESP32 dev board @ 80 MHz</td><td>Compute + BLE</td><td>—</td></tr>
<tr><td>MPU6050</td><td>IMU (±16 g, ±1000 °/s)</td><td>I²C @ 400 kHz · SDA=21 SCL=22</td></tr>
<tr><td>SSD1306 128×64</td><td>OLED display</td><td>I²C · addr 0x3D</td></tr>
<tr><td>MAX30205</td><td>Body temperature</td><td>I²C</td></tr>
<tr><td>MAX32664</td><td>HR / SpO₂ bio-hub</td><td>I²C · RESET=16 MFIO=17</td></tr>
<tr><td>Status LEDs</td><td>Activity indicator</td><td>green=32 (PWM) · yellow=33 · red=4</td></tr>
<tr><td>Buzzer</td><td>Fall alert</td><td>GPIO 23 · LEDC @ 1.7 kHz</td></tr>
<tr><td>Push button</td><td>UI navigation</td><td>GPIO 27 · INPUT_PULLUP · FALLING IRQ</td></tr>
<tr><td>Li-ion / LiPo cell</td><td>Power</td><td>VBAT → GPIO 35 (divider ×2)</td></tr>
</table>

> All pinout, timing, and threshold constants are grouped at the top of [`src/main.cpp`](src/main.cpp).

---

## ⚙️ Build & Flash

This is a PlatformIO project. The `huge_app.csv` partition (3 MB app) is **required** — the Edge Impulse SDK + Bluedroid BLE stack exceed the default 1.3 MB layout.

```bash
pio run                  # 🔨 build
pio run -t upload        # ⬆️  flash to ESP32
pio device monitor       # 📟 serial @ 115200
```

All library dependencies are pinned in [`platformio.ini`](platformio.ini).

---

## 🧠 The ML Models — Trained on Real Data with Edge Impulse

Two impulses live in [`src/merged/`](src/merged/), exported from Edge Impulse Studio as a multi-impulse C++ library and linked directly into the firmware. **Inference runs entirely on the ESP32** — no cloud at runtime.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Edge Impulse Studio                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐    │
│  │ Real-device  │ → │  Labelled    │ → │  Train + tune impulse  │    │
│  │ data capture │   │   dataset    │   │  (DSP + NN blocks)     │    │
│  └──────────────┘   └──────────────┘   └────────┬───────────────┘    │
│                                                  ▼                   │
│                                       ┌────────────────────┐         │
│                                       │ C++ library export │         │
│                                       └─────────┬──────────┘         │
└─────────────────────────────────────────────────│────────────────────┘
                                                  ▼
                                       ╔══════════════════╗
                                       ║   ESP32 @ 80MHz  ║
                                       ║  process_impulse ║
                                       ╚══════════════════╝
```

### 📡 Data collection — on the actual wearable

Raw 7-channel windows were streamed live into Edge Impulse Studio over the serial **data forwarder**, sampled at **100 Hz on the production hardware** (same MPU6050, same bicep strap, same ±16 g / ±1000 °/s range used at runtime). Channels:

```
ax | ay | az | gx | gy | gz | |a|     ← magnitude precomputed so the
                                        model doesn't have to relearn it
```

Sampling on the deployed device — not a phone or breadboard — is what makes the models work in the field. Sensor mounting, strap tightness, body coupling, and the IMU's full-scale range are all baked into the training distribution.

---

#### 🏃 Activity dataset (Model A)

Several minutes per class captured **on the upper arm during real activity**:

| Class | What was recorded |
|---|---|
| 🧍 `stationary` | Standing / sitting still |
| 🚶 `walking` | Indoor + outdoor pace |
| 🏃 `running` | Steady-state jogging |
| 👋 `hand_motion` | Gesturing, reaching, fidgeting — the hardest negative class for the fall model |

#### 💥 Fall dataset (Model B) — the critical work

| Positives | Negatives (false-positive bait) |
|---|---|
| Forward falls | Sitting down hard |
| Backward falls | Dropping into a chair |
| Left side falls | Slamming a door |
| Right side falls | Putting the arm down forcefully |
| Natural body-collapse simulations | Jumping, footstrikes during walking & running |

Each window was **labelled with the activity tag** that the device would have been emitting at that moment. That way the model learns to interpret the same impact differently based on context — a 3 g spike during `running` is a footstrike; the same spike during `stationary` is a real event.

---

### 🧩 Impulse design

#### Model A — Activity classifier

```
   IMU window (7 ch @ 100 Hz)
            ▼
   ┌──────────────────────┐
   │  Spectral analysis   │   FFT magnitudes + RMS
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │   Fully-connected NN │
   └──────────┬───────────┘
              ▼
   { hand_motion │ running │ stationary │ walking }   ← alphabetical
```

Labels are returned **alphabetically by Edge Impulse** and remapped to the firmware's internal order via `EI_TO_INTERNAL[]` in `main.cpp`.

#### Model B — Fall classifier

```
   IMU window (7 ch) + one-hot activity tag (4 ch) = 11 channels
                              ▼
                ┌──────────────────────────┐
                │      DSP feature block   │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │      1-D ConvNet         │
                └────────────┬─────────────┘
                             ▼
                       { fall │ not_fall }
```

The **per-sample activity tag** is the live output of Model A, frozen at trigger time. This is the secret sauce that lets a single fall model behave correctly across very different motion contexts.

---

### 🎓 Training

- **Split:** standard Edge Impulse 80 / 20 train-validation + held-out test set.
- **Augmentation:** Studio defaults — generalisation came from genuinely diverse negative examples, not augmentation tricks.
- **Operating point:** chosen from the precision-recall curve to **favour recall** — the firmware's stillness + orientation gates filter false positives downstream, so the ML stage can stay loose.

### 📦 Deployment

The Studio's **C++ library** export was unpacked into `src/merged/` and both impulses are linked into a single binary using Edge Impulse's multi-impulse build. Inference is **~30–50 ms per window** on the 80 MHz ESP32 — well inside budget: Model A runs at 1 Hz; Model B only runs on a trigger event.

| Impulse handle | Role | Input | Output |
|---|---|---|---|
| `impulse_handle_1003832_1` | Activity | 7-ch IMU window | 4 classes |
| `impulse_handle_1004073_1` | Fall | 11-ch window (IMU + activity tag) | binary |

---

## 🚦 The Three-Gate Detection Pipeline

A fall is **only** confirmed when all three independent layers agree. False positives from any single layer get rejected by the others.

```
       ┌──────────────────────────────────────────────────────────────┐
       │  GATE 1 — Trigger    (Core 1, hard real-time, 100 Hz)        │
       │                                                              │
       │   Path A:  freefall ≥30 ms  →  impact >2.2 g & >1.0 rad/s    │
       │   Path B:  high-G stumble — impact >3.5 g & >2.0 rad/s       │
       └────────────────────────────┬─────────────────────────────────┘
                                    ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  GATE 2 — ML  (Core 0, Edge Impulse Model B)                 │
       │                                                              │
       │   3 windows @ offsets 0 / 250 / 500 ms around impact         │
       │   Majority vote ≥ 2/3 above threshold                        │
       │   threshold = 0.85  (or 0.90 when activity = hand_motion)    │
       └────────────────────────────┬─────────────────────────────────┘
                                    ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  GATE 3 — Posture  (after 3 s settle)                        │
       │                                                              │
       │   Stillness: 38/50 samples with |a| ∈ [8, 11] m/s²           │
       │   Orientation: EMA gravity rotated ≥ 35° vs pre-fall ref     │
       └────────────────────────────┬─────────────────────────────────┘
                                    ▼
                          🚨  CONFIRMED FALL  🚨
                  (red LED + 3 s buzzer + BLE "FALL" event)
```

> 💡 The pre-fall gravity reference for Gate 3 is snapshotted **before** freefall begins so it isn't pulled toward zero by the low-G samples — preserving a clean upright-body baseline for the angle comparison.

---

## 🧵 Concurrency Model

| Task | Core | Priority | Responsibilities |
|---|---|---|---|
| `sampling_task` | **Core 1** | 20 | 100 Hz IMU read, ring-buffer write, EMA gravity, trigger detection |
| `background_task` | **Core 0** | 10 | Display, ML inference, slow sensors, BLE telemetry, fall evaluation |

- 🔒 **Ring buffer** protected by a `portMUX` spinlock.
- 🔒 **I²C bus** shared via a FreeRTOS mutex (3 ms non-blocking try in the sampler — falls back to last cached sample on contention).
- 🛰️ `background_task` priority is bumped above the BLE host's default (5) so fall evaluation isn't starved during BLE traffic.

---

## 📡 BLE Protocol

| Property | Value |
|---|---|
| **Device name** | `FallGuard` |
| **Service UUID** | `f00dbabe-0001-1000-8000-00805f9b34fb` |
| **Telemetry** (notify, 2 Hz) | 19-byte packed `TelemetryPacket` |
| **Event** (notify) | `FALL` string on confirmed fall |
| **Command** (write) | `alert_off` → silences red LED + buzzer |

<details>
<summary><b>📦 Telemetry packet layout (v4, 19 bytes)</b></summary>

| Offset | Field | Type | Notes |
|---:|---|---|---|
| 0 | `version` | u8 | = 4 |
| 1 | `activity` | u8 | 0=stationary 1=walking 2=running 3=hand-motion |
| 2 | `fall_state` | u8 | 0=idle · 1=fall confirmed |
| 3 | `battery_pct` | u8 | |
| 4 | `stress_pct` | u8 | 0–100 |
| 5 | `hr` | u8 | bpm |
| 6 | `spo2` | u8 | % |
| 7 | `posture` | u8 | 0=upright 1=reclined 2=lying 3=inverted |
| 8 | `sleep_state` | u8 | 0=awake 1=resting 2=light 3=deep |
| 9–10 | `step_count` | u16 LE | |
| 11 | `cadence_spm` | u8 | steps / min |
| 12 | `hrv_rmssd` | u8 | HR-based proxy |
| 13 | `resting_hr` | u8 | bpm |
| 14 | `temp_dc` | u8 | decode: `temp_C = byte/10 + 30` |
| 15–16 | `battery_mv` | u16 LE | |
| 17–18 | `time_left_min` | u16 LE | runtime estimate |

</details>

> ⚠️ Replace the demo UUIDs in [`src/main.cpp`](src/main.cpp) before deploying anything you don't want a stranger's app to pick up.

---

## 📁 Repository Layout

```
falling/
├── platformio.ini                  ← board, partition, dependencies
├── src/
│   ├── main.cpp                    ← all firmware logic (tasks, gates, BLE, UI)
│   ├── derived_metrics.{cpp,h}     ← steps, cadence, posture, sleep, HRV proxy
│   └── merged/                     ← Edge Impulse multi-impulse C++ export
│       ├── edge-impulse-sdk/
│       ├── model-parameters/
│       └── tflite-model/
└── README.md
```

---

## 📜 License

No license file is included; treat the code as **all-rights-reserved** unless one is added.

<div align="center">

---

*Built with ❤️ on ESP32, Edge Impulse, and a lot of padded-mat test falls.*

</div>
