# FallGuard

A wearable fall-detection and vitals-monitoring device built on the ESP32. Designed to be worn on the upper arm (bicep), it fuses an IMU with a heart-rate / SpO₂ sensor and a body-temperature sensor, runs two on-device Edge Impulse models, streams telemetry over BLE, and triggers an audible/visual alert when a real fall is detected.

## What it does

The firmware does four things in parallel on the ESP32's two cores:

1. **Samples the IMU at 100 Hz** (MPU6050, ±16 g / ±1000 °/s) into a 5-second ring buffer on Core 1.
2. **Classifies the current activity once per second** — *stationary*, *walking*, *running*, or *hand motion* — using a small Edge Impulse model.
3. **Detects falls** through a two-stage pipeline: a hand-tuned trigger (freefall + impact, *or* high-G stumble) followed by an Edge Impulse fall-classifier model running on the windowed IMU + activity tag, then post-event posture gates (stillness + orientation change ≥ 35°).
4. **Reads vitals** — heart rate and SpO₂ from a MAX32664 bio-sensor hub, body temperature from a MAX30205 — and computes a 0–100 stress score from HR vs. learned resting baseline, SpO₂, and temperature.

A confirmed fall lights a red LED, sounds the buzzer for 3 seconds, and emits a `FALL` BLE notification. A companion phone can connect over BLE to receive a 19-byte telemetry packet at 2 Hz (activity, fall state, battery %, stress, HR, SpO₂, posture, sleep state, steps, cadence, HRV proxy, resting HR, temperature, battery mV, runtime estimate) and can write `alert_off` to silence an active alert.

A 128×64 SSD1306 OLED with a single navigation button cycles through 9 pages (status / accel / gyro / temp / HR-SpO₂ / stress / fall-debug / posture-sleep / step-stats), with an always-on-display mode at minimum contrast for low-power background visibility.

## Hardware

| Part                      | Bus / Pin                                  |
| ------------------------- | ------------------------------------------ |
| ESP32 dev board (80 MHz)  | —                                          |
| MPU6050 IMU               | I²C @ 400 kHz, SDA=21 SCL=22               |
| SSD1306 128×64 OLED       | I²C, addr 0x3D                             |
| MAX30205 body temp        | I²C                                        |
| MAX32664 bio hub (HR/SpO₂)| I²C, RESET=16 MFIO=17                      |
| Status LEDs               | green=32 (PWM), yellow=33, red=4           |
| Buzzer                    | GPIO 23 (LEDC, 1.7 kHz)                    |
| Button                    | GPIO 27 (INPUT_PULLUP, FALLING IRQ)        |
| Battery sense             | GPIO 35 (ADC1_CH7, divider ×2)             |

Pinout, thresholds, and timing constants live at the top of `src/main.cpp` — edit them there before flashing.

## Build & flash

This is a PlatformIO project for `esp32dev` with the `huge_app.csv` partition (3 MB app), required because the Edge Impulse SDK + Bluedroid BLE stack exceed the default 1.3 MB partition.

```bash
pio run                  # build
pio run -t upload        # flash
pio device monitor       # serial @ 115200
```

`platformio.ini` already lists every library dependency.

## The ML models — trained on real data with Edge Impulse

Two models live in `src/merged/` (the `tflite-model/` and `model-parameters/` directories), exported from Edge Impulse as a C++ library and compiled into the firmware. Inference runs entirely on the ESP32 — there is no cloud component at runtime.

### Data collection

Raw 7-channel windows were recorded on the actual wearable (the same MPU6050 mounted on the same bicep strap that the production firmware uses), streamed live into Edge Impulse Studio via the data forwarder over serial at 100 Hz. The seven channels are `ax, ay, az, gx, gy, gz, |a|` — the last being precomputed accelerometer magnitude, included so the model doesn't have to learn that feature from scratch. Sampling on the deployed hardware (not a phone or dev board) is what makes the models work in the field: sensor mounting, strap tightness, body-coupling, and the ±16 g / ±1000 °/s range used at runtime are all baked into the training distribution.

#### Activity dataset (Model A)
Several minutes per class were captured for *stationary*, *walking*, *running*, and *hand motion*, with the device worn on the upper arm during real activity. Edge Impulse processes these into overlapping windows for the classifier.

#### Fall dataset (Model B)
This is where the data work matters most. Falls were collected by repeatedly dropping the device onto a padded mat from standing height in the four canonical directions (forward / backward / left / right), with the wearer simulating natural body collapse. Crucially, the negative class includes the events most likely to false-trigger the threshold-based front-end: sitting down hard, dropping into a chair, slamming a door, putting the arm down forcefully, jumping, and the various jolts that occur during walking and running. Each example was labelled with the activity tag that the device would have been emitting at that moment, so the model learns to interpret the same impact differently depending on context (a 3 g spike during *running* is a footstrike; the same spike during *stationary* is a real event).

### Impulse design

**Model A — activity classifier**
- Input: 7-channel IMU window at 100 Hz.
- DSP block: spectral-analysis (FFT magnitudes + RMS) over the window.
- Learn block: small fully-connected network.
- Output: 4 classes — labels returned **alphabetically** (`hand_motion, running, stationary, walking`), remapped to the firmware's internal order via `EI_TO_INTERNAL[]` in `main.cpp`.

**Model B — fall classifier**
- Input: 11-channel window — the same 7 IMU channels **plus a 4-element one-hot activity tag** appended to every sample. The tag is the live output of Model A, frozen at trigger time. This is what lets the same model treat a 3 g event during running very differently from a 3 g event while stationary.
- DSP block: raw / spectral features tuned during Studio experimentation.
- Learn block: 1-D conv net.
- Output: binary (`fall` vs `not_fall`), with `FALL_IDX = 0` matching the alphabetical label order.

### Training

Models were trained inside Edge Impulse Studio with the standard 80/20 train/validation split and a held-out test set. Training data augmentation was left at Studio defaults; the bulk of generalisation came from collecting genuinely diverse negative examples rather than augmentation tricks. The fall model's operating point was chosen from the precision-recall curve to favour recall, since the firmware adds two further independent gates (stillness + orientation change) downstream — false positives are filtered there, so the ML stage can be loose.

### Deployment

The Studio's **C++ library** export was unpacked into `src/merged/`, and both impulses are linked into a single binary using Edge Impulse's multi-impulse build. The firmware calls `process_impulse()` on each impulse handle (`impulse_handle_1003832_1` for activity, `impulse_handle_1004073_1` for fall), reusing pre-allocated feature buffers to avoid heap fragmentation. Inference is ~30–50 ms per window on the 80 MHz ESP32 — well inside the budget given the activity model runs at 1 Hz and the fall model only runs on a trigger event.

### Why the on-device pipeline is more than just the model

The ML output is one of three independent gates a fall must clear:

1. **Trigger gate** (Core 1, real-time): Path A = ≥30 ms freefall under 4 m/s² followed within 2 s by an impact above 2.2 g and 1.0 rad/s; Path B = high-G stumble (>3.5 g, >2.0 rad/s) without preceding freefall.
2. **ML gate** (Core 0): three windows at offsets 0 / 250 / 500 ms around the impact, majority-vote (2 of 3) above a confidence threshold (0.85, or 0.90 when activity is *hand motion* to suppress arm-gesture false positives).
3. **Posture gate**: after a 3 s settle, post-event acceleration magnitude must sit in [8, 11] m/s² for ≥38/50 samples (stillness), **and** EMA gravity must have rotated ≥35° from its pre-fall reference (orientation change). The pre-fall gravity reference is snapshotted *before* freefall begins so it isn't pulled toward zero by the low-G samples.

A confirmed fall requires all three. False positives from any single layer — a phantom freefall, a noisy ML window, a brief lie-down — get rejected by the others.

## Concurrency model

- **Core 1, priority 20** — `sampling_task`: 100 Hz IMU read, ring-buffer write, EMA gravity update, trigger detection. Hard real-time; uses `vTaskDelayUntil` and a 3 ms non-blocking I²C mutex try (falls back to last cached sample on contention).
- **Core 0, priority 10** — `background_task`: display, ML inference, slow sensors (HR/SpO₂/temp/battery), BLE telemetry, fall evaluation. Bumped above the BLE host's default priority 5 so fall evaluation isn't starved during BLE traffic.
- Shared state is annotated with its owner at the top of `main.cpp`. The ring buffer is protected by a portMUX spinlock, the I²C bus by a FreeRTOS mutex.

## BLE protocol

- Device name: `FallGuard`
- Service UUID: `f00dbabe-0001-1000-8000-00805f9b34fb`
- **Telemetry** (notify, 2 Hz): 19-byte packed `TelemetryPacket` — layout documented in `main.cpp`.
- **Event** (notify): `FALL` string on confirmed fall.
- **Command** (write): `alert_off` clears the red LED and buzzer.

Replace the UUIDs in `main.cpp` before deploying anything you don't want a stranger's app to pick up.

## License

No license file is included; treat the code as all-rights-reserved unless one is added.
