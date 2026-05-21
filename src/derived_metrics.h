#pragma once
#include <stdint.h>

// Tier-1 derived metrics computed on top of existing IMU + HR pipeline.
// Telemetry packet format (BLE) is documented in main.cpp where the packet is assembled.

enum DmSleepState : uint8_t { DM_AWAKE = 0, DM_RESTING = 1, DM_LIGHT_SLEEP = 2, DM_DEEP_SLEEP = 3 };
enum DmPosture    : uint8_t { DM_UPRIGHT = 0, DM_RECLINED = 1, DM_LYING = 2, DM_INVERTED = 3 };

struct DerivedMetrics {
    uint8_t  sleep_state;
    uint8_t  posture;
    uint16_t step_count;
    uint8_t  cadence_spm;
    uint8_t  hrv_rmssd;
    uint8_t  resting_hr;
};

void dm_init();

// Called from sampling_task at 100Hz immediately after the ring-buffer write.
// Must stay cheap — runs on the real-time core.
void dm_update_fast(float accMag);

// Called from background_task once per second.
//   activity_idx : 0=stationary,1=walking,2=running,3=hand_motion
//   hr_now       : current HR in bpm, or 0 if no valid reading
//   hr_baseline  : current resting HR estimate (0 if uncalibrated)
//   gx,gy,gz     : EMA gravity vector snapshot
void dm_update_slow(int activity_idx, int hr_now, int hr_baseline, float gx, float gy, float gz);

DerivedMetrics dm_get();
