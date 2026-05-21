#include "derived_metrics.h"
#include <math.h>

// ---------- Tunables ----------
#define STEP_AC_THRESHOLD     2.5f   // m/s² above gravity tracker = candidate step peak
#define STEP_REFRACTORY_SAMP  25     // 250ms at 100Hz — minimum spacing between peaks
#define STEP_EMA_ALPHA        0.02f  // ~500ms time constant for gravity tracker

// Posture tilt (deg from gravity vector pointing "up" along device +Z)
#define POSTURE_UPRIGHT_MAX   30.0f
#define POSTURE_RECLINED_MAX  60.0f
#define POSTURE_LYING_MAX     110.0f

// Sleep state transitions (consecutive seconds of restful conditions)
#define REST_TO_RESTING_S      120     // 2 min
#define REST_TO_LIGHT_S        600     // 10 min
#define REST_TO_DEEP_S         1800    // 30 min
#define ACTIVE_TO_AWAKE_S      30      // sustained motion drops back to AWAKE

// Cadence window
#define CADENCE_WINDOW_S       6

// HRV ring (1Hz samples)
#define HR_RING_LEN            60

// ---------- Fast-path state (sampling_task only) ----------
static float fp_ema           = 9.81f;
static int   fp_refractory    = 0;
static volatile uint32_t fp_step_count_raw = 0;

// ---------- Slow-path state (background_task only) ----------
static uint32_t sp_last_step_count = 0;
static uint16_t sp_step_history[CADENCE_WINDOW_S] = {0};
static uint8_t  sp_step_history_idx = 0;

static int      sp_hr_ring[HR_RING_LEN] = {0};
static uint8_t  sp_hr_ring_idx = 0;
static uint8_t  sp_hr_ring_fill = 0;
static int      sp_last_hr = 0;

static uint8_t  sp_sleep_state    = DM_AWAKE;
static uint16_t sp_rest_seconds   = 0;
static uint16_t sp_active_seconds = 0;

static DerivedMetrics sp_out = {};

// ---------- Public API ----------

void dm_init() {
    fp_ema = 9.81f;
    fp_refractory = 0;
    fp_step_count_raw = 0;
    sp_last_step_count = 0;
    sp_step_history_idx = 0;
    sp_hr_ring_idx = 0;
    sp_hr_ring_fill = 0;
    sp_last_hr = 0;
    sp_sleep_state = DM_AWAKE;
    sp_rest_seconds = 0;
    sp_active_seconds = 0;
    sp_out = {};
}

// Runs at 100Hz on Core 1. Single-writer for fp_* state — no lock needed.
// fp_step_count_raw is read non-atomically from Core 0; uint32 reads are atomic on Xtensa.
void dm_update_fast(float accMag) {
    fp_ema += STEP_EMA_ALPHA * (accMag - fp_ema);
    float ac = accMag - fp_ema;

    if (fp_refractory > 0) {
        fp_refractory--;
        return;
    }
    if (ac > STEP_AC_THRESHOLD) {
        fp_step_count_raw++;
        fp_refractory = STEP_REFRACTORY_SAMP;
    }
}

static uint8_t classify_posture(float gx, float gy, float gz) {
    float mag = sqrtf(gx*gx + gy*gy + gz*gz);
    if (mag < 1.0f) return DM_UPRIGHT;
    // tilt from +Z axis (device frame); same convention as the fall-orientation gate
    float c = gz / mag;
    if (c > 1.0f) c = 1.0f; else if (c < -1.0f) c = -1.0f;
    float tilt_deg = acosf(c) * 57.2958f;

    if (tilt_deg < POSTURE_UPRIGHT_MAX)   return DM_UPRIGHT;
    if (tilt_deg < POSTURE_RECLINED_MAX)  return DM_RECLINED;
    if (tilt_deg < POSTURE_LYING_MAX)     return DM_LYING;
    return DM_INVERTED;
}

static uint8_t compute_hrv_rmssd() {
    if (sp_hr_ring_fill < 4) return 0;
    // Iterate successive diffs in chronological order
    int prev = -1;
    float sum_sq = 0.0f;
    int n = 0;
    uint8_t count = sp_hr_ring_fill;
    uint8_t start = (sp_hr_ring_idx + HR_RING_LEN - count) % HR_RING_LEN;
    for (uint8_t i = 0; i < count; i++) {
        int v = sp_hr_ring[(start + i) % HR_RING_LEN];
        if (v <= 0) { prev = -1; continue; }
        if (prev > 0) {
            float d = (float)(v - prev);
            sum_sq += d * d;
            n++;
        }
        prev = v;
    }
    if (n == 0) return 0;
    float rmssd = sqrtf(sum_sq / (float)n);
    if (rmssd > 255.0f) rmssd = 255.0f;
    return (uint8_t)rmssd;
}

void dm_update_slow(int activity_idx, int hr_now, int hr_baseline, float gx, float gy, float gz) {
    // --- Posture ---
    uint8_t posture = classify_posture(gx, gy, gz);

    // --- Steps + cadence ---
    uint32_t raw = fp_step_count_raw;
    uint32_t delta32 = raw - sp_last_step_count;
    sp_last_step_count = raw;
    uint16_t delta = (delta32 > 0xFFFF) ? 0xFFFF : (uint16_t)delta32;

    // Only count toward step_count when walking or running. Other activities (hand motion,
    // stationary) often produce spurious peaks that aren't gait.
    bool is_gait = (activity_idx == 1 || activity_idx == 2);
    if (is_gait) {
        sp_out.step_count = (uint16_t)(sp_out.step_count + delta);
    }

    sp_step_history[sp_step_history_idx] = is_gait ? delta : 0;
    sp_step_history_idx = (sp_step_history_idx + 1) % CADENCE_WINDOW_S;
    uint32_t window_sum = 0;
    for (uint8_t i = 0; i < CADENCE_WINDOW_S; i++) window_sum += sp_step_history[i];
    // window covers CADENCE_WINDOW_S seconds → steps/min = window_sum * 60 / WINDOW_S
    uint32_t spm = (window_sum * 60u) / CADENCE_WINDOW_S;
    sp_out.cadence_spm = (spm > 255) ? 255 : (uint8_t)spm;

    // --- HRV ring (only push on new reading) ---
    if (hr_now > 0 && hr_now != sp_last_hr) {
        sp_hr_ring[sp_hr_ring_idx] = hr_now;
        sp_hr_ring_idx = (sp_hr_ring_idx + 1) % HR_RING_LEN;
        if (sp_hr_ring_fill < HR_RING_LEN) sp_hr_ring_fill++;
        sp_last_hr = hr_now;
    }
    sp_out.hrv_rmssd = compute_hrv_rmssd();

    // --- Resting HR (passed through from main's update_stress_level baseline) ---
    int rh = hr_baseline;
    if (rh < 0) rh = 0; else if (rh > 255) rh = 255;
    sp_out.resting_hr = (uint8_t)rh;

    // --- Sleep state machine ---
    bool stationary = (activity_idx == 0);
    bool horizontal = (posture == DM_LYING || posture == DM_RECLINED);
    bool hr_low     = (hr_now > 0 && hr_baseline > 30 && hr_now < (hr_baseline - 3));
    // HR is optional: if we have no baseline yet, don't require it.
    bool restful = stationary && horizontal && (hr_baseline < 30 || hr_low || hr_now == 0);

    if (restful) {
        if (sp_rest_seconds < 0xFFFE) sp_rest_seconds++;
        sp_active_seconds = 0;
    } else {
        sp_rest_seconds = 0;
        if (!stationary || posture == DM_UPRIGHT) {
            if (sp_active_seconds < 0xFFFE) sp_active_seconds++;
        }
    }

    if (sp_active_seconds >= ACTIVE_TO_AWAKE_S) {
        sp_sleep_state = DM_AWAKE;
    } else if (sp_rest_seconds >= REST_TO_DEEP_S) {
        sp_sleep_state = DM_DEEP_SLEEP;
    } else if (sp_rest_seconds >= REST_TO_LIGHT_S) {
        sp_sleep_state = DM_LIGHT_SLEEP;
    } else if (sp_rest_seconds >= REST_TO_RESTING_S) {
        sp_sleep_state = DM_RESTING;
    } else if (sp_active_seconds > 0) {
        sp_sleep_state = DM_AWAKE;
    }

    sp_out.posture     = posture;
    sp_out.sleep_state = sp_sleep_state;
}

DerivedMetrics dm_get() {
    return sp_out;
}
