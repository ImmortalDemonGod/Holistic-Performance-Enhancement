session_meta:
  date_local: "2025-05-17"
  utc_start: "2025-05-17T17:02:10Z"
  plan_id:  "W20-Sat-PlyoRun"
  planned_duration_min: 60        # ↓—use single integer for min
  planned_duration_max: 75        # ↑—optional: delete if importer balks
  actual_duration_min: 95
  misc_rest_breaks_min: 10        # explains gap between block totals & 95
  wellness_light: "Green"
  env_temp_c: null
  location:
    - "outdoor park (warm-up, drills, plyos, run, pull-ups)"
    - "home (calisthenics finish)"
  video_captured: "no"
  session_notes: >
    Warm-up extended (quad fatigue); cadence-focused run on Cameron loop;
    high-HR alert muted; metronome at 165–170 spm. Pull-ups done at the park,
    remainder of calisthenics at home.

session_rpe:
  upper_body: 9
  lower_body: 8
  core:       10                  # hollow hold to failure

# ───────────────────────────────── SET-BY-SET LOG ─────────────────────────
sets:
  # Running drills
  - {exercise: "High Knees", set: 1, distance_m: 25, weight_kg: 0}
  - {exercise: "High Knees", set: 2, distance_m: 25, weight_kg: 0}
  - {exercise: "A-Skips",    set: 1, distance_m: 25, weight_kg: 0}
  - {exercise: "A-Skips",    set: 2, distance_m: 25, weight_kg: 0}
  - {exercise: "C-Skips",    set: 1, distance_m: 25, weight_kg: 0}
  - {exercise: "B-Skips",    set: 1, distance_m: 25, weight_kg: 0}
  # Butt Kicks – skipped (documented so importer doesn't assume omission)

  # Plyometrics
  - {exercise: "Ankle Hops", set: 1, reps: 12, weight_kg: 0, rpe: 4}
  - {exercise: "Ankle Hops", set: 2, reps: 15, weight_kg: 0, rpe: 5}
  - {exercise: "Pogo Jumps", set: 1, reps: 18, distance_m: 30, weight_kg: 0}
  - {exercise: "Pogo Jumps", set: 2, reps: 29, distance_m: 38, weight_kg: 0}
  - {exercise: "Tuck Jumps", set: 1, reps: 3,  weight_kg: 0, rpe: 7}
  - {exercise: "Tuck Jumps", set: 2, reps: 4,  weight_kg: 0, rpe: 7}

  # Calisthenics
  - {exercise: "Pull-up",            set: 1, reps: 1, weight_kg: 0, rpe: 10,
     notes: "Max effort, chin over bar"}
  - {exercise: "Pull-up (attempt)",  set: 2, reps: 0, weight_kg: 0, rpe: 10,
     notes: "Failed – halfway up"}
  - {exercise: "Hamstring Walkouts", set: 1, reps: 10, weight_kg: 0, rpe: 7}
  - {exercise: "Single-Leg Romanian Deadlift",
     set: 1, reps: "7 each leg", weight_kg: 0, rpe: 6}
  - {exercise: "Single-Leg Glute Bridge",
     set: 1, reps: 10, weight_kg: 0, rpe: 7.5,
     notes: "Approximate reps; effort ~7–8"}
  - {exercise: "Hollow Body Hold",   set: 1, duration_s: 16, weight_kg: 0, rpe: 10}

# ────────────────────────── BLOCK-LEVEL SUMMARY ──────────────────────────
warmup_block:
  duration_min: 16
  notes: "Walk to park + dynamic mobility; extra quad/hip activation"

plyo_block:
  duration_min: 24
  notes: "Ankle hops, pogo jumps, tuck jumps; brief glute/ham activation"

run_block:
  duration_min: 17
  distance_km: null
  distance_source: "not recorded"
  cadence_spm_target: "165–170"
  cadence_spm_avg: null
  peak_hr_bpm: 183
  rpe: 7-8
  notes: >
    Cameron loop; cadence low-alert set to 165 spm; struggled early,
    adapted after alert change.

cooldown_block:
  duration_min: 10
  notes: "Walk home for HR down-regulation"

strength_block:
  duration_min: 18
  notes: "Calisthenics at home after hydration break"