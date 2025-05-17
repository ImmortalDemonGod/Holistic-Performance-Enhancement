# Strength Data Schemas

## strength_sessions.parquet
| Column                        | Type      | Description                                             |
|-------------------------------|-----------|---------------------------------------------------------|
| session_id                    | string    | PK, e.g., YYYYMMDD_HHMMSS_strength_focus                |
| session_datetime_utc          | timestamp | Start time of session                                   |
| plan_id                       | string    | FK to planned workout file (e.g., Week20_Tue_ECONOMY_DEV_1) |
| wellness_light                | string    | Green/Amber/Red                                         |
| overall_rpe_upper_body        | float     | RPE 0-10                                                |
| overall_rpe_lower_body        | float     | RPE 0-10                                                |
| overall_rpe_core              | float     | optional RPE for core                                   |
| session_duration_planned_min  | int       | Planned duration (min)                                  |
| session_duration_actual_min   | int       | Actual duration (min)                                   |
| environment_temp_c            | float     | optional                                                |
| location_type                 | string    | e.g., home, gym                                         |
| video_captured                | bool      | Was video recorded?                                     |
| session_notes                 | string    | General session notes                                   |

## strength_exercises_log.parquet
| Column               | Type    | Description                                                |
|----------------------|---------|------------------------------------------------------------|
| log_id               | string  | PK                                                        |
| session_id           | string  | FK to strength_sessions.parquet                           |
| exercise_name        | string  | Standardized from exercise_library                         |
| set_number           | int     | Set index                                                  |
| reps_planned         | int     | optional                                                   |
| reps_actual          | int     | Actual reps                                                |
| weight_kg_planned    | float   | optional                                                   |
| weight_kg_actual     | float   | Actual weight                                              |
| distance_m_actual    | float   | e.g., jumps                                                |
| duration_s_actual    | int     | e.g., timed holds                                          |
| rpe_set              | float   | RPE 0-10                                                  |
| rir_set              | int     | Reps in Reserve                                           |
| rest_s_after_set     | int     | optional                                                  |
| is_warmup_set        | bool    | Default False                                             |
| is_failure_set       | bool    | Default False                                             |
| substitution_reason  | string  | If exercise subbed from plan                              |
| set_notes            | string  | Qualitative notes for the set                             |

## exercise_library.csv
| Column                  | Type    | Description                                   |
|-------------------------|---------|-----------------------------------------------|
| exercise_name           | string  | PK, e.g., "Barbell Back Squat"              |
| exercise_alias          | string  | semicolon-separated list of aliases           |
| primary_muscle_group    | string  | e.g., Quads, Chest                           |
| secondary_muscle_groups | string  | comma-separated                              |
| exercise_type           | string  | Compound, Isolation, Plyometric, Drill, etc. |
| movement_pattern        | string  | e.g., Squat, Hinge, Push_Horizontal          |
| equipment_needed        | string  | e.g., Barbell, Bodyweight                    |
| unilateral              | bool    |                                               |
| metric_type             | string  | reps_x_weight, duration, reps_only            |
