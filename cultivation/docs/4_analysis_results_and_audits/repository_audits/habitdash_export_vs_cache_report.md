# Habit Dash Export vs. Current Wellness Cache

**Flat file analyzed:** cultivation/data/raw/habitdash_export_2025-06-06/2025-06-06 Habit Dash - Integrations (flat file).csv

**Current cache:** cultivation/data/daily_wellness.parquet
**Useful Guide** https://wearipedia.readthedocs.io/en/latest/notebooks/whoop_strap_4.html
**Total canonical fields:** 89
**Fields with data (non-null):** 89
**Most complete fields:** ['heart_rate_variability_whoop', 'recovery_score_whoop', 'resting_heart_rate_whoop', 'deep_sleep_percent_whoop', 'deep_sleep_whoop']
**Least complete fields:** ['height_withings', 'weight_garmin', 'avg_heart_rate_withings', 'diastolic_blood_pressure_withings', 'systolic_blood_pressure_withings']

**Schema notes:**
- Canonical export field names are now used throughout.
- Cache is wide-format (one column per metric).
- Legacy columns removed.
- All export fields are present in cache.

## Canonical Field Inventory (Grouped by Category/Source)

### Activity

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| steps_garmin | Steps | Steps | 14079 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| active_time_garmin | Active Time | Minutes | 14079 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| activity_distance_garmin | Activity Distance | Miles | 14079 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| met_min_high_intensity_garmin | MET Min High Intensity | Minutes | 14074 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| time_resting_garmin | Time Resting | Minutes | 13844 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| avg_stress_level_garmin | Avg Stress Level | % | 14074 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| max_stress_level_garmin | Max Stress Level | % | 13907 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| met_min_medium_intensity_garmin | MET Min Medium Intensity | Minutes | 14074 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| medium_intensity_activity_garmin | Medium Intensity Activity | Minutes | 13843 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| low_intensity_activity_garmin | Low Intensity Activity | Minutes | 13824 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| total_activity_garmin | Total Activity | Minutes | 13843 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| high_intensity_activity_garmin | High Intensity Activity | Minutes | 13702 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| strain_score_whoop | Strain Score | nan | 18415 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |

#### Source: withings

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| floors_climbed_withings | Floors Climbed | Floors | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| total_activity_withings | Total Activity | Minutes | 2660 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| high_intensity_activity_withings | High Intensity Activity | Minutes | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| activity_distance_withings | Activity Distance | Miles | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| medium_intensity_activity_withings | Medium Intensity Activity | Minutes | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| low_intensity_activity_withings | Low Intensity Activity | Minutes | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |
| steps_withings | Steps | Steps | 7323 | time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals |

### Body

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| weight_garmin | Weight | Pounds | 52 | weight/body comp trends, BMI, health risk flags |
| fitness_age_garmin | Fitness Age | Years | 4148 | weight/body comp trends, BMI, health risk flags |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| skin_temperature_whoop | Skin Temperature | Fahrenheit | 18462 | weight/body comp trends, BMI, health risk flags |

#### Source: withings

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| weight_withings | Weight | Pounds | 3134 | weight/body comp trends, BMI, health risk flags |
| fat_mass_percent_withings | Fat Mass Percent | % | 3134 | weight/body comp trends, BMI, health risk flags |
| lean_mass_withings | Lean Mass | Pounds | 3134 | weight/body comp trends, BMI, health risk flags |
| fat_mass_withings | Fat Mass | Pounds | 3134 | weight/body comp trends, BMI, health risk flags |
| lean_mass_percent_withings | Lean Mass Percent | % | 3134 | weight/body comp trends, BMI, health risk flags |
| height_withings | Height | Feet | 51 | weight/body comp trends, BMI, health risk flags |

### Heart

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| min_heart_rate_garmin | Min Heart Rate | Beats / min | 13954 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| avg_heart_rate_garmin | Avg Heart Rate | Beats / min | 13954 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| max_heart_rate_garmin | Max Heart Rate | Beats / min | 13954 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| resting_heart_rate_garmin | Resting Heart Rate | Beats / min | 13954 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| max_heart_rate_whoop | Max Heart Rate | Beats / min | 18415 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| avg_heart_rate_whoop | Avg Heart Rate | Beats / min | 18415 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| heart_rate_variability_whoop | Heart Rate Variability | Milliseconds | 18573 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |
| resting_heart_rate_whoop | Resting Heart Rate | Beats / min | 18573 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |

#### Source: withings

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| avg_heart_rate_withings | Avg Heart Rate | Beats / min | 808 | resting/avg/max trends, HRV analysis, stress/fatigue modeling |

### Heart + Lungs

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| vo₂_max_garmin | VO₂ Max | mL/kg/min | 4148 | aerobic fitness, VO₂ max, blood oxygenation, altitude adaptation |
| blood_oxygenation_garmin | Blood Oxygenation | % | 2882 | aerobic fitness, VO₂ max, blood oxygenation, altitude adaptation |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| blood_oxygenation_whoop | Blood Oxygenation | % | 18287 | aerobic fitness, VO₂ max, blood oxygenation, altitude adaptation |

### Metabolism

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| active_calories_burned_garmin | Active Calories Burned | Calories | 14079 | caloric balance, activity vs. calories, weight change modeling |
| total_calories_burned_garmin | Total Calories Burned | Calories | 14074 | caloric balance, activity vs. calories, weight change modeling |
| resting_calories_burned_garmin | Resting Calories Burned | Calories | 14074 | caloric balance, activity vs. calories, weight change modeling |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| total_calories_burned_whoop | Total Calories Burned | Calories | 18415 | caloric balance, activity vs. calories, weight change modeling |

#### Source: withings

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| active_calories_burned_withings | Active Calories Burned | Calories | 7323 | caloric balance, activity vs. calories, weight change modeling |
| total_calories_burned_withings | Total Calories Burned | Calories | 7323 | caloric balance, activity vs. calories, weight change modeling |
| resting_calories_burned_withings | Resting Calories Burned | Calories | 7323 | caloric balance, activity vs. calories, weight change modeling |

### Recovery

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| body_battery_garmin | Body Battery | % | 13954 | readiness, recovery prediction, overtraining risk |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| recovery_score_whoop | Recovery Score | % | 18573 | readiness, recovery prediction, overtraining risk |

### Sleep Length

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| deep_sleep_garmin | Deep Sleep | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| time_awake_garmin | Time Awake | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| sleep_start_time_garmin | Sleep Start Time | 24-hour time | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| sleep_end_time_garmin | Sleep End Time | 24-hour time | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| time_in_bed_garmin | Time in Bed | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| deep_sleep_percent_garmin | Deep Sleep Percent | % | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| rem_sleep_percent_garmin | REM Sleep Percent | % | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| light_sleep_garmin | Light Sleep | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| rem_sleep_garmin | REM Sleep | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| light_sleep_percent_garmin | Light Sleep Percent | % | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |
| total_sleep_garmin | Total Sleep | Hours | 12924 | sleep duration trends, stage distribution, impact on recovery/performance |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| deep_sleep_percent_whoop | Deep Sleep Percent | % | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| deep_sleep_whoop | Deep Sleep | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| rem_sleep_percent_whoop | REM Sleep Percent | % | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| total_sleep_whoop | Total Sleep | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| rem_sleep_whoop | REM Sleep | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| time_in_bed_whoop | Time in Bed | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| sleep_start_time_whoop | Sleep Start Time | 24-hour time | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| time_awake_whoop | Time Awake | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| light_sleep_percent_whoop | Light Sleep Percent | % | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| sleep_end_time_whoop | Sleep End Time | 24-hour time | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| light_sleep_whoop | Light Sleep | Hours | 18510 | sleep duration trends, stage distribution, impact on recovery/performance |
| nap_count_whoop | Nap Count | Naps | 2493 | sleep duration trends, stage distribution, impact on recovery/performance |
| nap_sleep_whoop | Nap Sleep | Hours | 2493 | sleep duration trends, stage distribution, impact on recovery/performance |

### Sleep Quality

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| time_awake_percent_garmin | Time Awake Percent | % | 12924 | fragmentation, disturbances, onset/latency, correlation with next-day performance |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| sleep_disturbances__per__hour_whoop | Sleep Disturbances / Hour | Disturbances | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_disturbances_duration_whoop | Sleep Disturbances Duration | Minutes | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_onset_latency_whoop | Sleep Onset Latency | Minutes | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_cycles_whoop | Sleep Cycles | Cycles | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| time_awake_percent_whoop | Time Awake Percent | % | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_disturbances_whoop | Sleep Disturbances | Disturbances | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| time_asleep_percent_whoop | Time Asleep Percent | % | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_score_whoop | Sleep Score | % | 18510 | fragmentation, disturbances, onset/latency, correlation with next-day performance |
| sleep_consistency_whoop | Sleep Consistency | % | 17603 | fragmentation, disturbances, onset/latency, correlation with next-day performance |

### Sleep Vitals

#### Source: garmin

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| sleep_respiratory_rate_garmin | Sleep Respiratory Rate | Breaths / min | 12924 | respiratory rate trends, anomaly detection |
| sleep_blood_oxygenation_garmin | Sleep Blood Oxygenation | % | 2796 | respiratory rate trends, anomaly detection |

#### Source: whoop

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| sleep_respiratory_rate_whoop | Sleep Respiratory Rate | Breaths / min | 18510 | respiratory rate trends, anomaly detection |

### Vitals

#### Source: withings

| cache_col | name | units | non-null count | suggested analyses |
|-----------|------|-------|----------------|-------------------|
| diastolic_blood_pressure_withings | Diastolic Blood Pressure | mmHg | 966 | blood pressure tracking, cardiovascular risk |
| systolic_blood_pressure_withings | Systolic Blood Pressure | mmHg | 966 | blood pressure tracking, cardiovascular risk |

## Field Availability Matrix (high-level)

| Field | Non-null Count |
|-------|----------------|
| steps_garmin | 14079 |
| active_calories_burned_garmin | 14079 |
| active_time_garmin | 14079 |
| activity_distance_garmin | 14079 |
| weight_withings | 3134 |
| fat_mass_percent_withings | 3134 |
| lean_mass_withings | 3134 |
| fat_mass_withings | 3134 |
| lean_mass_percent_withings | 3134 |
| diastolic_blood_pressure_withings | 966 |
| systolic_blood_pressure_withings | 966 |
| max_heart_rate_whoop | 18415 |
| total_calories_burned_whoop | 18415 |
| avg_heart_rate_whoop | 18415 |
| strain_score_whoop | 18415 |
| deep_sleep_percent_whoop | 18510 |
| deep_sleep_whoop | 18510 |
| heart_rate_variability_whoop | 18573 |
| rem_sleep_percent_whoop | 18510 |
| total_sleep_whoop | 18510 |
| rem_sleep_whoop | 18510 |
| sleep_respiratory_rate_whoop | 18510 |
| sleep_disturbances__per__hour_whoop | 18510 |
| sleep_disturbances_duration_whoop | 18510 |
| time_in_bed_whoop | 18510 |
| sleep_onset_latency_whoop | 18510 |
| sleep_start_time_whoop | 18510 |
| sleep_cycles_whoop | 18510 |
| time_awake_whoop | 18510 |
| light_sleep_percent_whoop | 18510 |
| time_awake_percent_whoop | 18510 |
| sleep_disturbances_whoop | 18510 |
| recovery_score_whoop | 18573 |
| resting_heart_rate_whoop | 18573 |
| blood_oxygenation_whoop | 18287 |
| time_asleep_percent_whoop | 18510 |
| sleep_score_whoop | 18510 |
| skin_temperature_whoop | 18462 |
| sleep_end_time_whoop | 18510 |
| light_sleep_whoop | 18510 |
| nap_count_whoop | 2493 |
| nap_sleep_whoop | 2493 |
| sleep_consistency_whoop | 17603 |
| active_calories_burned_withings | 7323 |
| total_calories_burned_withings | 7323 |
| floors_climbed_withings | 7323 |
| total_activity_withings | 2660 |
| high_intensity_activity_withings | 7323 |
| resting_calories_burned_withings | 7323 |
| activity_distance_withings | 7323 |
| medium_intensity_activity_withings | 7323 |
| low_intensity_activity_withings | 7323 |
| steps_withings | 7323 |
| height_withings | 51 |
| avg_heart_rate_withings | 808 |
| body_battery_garmin | 13954 |
| min_heart_rate_garmin | 13954 |
| met_min_high_intensity_garmin | 14074 |
| time_resting_garmin | 13844 |
| avg_stress_level_garmin | 14074 |
| total_calories_burned_garmin | 14074 |
| max_stress_level_garmin | 13907 |
| avg_heart_rate_garmin | 13954 |
| max_heart_rate_garmin | 13954 |
| resting_calories_burned_garmin | 14074 |
| met_min_medium_intensity_garmin | 14074 |
| resting_heart_rate_garmin | 13954 |
| medium_intensity_activity_garmin | 13843 |
| low_intensity_activity_garmin | 13824 |
| total_activity_garmin | 13843 |
| weight_garmin | 52 |
| high_intensity_activity_garmin | 13702 |
| deep_sleep_garmin | 12924 |
| time_awake_percent_garmin | 12924 |
| time_awake_garmin | 12924 |
| sleep_start_time_garmin | 12924 |
| sleep_end_time_garmin | 12924 |
| time_in_bed_garmin | 12924 |
| sleep_respiratory_rate_garmin | 12924 |
| deep_sleep_percent_garmin | 12924 |
| rem_sleep_percent_garmin | 12924 |
| light_sleep_garmin | 12924 |
| rem_sleep_garmin | 12924 |
| light_sleep_percent_garmin | 12924 |
| vo₂_max_garmin | 4148 |
| fitness_age_garmin | 4148 |
| total_sleep_garmin | 12924 |
| blood_oxygenation_garmin | 2882 |
| sleep_blood_oxygenation_garmin | 2796 |

## Fields in Export Only (not in cache; ❌ need to backfill)


## Fields in Cache Only (legacy/orphaned; ⚠️ should be migrated/removed)

- resp_rate_garmin
- sleep_disturbances_per_hour_whoop
- date

---
- ✅ = field exists in both export and cache
- ❌ = field is new (not in cache, but available in export)
- ⚠️ = legacy/orphaned field in cache (not in export; should be migrated/removed)

## Next Analysis Opportunities (Unlocked by Full Data)

- Multi-device comparison: e.g., compare RHR, HRV, sleep between Garmin, Whoop, Withings.
- Longitudinal health trends: weight, HRV, sleep, activity over months/years.
- Machine learning: predict fatigue, recovery, or injury risk from full metric set.
- Cross-metric correlation: e.g., sleep quality vs. next-day activity, HRV vs. stress.
- Anomaly detection: flag outlier days for any metric.
- Custom dashboards: build performance, wellness, or risk dashboards using the full schema.
