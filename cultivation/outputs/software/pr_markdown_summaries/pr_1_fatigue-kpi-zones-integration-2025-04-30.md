# PR #1: Fatigue kpi zones integration 2025 04 30

- **Author:** ImmortalDemonGod
- **State:** MERGED
- **Created:** 2025-05-02 03:48
- **Closed:** 2025-05-05 23:06
- **Merged:** 2025-05-05 23:06
- **Base branch:** `master`
- **Head branch:** `fatigue-kpi-zones-integration-2025-04-30`
- **Files changed:** 203
- **Additions:** 135805
- **Deletions:** 8656

## Summary


<!-- This is an auto-generated comment: release notes by coderabbit.ai -->
## Summary by CodeRabbit

- **New Features**
  - Added new GitHub Actions workflows for fatigue monitoring, KPI metrics calculation, scheduling runs, and Habit Dash wellness data sync.
  - Introduced stride detection and fatigue KPI zone analysis in running analytics.
  - Added personal training zones configuration and enhanced run metrics with aerobic decoupling and training stress calculations.
  - Implemented GPX heart rate override using FIT files.
  - Added comprehensive training plan files for Base-Ox Week 2 with detailed session instructions and device alert settings.
  - Introduced new documentation on heart rate and pace zone models, running session audit templates, and Habit Dash integration.
  - Added new scripts for aggregating weekly runs, syncing wellness data, and PID scheduling.
  - Added extensive test coverage including metrics, GPX parsing, and walk utilities with sample data for edge cases.

- **Improvements**
  - Enhanced run performance analysis with wellness context, expanded weather data fetching with caching, and improved reporting outputs.
  - Updated training plans to emphasize heart rate zone adherence, cadence targets, and execution guidance.
  - Improved parsing and analytics robustness, including walk detection enhancements and handling of missing data.
  - Refined process_all_runs to prioritize override files, sync wellness data conditionally, and add weekly comparison steps.
  - Added persistent weather data caching and improved retry logic in weather utilities.
  - Added CLI options and verbosity control in multiple scripts.

- **Bug Fixes**
  - Fixed cadence scaling issues and corrected distance and speed calculations.
  - Improved error handling and fallback mechanisms in fatigue monitoring and weather fetching.

- **Chores**
  - Added new dependencies: pyarrow and python-dotenv.
  - Updated `.gitignore` to exclude `.png` files globally.
  - Added `.env.template` for environment variable examples.

- **Documentation**
  - Added new guides for testing, Habit Dash integration, training plan management, and running session audit templates.
  - Updated training plan documents with revised zone models and execution notes.

- **Tests**
  - Introduced new and expanded test modules for metrics, GPX parsing, and walk utilities ensuring reliability and correctness.
  - Added sample GPX files covering missing heart rate and cadence data scenarios.
<!-- end of auto-generated comment: release notes by coderabbit.ai -->

## Top-level Comments
- **coderabbitai**: <!-- This is an auto-generated comment: summarize by coderabbit.ai -->
<!-- This is an auto-generated comment: failure by coderabbit.ai -->

> [!CAUTION]
> ## Review failed
> 
> The pull request is closed.

<!-- end of auto-generated comment: failure by coderabbit.ai -->
<!-- walkthrough_start -->

... (truncated)

## CodeRabbit Walkthrough
## Walkthrough

Added an environment template and global `.gitignore` update. Introduced multiple GitHub Actions workflows for metrics gating, scheduling, fatigue monitoring, and wellness data sync. Revamped README with testing and QA guidance. Added extensive new data outputs and detailed run reports. Enhanced numerous scripts for parsing, metrics, scheduling, processing, and analysis. Added Habit Dash API integration utilities and synchronization scripts. Introduced stride and walk detection utilities. Added tests and new dependencies.

## Changes

| File(s)                                         | Summary                                                                                                                           |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `.env.template`                                 | New template with example environment variables (`WEATHER_API_KEY`, `OTHER_SECRET`).                                              |
| `.gitignore`                                    | Added global rule to ignore `*.png` files.                                                                                        |
| `.github/workflows/*.yml`                       | Added workflows: `fatigue-watch` (daily/manual triggers), `run-metrics` (KPI calc + gating), `schedule-next-week` (triggered post-metrics), and `sync-habitdash` (daily/manual). |
| `README.md`                                     | Replaced Troubleshooting with Testing & QA section; added CI badge, pytest instructions, environment variable and dependency management, and Next Steps. |
| `requirements.txt`                              | Added `pyarrow` and `python-dotenv` dependencies.                                                                                 |
| `tests/`                                        | Added `tests/README.md`, `conftest.py`, sample GPX (`sample_missing_hr_cadence.gpx`), and test modules for metrics, GPX parsing, and walk utilities. |
| `cultivation/data/status.json`, `zones_personal.yml` | Added `status.json` (phase info) and `zones_personal.yml` (personal HR/pace zone definitions).                                      |
| `cultivation/scripts/__init__.py`               | Added empty line (no functional change).                                                                                          |
| `cultivation/scripts/running/aggregate_weekly_runs.py` | New script to aggregate weekly running metrics from processed CSV files, outputting Parquet summary.                              |
| `cultivation/scripts/running/analyze_hr_pace_distribution.py` | Enhanced to include power and cadence distributions with plots and statistics output.                                             |
| `cultivation/scripts/running/detect_strides_temp.py` | New stride detection script based on pace and cadence thresholds, with CLI interface and summary output.                          |
| `cultivation/scripts/running/fatigue_watch.py`  | New fatigue monitoring script analyzing subjective and objective data, creating GitHub issues on fatigue alerts.                 |
| `cultivation/scripts/running/metrics.py`         | Enhanced GPX parsing to extract cadence and power; added personal zone loading; improved aerobic decoupling calculation; CLI for zone file regeneration and verbose debug. |
| `cultivation/scripts/running/override_gpx_hr_with_fit.py` | New script to override GPX heart rate data using FIT file HR samples, matching timestamps.                                         |
| `cultivation/scripts/running/parse_run_files.py` | Major enhancements: added walk detection, improved distance/speed calculations, verbose diagnostics, multiple output files, and CLI options. |
| `cultivation/scripts/running/pid_scheduler.py`   | New scheduling script to automate running session tasks per week with phase management and KPI gate check via GitHub API.         |
| `cultivation/scripts/running/process_all_runs.py`| Enhanced to sync wellness data, prioritize override GPX files, add planning IDs, detect marker files, and run weekly comparisons. |
| `cultivation/scripts/running/run_performance_analysis.py` | Refactored to use metrics module zones, added fatigue KPI zones, stride detection, improved HR drift filtering, wellness context loading, expanded weather data handling, and enhanced reporting and plotting. |
| `cultivation/scripts/running/walk_utils.py`      | New module for walk segment detection, GPS jitter filtering, segment summarization, and time-weighted pace calculation.           |
| `cultivation/scripts/running/weather_utils.py`   | Added persistent weather data caching, WMO code mapping, improved Open-Meteo API usage with caching and retries, returning raw JSON data. |
| `cultivation/scripts/sync_habitdash.py`           | New script to sync wellness metrics from Habit Dash API into a Parquet cache file with data merging and logging.                   |
| `cultivation/scripts/utilities/field_id_discovery.py` | New utility to discover Habit Dash API field IDs for priority metrics across sources.                                             |
| `cultivation/scripts/utilities/habitdash_api.py`  | New Habit Dash API client module with rate limiting, error handling, and methods to fetch fields and metric data.                 |
| `cultivation/scripts/running/debug_process_runs.sh` | New bash script to clean output directories and rerun all runs processing for debugging.                                          |
| `cultivation/outputs/figures/**`                  | Deleted legacy run outputs; added detailed run metrics, distributions, drift analyses, pacing and fatigue zone summaries across weeks 17–18; updated compare files. |
| `cultivation/outputs/reports/2025_05_01_run_report.md` | Added detailed runner profile and baseline model report with recommendations and monitoring plan.                                 |
| `cultivation/docs/4_analysis/running_analysis_template.md` | Added optimized audit template for running session analysis with detailed prompt and instructions.                                |
| `cultivation/docs/habitdash_integration.md`       | Added documentation for Habit Dash API integration, data flow, usage, automation, and troubleshooting.                            |
| `cultivation/outputs/training_plans/base_ox_block.md` | Updated weekly micro-cycle blueprint with HR% zones, cadence targets, instrumentation, and next steps.                            |
| `cultivation/outputs/training_plans/pace-zones.md` | Added detailed reference for heart rate and pace zone models used in training blocks with switching instructions.                 |
| `cultivation/outputs/training_plans/baseox_daily_plans/week2/*.md` | Added multiple Week 2 daily training plan markdown files with session details and execution checklists.                            |
| `cultivation/outputs/training_plans/baseox_daily_plans/week2/Sat_2025-05-10_LONG‑RUN.md` | Added long-run session plan for durability and fuel pathway efficiency.                                                           |

## Sequence Diagram(s)

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant GH as GitHub Actions
    participant Metrics as run-metrics
    participant Scheduler as schedule-next-week
    participant PID as pid_scheduler

    Dev->>GH: push to main
    GH->>Metrics: trigger on push to processed data
    Metrics->>Metrics: checkout, install, run aggregate_weekly_runs, calc KPIs
    Metrics->>Metrics: set outputs (ef, drift, gate_fail)
    alt gate_passed
        Metrics-->>Scheduler: workflow_run (success)
        Scheduler->>Scheduler: checkout, setup Python, run pid_scheduler.py
        Scheduler->>PID: schedule tasks
    else gate_failed
        Metrics-->>Dev: fail workflow
    end
```

```mermaid
sequenceDiagram
    participant Scheduler as GitHub (cron)
    participant FW as fatigue-watch
    participant Script as fatigue_watch.py
    participant GHIssue as GitHub Issue API

    Scheduler->>FW: daily at 06:00 UTC
    FW->>Script: checkout, setup, run fatigue_watch.py
    Script->>Script: analyze HRV, RHR, sRPE, pain, Saturday run
    alt fatigue_detected
        Script->>GHIssue: create issue "fatigue-alert"
    else no_fatigue
        Script-->>FW: exit
    end
```

## Possibly related PRs

- ImmortalDemonGod/Holistic-Performance-Enhancement#1: Introduces the same `.env.template` and `fatigue-watch.yml` workflow, indicating overlapping scope and shared code additions.

## Poem

> 🐰  
> I hopped through code both old and new,  
> Env templates shining bright as dew.  
> Workflows cranked and data spun,  
> Scripts enhanced with every run.  
> Tests now guard each clever tweak—  
> A rabbit’s cheer for progress sleek!  
> 🎉

## Git Commit Log

```text
* 77f1d48 (origin/fatigue-kpi-zones-integration-2025-04-30, fatigue-kpi-zones-integration-2025-04-30) 📝 docs(week2): update training plan for May 9, 2025
* 21e1029 📝 docs(walk_utils): update docstring for filter_gps_jitter function
* 6daac73 ♻️ refactor(pid_scheduler): improve path configuration and environment variable access
* e46b204 Update training data files and add new workout metrics for week 18
* 609c267 Add new run data files and update pace/HR zone metrics for base training
* ac82907 🔧 chore(data): remove outdated files and documentation
* 51f2f80 ✨ feat(performance_analysis): add wellness context loading
* 1ef2b34 ✨ feat(debug): add comparison plots and processing logs
* 140c7f5 ✨ feat(training_plans): add Base-Ox mesocycle documentation
* 536b0ae ✨ feat(performance_analysis): add fatigue KPI zones and recommendations
* a84d556 ✨ feat(pid_scheduler): add CLI support for week selection and dry-run
* 83f5381 ✨ feat(parse_run_files): enhance walking segment data processing
* 3f5c399 ♻️ refactor(fatigue_watch): clean up imports and enhance error handling
* 5a9a6d6 📝 docs(detect_strides_temp): add detailed documentation and usage
* 46b51b8 ✨ feat(sync_habitdash): enhance data synchronization process
* 153657c ♻️ refactor(walk_utils): improve GPS jitter filtering logic
* f771524 ✨ feat(process_all_runs): add wellness data syncing before processing runs
* cd50cc0 ✨ feat(scripts): add debug script for reprocessing runs
* fa41785 📝 docs(zones_personal): update pace values for training zones
* 062169c ✨ feat(data): add pid lookup data for 2025 week 1
* 4464340 ✨ feat(workflows): enhance metrics calculation and KPI checks
* 2b1548c ✨ feat(weather_utils): add weather description function
* 4a456a8 ✅ test(tests): update error type assertions in tests
* 4396427 📝 docs(analysis): add optimized cultivation run analyzer template
* 77f4fe2 ✨ feat(workflow): add Habit Dash wellness data sync workflow
* 7efc5e8 ✨ feat(data): update personal zones configuration
* 84bd77f Update HR zones and add model type to zones config file
* 7bf799f Rename run files and update run report with calibration data and Base-Ox week 1 metrics
* 7723aa5 Update run data files with new calibration and BaseOx workout data
* 90a52c3 updated text files
* e637438 ✨ feat(cultivation): add initial cultivation module and update zones
* 23a9f38 ✨ feat(docs): add heart-rate & pace zones reference document
* 1e642d0 ✨ feat(cultivation): add walk summary report
* 72efcb8 📝 docs(tests): add testing guide for cultivation project
* 782365e ✨ feat(debug): add debug command for output management
* 6249e3d ✨ feat(scripts): add metrics and walk utilities for analysis
* ba761fd ✅ test(tests): add unit tests for walk_utils functions
* 0512f1f ✨ feat(week17): add walk segments CSV for evening run
* f1e68fc ✨ feat(week17): add run-only summary file
* b529e6e ✨ feat(week17): add summary files for evening and afternoon runs
* 96df7ca ✨ feat(cultivation): add new walk and run summaries with metrics
* 03c4359 data update
* cac8646 🔧 chore(files): remove compressed and uncompressed output files
* 646235e 📝 docs(README): update testing and quality assurance section
* 919e425 ✨ feat(env): add example environment variables template
* fc0d630 ✨ feat(cultivation): add GPX summary walk files for multiple runs
* bd8b157 ✨ feat(data): add initial data files for cultivation
* 232dade ✨ feat(metrics): add argparse for command line options
* 049a5a5 ✨ feat(data): add personal heart rate zones configuration
* d10ba4c ✨ feat(workflows): add scheduler for next week metrics
* 4877774 ✨ feat(data): add new subjective data tracking
* a1557d0 ✨ feat(pid_scheduler): enhance scheduling logic and phase management
* deb19aa ✨ feat(week18): add new data files for analysis
* 5a270ad ✨ feat(fatigue_watch): enhance fatigue tracking with new alerts
* dcd21df ✅ test(tests): add test for parsing GPX with missing HR/cadence
* 6f41193 ✨ feat(run_summary): add run summary for May 1 session
* 62b3728 ✨ feat(cultivation): update run analysis data for week 17 and 18
* 390d272 ✨ feat(analysis): update pace and power distribution metrics
* d3ad40b 🐛 fix(hr_distribution): update heart rate statistics
* c9f1c5e 🔧 chore(run_summary): update run analysis data
* 48e56c8 ♻️ refactor(scripts): improve data processing in aggregate_weekly_runs.py
* 1135a09 data update
* 58c6f24 🔧 chore(advanced_metrics): remove obsolete metrics files
* 34bd65f ✨ feat(metrics): add advanced and run summary metrics files
* 18eb68d ✨ feat(performance): enhance performance analysis metrics
* eee19b3 ✅ test(metrics): add tests for training zones and metrics
* cc48a57 ✨ feat(workflow): add KPI gate checks for metrics
* 387b134 🔧 chore(.gitignore): add PNG files to ignore list
* 624e1ec ✨ feat(metrics): add personal training zones computation
* 889875a ✨ feat(parse_run_files): enhance GPX parsing with temperature
* 10f5a5f data update
* 082e613 ✨ feat(github workflows): add fatigue watch and metrics workflows
* 2321418 📝 docs(zones_personal): update HR & pace bands documentation
* 0f2b9ba ✨ feat(pid_scheduler): add FIT and GPX file parsing functionality
* 1bb9c53 ✨ feat(week18): add stride summary and comparison files
* 1752a01 ✨ feat(data): add subjective data and personal zones configuration
* 72580a2 ✨ feat(scripts): add weekly runs aggregation script
* b7d9c9d ✨ feat(metrics): add metrics computation and testing
* c31e081 ✨ feat(cultivation): add new analysis text files for pacing strategy
* 9612a2c ✨ feat(cultivation): add new run analysis files
* 0a417ad ✨ feat(cultivation): add detailed performance analysis files
* cf1d397 📦 chore(week17): add cadence distribution and run summary files
* b317873 ✨ feat(cultivation): add stride and zone summary files
* e87a042 ✨ feat(week17): add cadence distribution data and update run summary
* 07765cd ✨ feat(metrics): add cadence and power metrics to GPX parsing
* f7b4736 ✨ feat(scripts): add fatigue watch script for monitoring
* b501905 ✨ feat(cultivation): add script to aggregate weekly runs
* db89d91 ✨ feat(cultivation): add power and cadence analysis
* cf4b869 ✨ feat(cultivation): add new heart rate and pace analysis files
* 5e8da4a data update
* 9da71a0 ✨ feat(running): add stride detection functionality
* 612a85e ✨ feat(performance_analysis): enhance weather data handling
* a9ce6fb ✨ feat(running): allow explicit reprocessing of runs
* 38609dc ✨ feat(weather): improve weather fetching mechanism
* f0f062b data update
* da7d50e ✨ feat(scripts): add GPX HR override with FIT data
* 68bc782 🔧 chore(week18): remove outdated HR and pace analysis files
* 5174eec 📝 docs(run_summary): add advanced metrics and weather details
* 00f05c1 data update
* ca2dbf0 ✨ feat(reports): add detailed run report for April 2025
* 1cbe261 ✨ feat(weather): add weather fetching utility
* acfd33d ✨ feat(performance_analysis): add advanced metrics and weather info
* f811b63 ✨ feat(running): skip already processed run files
* 6004b58 ✨ feat(parse_run_files): integrate advanced metrics for GPX
* 6c6f31b ✨ feat(metrics): add GPX parsing and run metrics calculation
* 72eb7ce ✨ feat(requirements): add requests package to dependencies
* 6d0d4dd 📝 docs(base_ox_block): update Base-Ox mesocycle documentation
* b28316e ✨ feat(docs): add Base-Ox Mesocycle training plan
* 6b2b77a ✨ feat(performance_analysis): enhance output organization and summaries
* ebcb547 ✨ feat(compare_weekly_runs): add image and text output for comparisons
* f92bbe8 ✨ feat(analyze_hr_pace_distribution): add image and text output directories
* 717b8d6 ✨ feat(cultivation): add pace comparison for week 17
* 1fcae2d ✨ feat(cultivation): add heart rate comparison for week 17
* 3aa850c ✨ feat(cultivation): add time in heart rate zone file
* f3ccfb1 ✨ feat(cultivation): add run summary output file
* f7eadf6 ✨ feat(cultivation): add pacing strategy analysis output
* a71ebcb ✨ feat(cultivation): add pace distribution output file
* 42e85e7 ✨ feat(cultivation): add heart rate vs pace correlation data
* 84cf549 ✨ feat(cultivation): add heart rate drift analysis output
* 7543576 ✨ feat(figures): add heart rate distribution data file
* 4123cb0 ✨ feat(cultivation): add time in heart rate zones data
* d7d7a1a ✨ feat(cultivation): add run summary output file
* bc95e1e ✨ feat(cultivation): add pace over time analysis file
* 683ed8e ✨ feat(cultivation): add pace distribution data file
* 79d4093 ✨ feat(cultivation): add heart rate vs pace correlation data
* deec77b ✨ feat(cultivation): add heart rate drift analysis output
* f57e45e ✨ feat(cultivation): add heart rate distribution data file
* cc349c5 🔧 chore(.gitignore): update ignore rules for figures
* 37faeba ✨ feat(performance_analysis): add dynamic figure directory creation
* a1b62e5 ✨ feat(scripts): add weekly comparison step for runs
* aaea7f2 ✨ feat(cultivation): add weekly run comparison script
* b5b320e ✨ feat(analyze_hr_pace_distribution): add figure saving directory structure
* a39538b updated files
* a328e1b ✨ feat(running): update paths in process_all_runs script
* 71abbee 📝 docs(README): add quick start guide for automated data analysis
* c447cbe 🔧 chore(.gitignore): add ignore rules for generated figures
* d54d06e ♻️ refactor(process_all_runs): update project root path
* 6bf37a1 ♻️ refactor(scripts): improve file renaming and processing logic
* ac3e359 ✨ feat(docs): add automated running data ingestion workflow
* 80e5b07 🔧 chore(create_structure): remove create_structure.py file
* 231afbb ✨ feat(requirements): add new data visualization libraries
* 607d9eb ✨ feat(performance_analysis): add advanced run performance analysis script
* bc39215 ✨ feat(scripts): add batch processing for running data files
* ceb502b ✨ feat(scripts): add file parser for FIT and GPX formats
* 71a22c3 ✨ feat(scripts): add auto-rename functionality for raw files
* d5de4cb ✨ feat(scripts): add HR and pace distribution analysis tool
* dbcd84d ✨ feat(reports): add placeholder file for reports directory
* 0fe43f5 ✨ feat(figures): add time in hr zone figure
* 655a5a9 ✨ feat(figures): add pace over time figure
* 693781b ✨ feat(figures): add pace distribution figure
* f0c9cce ✨ feat(figures): add heart rate vs pace hexbin plot
* f5437ce ✨ feat(figures): add HR over time drift figure
* 77bce6e ✨ feat(figures): add heart rate distribution figure
* 9c6a442 ✨ feat(figures): add placeholder for figures output directory
* 308bf12 new run data
* b6bda67 ✨ feat(data): add placeholder file for raw data directory
* 0c25807 new running data
* 3666a6e ✨ feat(processed): add placeholder file for processed data
* 3a137ba ✨ feat(requirements): add initial requirements file
* 035a68e Create systems‑map_and_market‑cheatsheet.md
* ddf2f9c Create system_readiness_audit_2025‑04‑18.md
* 431aae5 Create operational_playbook.md
* e45ef98 Rename Testing-requirements.md to  flashcards_2.md
* b9fb65c Create flashcards_1.md
* 047bc11 Create literature_system_overview.md
* 083e7ce Update design_overview.md
* eacb6de Update Progress.md
* c0f67d9 Update Progress.md
* 842e60c Rename biology_eda.ipynb to malthus_logistic_demo.ipynb
* 52719d5 Update Progress.md
* 85a45aa Update task_master_integration.md
* 94772b8 Create task_master_integration.md
* 45ec03d Update analysis_overview.md
* a65fb4d Create Progress.md
* bdab714 Rename Testing-requirements to Testing-requirements.md
* 2f2cc29 Create lean_guide.md
* 3a732a2 Create roadmap_vSigma.md
* 5e26925 Create math_stack.md
* e6cbfad Create generate_podcast_example.py
* d927c22 🔧 chore(notebooks): update metadata for biology_eda notebook
* a950c52 📝 docs(outline): add detailed framework for raising potential and leveraging synergy
* 2ae9c1a Create Testing-requirements
* 356e119 Rename section_1_test to section_1_test.md
* adb08fa Create section_1_test
* 6f489ac 📝 docs(biology_eda): add detailed explanation and examples
* 0077451 Add Chapter 1: Continuous Population Models for Single Species under docs/5_mathematical_biology
* 2d6a05e Update README.md
* 7619853 keeping the repo txt up to date
* 78c8b04 inital repo commit with all the current documentation and repo structure
* 14b05d7 Initial commit
```

