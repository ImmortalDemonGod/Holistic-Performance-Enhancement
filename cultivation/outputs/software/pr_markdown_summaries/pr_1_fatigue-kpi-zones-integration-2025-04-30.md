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

## Git Commit Log

> No commit log found for this PR.

