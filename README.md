# 🧙🏾‍♂️ Holistic Performance Enhancement (Cultivation)

Welcome to the **Holistic Performance Enhancement** project repository! This initiative aims to integrate multiple domains—**Running**, **Biology**, **Software Development**, and **Synergy**—to enhance overall performance through data-driven insights and a structured framework.

![Cultivation Banner](path_to_your_banner_image)

## 🚀 Quick Start: Automated Running Data Analysis

You can now analyze your running data with a fully automated, one-click workflow:

1. **Drop your raw files** (from any device) into `cultivation/data/raw/`. Files can have any name (e.g., `activity.fit`, `export.gpx`, `run1.fit`, etc.).
2. **Run the batch processing script from the project root:**
   ```bash
   .venv/bin/python cultivation/scripts/running/process_all_runs.py
   ```
   - No arguments are needed for standard use.
   - The script will:
     - Automatically rename all files using the correct date and time from their metadata.
     - Process and analyze every `.fit` and `.gpx` file found.
     - Save all outputs (summaries, plots, analytics) in the appropriate folders.
     - Print a summary and recommendations for each run.

**You do not need to run any other script or specify any arguments.**
Just drop your files and run the batch script—everything else is handled for you.

### Output Locations
- **Processed summaries:** `cultivation/data/processed/`
- **Figures and plots:** `cultivation/outputs/figures/`

### Advanced Usage
If you want to override the default directories, you can use:
```bash
.venv/bin/python cultivation/scripts/running/process_all_runs.py \
  --raw_dir <raw_dir> \
  --processed_dir <processed_dir> \
  --figures_dir <figures_dir>
```
But for most users, the defaults will work out of the box!

## 📁 Repository Structure

A well-organized repository is key to maintaining clarity and efficiency. Below is the proposed directory structure for the Cultivation project, along with brief explanations for each component.

```
cultivation/
├── docs/
│   ├── 1_background/
│   ├── 2_requirements/
│   ├── 3_design/
│   └── 4_analysis/
├── data/
│   ├── running/
│   ├── biology/
│   ├── software/
│   └── synergy/
├── scripts/
│   ├── running/
│   ├── biology/
│   ├── software/
│   └── synergy/
├── notebooks/
│   ├── running/
│   ├── biology/
│   ├── software/
│   └── synergy/
├── ci_cd/
│   └── ...
├── README.md
└── .gitignore
```

### 📄 Detailed Directory Descriptions

#### 1. `docs/` - Documentation
- **`1_background/`**  
  Contains high-level discussions, notes, background readings, and references. Convert initial conversations into structured Markdown files (e.g., `potential_overview.md`, `domains_background.md`).

- **`2_requirements/`**  
  Gathers functional and technical requirements for each domain, such as synergy measurement plans, data formats, and tooling constraints (e.g., Python libraries).

- **`3_design/`**  
  Stores architectural and process design documents detailing how running data, biology experiments, and coding metrics interconnect. Includes flowcharts, synergy measurement formulas, and UML diagrams.

- **`4_analysis/`**  
  Documents results from initial experiments or baseline studies. Each synergy test or domain-specific experiment has its own Markdown file with summaries, data references, and conclusions.

#### 2. `data/` - Data Management
- **Structure**: Organized by domain to separate raw and processed data.
  - **`running/`**: CSV/JSON files for run logs, pace data, heart rate metrics.
  - **`biology/`**: Research logs, quiz/test results, raw data sets from lab experiments.
  - **`software/`**: Code metrics like commit logs, bug counts, lint/test coverage outputs.
  - **`synergy/`**: Combined datasets merging variables from multiple domains (e.g., merged CSV with run data and commit frequency).

#### 3. `scripts/` - Automation Scripts
- **Purpose**: Contains Python or Bash scripts for data cleaning, analysis, and synergy calculations.
  - **`running/`**: Scripts to parse wearable device logs, calculate weekly aggregates, run regression models on pace vs. VO₂ data.
  - **`biology/`**: Scripts for summarizing paper readings, analyzing quiz performance, scraping research databases.
  - **`software/`**: Automation scripts for analyzing commit logs, testing coverage, generating summaries.
  - **`synergy/`**: Specialized scripts that cross-reference running performance with code quality improvements.

#### 4. `notebooks/` - Jupyter Notebooks
- **Purpose**: Interactive notebooks for exploratory data analysis, visualization, and synergy prototyping.
  - **`running/`**: Visualizing run pace improvements, discovering correlations.
  - **`biology/`**: Analyzing reading logs, quiz scores, research data sets.
  - **`software/`**: Exploring commit or code review metrics, identifying trends over time.
  - **`synergy/`**: Consolidating data from all domains to test synergy hypotheses (e.g., running’s impact on coding output).

#### 5. `ci_cd/` - Continuous Integration/Continuous Deployment
- **Purpose**: Configuration files for CI/CD pipelines (e.g., GitHub Actions, Jenkins) that automate tests and data analyses on commits or schedules.
  - **Example**: GitHub Actions YAML file that triggers synergy scripts on new data pushes and updates a dashboard with results.

#### 6. `README.md` - Project Overview
- **Purpose**: Provides a high-level overview of the repository, setup instructions, contribution guidelines, and licensing information.

#### 7. `.gitignore` - Git Configuration
- **Purpose**: Specifies files and directories to ignore in Git (e.g., large data logs, secret tokens, environment files, script outputs).

## 🏃 Automated Running Data Ingestion & Analysis Workflow

This project supports robust, scalable ingestion and analysis of running data files (.fit and .gpx) with zero manual intervention required—even if your device exports files with generic names like `activity.fit` or `run1.gpx`.

### How It Works

1. **Drop your raw files** (from any device) into `cultivation/data/raw/`. Files can have any name (e.g., `activity.fit`, `export.gpx`, `run1.fit`, etc.).
2. **Run the batch processing script:**
   ```bash
   cd cultivation/scripts/running/
   python3 process_all_runs.py
   ```
   - This will automatically:
     - Detect and auto-rename any generic file names to a descriptive format (e.g., `20250427_auto.fit`), extracting the activity date from file content.
     - Parse each run file into a summary CSV in `cultivation/data/processed/`.
     - Generate all plots and advanced analytics, saving them in `cultivation/outputs/figures/`, all named after their corresponding run.

### Output Naming Conventions

- **Raw data:** `cultivation/data/raw/YYYYMMDD_<label>.fit` or `.gpx` (auto-renamed if needed)
- **Processed:** `cultivation/data/processed/YYYYMMDD_<label>_fit_summary.csv` (or `_gpx_summary.csv`)
- **Figures:** `cultivation/outputs/figures/YYYYMMDD_<label>_<analysis>.png`

### Adding New Runs

- You can add any number of new files to `data/raw/` at any time. The pipeline will process only new or renamed files, with no risk of overwriting or confusion.
- All outputs are organized and traceable by run and date.

### Troubleshooting

- If a file cannot be auto-renamed (e.g., missing date in metadata), it will be skipped and a message will be printed.
- For best results, use devices that embed timestamps in FIT/GPX files.

## 🛠️ Requirements

- Python 3.8+
- Install dependencies:
  ```bash
  pip install fitdecode gpxpy pandas numpy matplotlib seaborn haversine
  ```
  (You may also use a `requirements.txt` file if provided.)

## 📜 Script Overview

- **auto_rename_raw_files.py**: Renames generically named `.fit`/`.gpx` files in `data/raw/` to a date-based format using file metadata.
- **parse_run_files.py**: Parses a `.fit` or `.gpx` file to a summary CSV with computed metrics.
- **analyze_hr_pace_distribution.py**: Generates heart rate and pace distribution plots from a summary CSV.
- **run_performance_analysis.py**: Produces advanced analytics (training zones, HR drift, pacing) and plots from a summary CSV.
- **process_all_runs.py**: Orchestrates the full pipeline—auto-renames files, parses all runs, and generates all analyses/plots in batch.

## 🗂️ Example Output Structure

After running the batch script, your directories will look like:

```
cultivation/data/raw/
    20250427_auto.fit
    20250426_evening.gpx
cultivation/data/processed/
    20250427_auto_fit_summary.csv
    20250426_evening_gpx_summary.csv
cultivation/outputs/figures/
    20250427_auto_hr_distribution.png
    20250427_auto_pace_distribution.png
    20250427_auto_hr_vs_pace_hexbin.png
    20250427_auto_time_in_hr_zone.png
    20250427_auto_hr_over_time_drift.png
    20250427_auto_pace_over_time.png
    20250426_evening_hr_distribution.png
    ...
```

## ⚙️ Extending & Customizing

- **Add new analysis scripts:** Follow the argument pattern (`--input`, `--output`, `--prefix`) and add your script to `process_all_runs.py`.
- **Change output locations:** Adjust the directories in `process_all_runs.py`.
- **Change naming conventions:** Edit the prefix logic in the batch script and/or renaming script.

## 🧪 Testing and Quality Assurance

This repository uses `pytest` for unit testing. To run the tests:

```bash
pytest
```

A sample test for GPX parsing with missing HR/cadence data is included in `tests/test_parse_gpx.py`.

### Continuous Integration

![CI](https://github.com/ImmortalDemonGod/Holistic-Performance-Enhancement/actions/workflows/run-metrics.yml/badge.svg)

---

## 🔒 Secrets and Environment Variables

Secrets (such as API keys) are not stored in the repository. To set up your environment:

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```
2. Fill in your API keys and other secrets in `.env`.
3. Scripts will read from environment variables automatically.

---

## 📦 Dependency Management

All dependencies are listed in `requirements.txt`. To update or lock dependencies:

```bash
pip freeze > requirements.txt
```

For advanced dependency management, consider using [pip-tools](https://github.com/jazzband/pip-tools) or Poetry.

---

## 🗂️ Tests and Sample Data

- All tests are in the `tests/` directory.
- Sample GPX files for edge cases are in `tests/data/`.

---

## 🏁 Next Steps

- Expand tests for weather fallback and KPI phase jump logic.
- Use the `.env.template` for all new secrets.
- Keep `requirements.txt` up to date and locked.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cultivation.git
```

### 2. Navigate to the Project Directory

```bash
cd cultivation
```

### 3. Set Up Your Environment

- **Create a Virtual Environment**

  ```bash
  python -m venv venv
  ```

- **Activate the Virtual Environment**

  - On macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

  - On Windows:

    ```bash
    venv\Scripts\activate
    ```

- **Install Required Packages**

  ```bash
  pip install -r requirements.txt
  ```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add necessary environment variables as specified in `docs/2_requirements/environment_setup.md`.

### 5. Run Initial Scripts

Execute setup scripts to initialize databases or preprocess data.

```bash
python scripts/setup.py
```

## 🛠️ Usage

Provide examples and instructions on how to use the scripts and notebooks.

### Running Analysis Scripts

```bash
python scripts/running/analyze_pace.py
```
## 📚 Literature Processing Pipeline

We've introduced a set of scripts under `cultivation/scripts/literature/` to fetch, process, and report on arXiv papers:

1. **Fetch a single paper**
   ```bash
   python -m cultivation.scripts.literature.fetch_paper --arxiv_id 2310.04822
   ```
   - Downloads PDF to `literature/pdf/`.
   - Writes metadata JSON to `literature/metadata/`.
   - Generates a markdown note in `literature/notes/`.
   - Runs DocInsight and appends summary/novelty.

2. **Batch-fetch by query**
   ```bash
   python -m cultivation.scripts.literature.fetch_arxiv_batch \
     --queries "cs.AI" "stat.ML" \
     --state-file .fetch_batch_state.json
   ```
   - Tracks last run dates in the specified state file (default: `.fetch_batch_state.json`).
   - Calls `fetch_paper` for each new arXiv ID.

3. **Aggregate reading metrics**
   ```bash
   python -m cultivation.scripts.literature.metrics_literature
   ```
   - Reads all JSON in `literature/metadata/`.
   - Produces `literature/reading_stats.parquet` with columns:
     `iso_week`, `papers_read`, `avg_novelty`.

### Advanced Usage
- **Override DocInsight URL:**
  ```bash
  export DOCINSIGHT_API_URL=http://your-docinsight-server:8000
  ```
- **Override output directories:**
  ```bash
  export LIT_DIR_OVERRIDE=/path/to/custom/literature
  ```

Use these in your CI or scheduled tasks to automate literature review and reporting.

### Launching Jupyter Notebooks

```bash
jupyter notebook notebooks/synergy/synergy_analysis.ipynb
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory. It covers background information, requirements, design architecture, and analysis results.

- **Background**: `docs/1_background/`
- **Requirements**: `docs/2_requirements/`
- **Design**: `docs/3_design/`
- **Analysis**: `docs/4_analysis/`

## 🤝 Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📧 Contact

For questions or support, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).

## 📝 Changelog

All notable changes to this project will be documented in the [CHANGELOG.md](CHANGELOG.md) file.

## 📊 Project Status

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/cultivation)
![GitHub stars](https://img.shields.io/github/stars/yourusername/cultivation?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/cultivation?style=social)
![License](https://img.shields.io/github/license/yourusername/cultivation)
