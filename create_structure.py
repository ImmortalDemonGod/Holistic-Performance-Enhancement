import os
import argparse
from pathlib import Path

def create_directory(path: Path):
    """
    Create a directory if it doesn't exist.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")

def create_file(path: Path, content: str = ""):
    """
    Create a file with the given content if it doesn't already exist.
    """
    try:
        if not path.exists():
            path.write_text(content, encoding='utf-8')
            print(f"Created file: {path}")
        else:
            print(f"File already exists: {path}")
    except Exception as e:
        print(f"Failed to create file {path}: {e}")

def rename_file(old_path: Path, new_path: Path):
    """
    Rename a file if it exists. If new_path exists, we won't overwrite it.
    """
    if old_path.exists() and not new_path.exists():
        try:
            old_path.rename(new_path)
            print(f"Renamed {old_path.name} to {new_path.name}")
        except Exception as e:
            print(f"Failed to rename {old_path.name} to {new_path.name}: {e}")
    else:
        # If old_path doesn't exist or new_path already exists, do nothing
        if not old_path.exists():
            print(f"File not found for rename: {old_path}")
        if new_path.exists():
            print(f"New file name already exists, not overwriting: {new_path}")

def setup_repo_structure(base_dir='cultivation'):
    """
    Set up only the directory structure and placeholder files.
    No .gitignore, no README, no Git init.
    """
    base_path = Path(base_dir)
    create_directory(base_path)

    # Define the directory structure you want to enforce
    structure = {
        'docs': ['1_background', '2_requirements', '3_design', '4_analysis'],
        'data': ['running', 'biology', 'software', 'synergy'],
        'scripts': ['running', 'biology', 'software', 'synergy'],
        'notebooks': ['running', 'biology', 'software', 'synergy'],
        'ci_cd': [],
    }

    # Create subdirectories
    for folder, subfolders in structure.items():
        folder_path = base_path / folder
        create_directory(folder_path)
        for sub in subfolders:
            sub_path = folder_path / sub
            create_directory(sub_path)

    #
    # docs/1_background placeholders
    #
    bkgd_path = base_path / 'docs' / '1_background'
    # Ensure directory exists
    create_directory(bkgd_path)

    # We will rename the existing 1.md through 6.md to more descriptive filenames.
    # If the old files don’t exist yet, we’ll just create the new ones.
    rename_map = {
        "1.md": "potential_overview.md",
        "2.md": "domains_scope.md",
        "3.md": "synergy_concept.md",
        "4.md": "critique_and_refinement.md",
        "5.md": "ultimate_goals.md",
        "6.md": "final_thoughts.md"
    }

    for old_name, new_name in rename_map.items():
        old_file = bkgd_path / old_name
        new_file = bkgd_path / new_name
        # Attempt to rename if the old file exists; otherwise, create the new file if needed.
        if old_file.exists():
            rename_file(old_file, new_file)
        else:
            # If the new file doesn't exist, create it
            create_file(new_file, f"# {new_name.replace('_',' ').title()}\n")

    #
    # docs/2_requirements folder
    #
    req_path = base_path / 'docs' / '2_requirements'
    create_file(req_path / "requirements.md", "# Project Requirements\n")

    #
    # docs/3_design folder
    #
    design_path = base_path / 'docs' / '3_design'
    create_file(design_path / "design_overview.md", "# Design Overview\n")

    #
    # docs/4_analysis folder
    #
    analysis_path = base_path / 'docs' / '4_analysis'
    create_file(analysis_path / "analysis_overview.md", "# Analysis Overview\n")

    #
    # scripts/ placeholders
    #
    scripts_base = base_path / 'scripts'
    script_placeholders = {
        'running': [
            "process_run_data.py",
        ],
        'biology': [
            "analyze_literature.py",
        ],
        'software': [
            "commit_metrics.py",
        ],
        'synergy': [
            "calculate_synergy.py",
        ]
    }
    for subfolder, files in script_placeholders.items():
        sf_path = scripts_base / subfolder
        for f_name in files:
            create_file(sf_path / f_name, f"# Placeholder: {f_name}\n")

    #
    # notebooks/ placeholders
    #
    notebooks_base = base_path / 'notebooks'
    nb_placeholders = {
        'running': [
            "running_eda.ipynb",
        ],
        'biology': [
            "biology_eda.ipynb",
        ],
        'software': [
            "software_eda.ipynb",
        ],
        'synergy': [
            "synergy_experiment.ipynb",
        ]
    }
    for subfolder, files in nb_placeholders.items():
        sf_path = notebooks_base / subfolder
        for f_name in files:
            create_file(sf_path / f_name, """{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
""")

    #
    # ci_cd/ placeholder
    #
    cicd_path = base_path / 'ci_cd'
    create_file(cicd_path / "placeholder.md", "# CI/CD Configurations Go Here\n")

    print("\nRepository structure setup complete (renaming existing background files).")

def main():
    parser = argparse.ArgumentParser(description="Set up the Holistic Performance Enhancement repository structure (placeholders only).")
    parser.add_argument('--dir', type=str, default='cultivation', help="Name of the base directory for the repository.")
    args = parser.parse_args()

    setup_repo_structure(base_dir=args.dir)

if __name__ == '__main__':
    main()
