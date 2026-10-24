# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:34:36 2026

@author: hussa
"""


from pathlib import Path
import sys
import platform
import subprocess

PROJECT_DIR = Path(
    r"D:\Data Science Assignment\My Projects\DS-A3"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DS-A3 ENVIRONMENT AUDIT")
print("=" * 65)

print("\nPython executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

print("\nPlatform:")
print(platform.platform())

# Record the packages used by this project.
packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "statsmodels",
]

output_lines = []

for package in packages:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "show",
            package,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    name = None
    version = None

    for line in result.stdout.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()

    if name and version:
        output_lines.append(f"{name}=={version}")
        print(f"{name}=={version}")
    else:
        print(f"NOT FOUND OR NOT VISIBLE: {package}")

lock_file = RESULTS_DIR / "requirements-lock.txt"
lock_file.write_text(
    "\n".join(output_lines) + "\n",
    encoding="utf-8",
)

environment_file = RESULTS_DIR / "environment-details.txt"

environment_file.write_text(
    f"Python executable: {sys.executable}\n"
    f"Python version: {sys.version}\n"
    f"Platform: {platform.platform()}\n",
    encoding="utf-8",
)

print("\nSaved:")
print(lock_file)
print(environment_file)

print(
    "\nImportant: this records the current environment. "
    "It proves the original execution environment only if "
    "this is the same environment used for the analyses."
)
