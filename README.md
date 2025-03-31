# RunningFastAndSlow 🐀

## Overview
This repository contains code to process and generate the figures for the preprint [Running, Fast and Slow: The Dorsal Striatum Sets the Cost of Movement During Foraging](https://www.biorxiv.org/content/10.1101/2024.05.31.596850v1). The code is provided as is—it should function correctly and produce the expected results, but it could benefit from refactoring for improved readability, maintainability, and performance.

## Contents
- `.ipynb` notebooks for each main and supplementary figure.
- Two `.ipynb` notebooks for data analysis:
  - `process_data.ipynb`: Processes raw data.
  - `model_fitting.ipynb`: Fits models to the data.
- `.py` files containing reusable functions for processing, plotting, statistics, and utilities.
- `Figures/` folder with illustrations and a style sheet.
- `Figures_paper/` folder containing the final main and supplementary figures in `.png` and `.svg` formats.
- `picklejar/` folder with preprocessed results.

## Compatibility
The code was developed using **Python 3.7.11**, which is now **end-of-life (EOL)**. While it runs with more recent Python versions, it has not been extensively tested and may produce deprecation warnings or errors.

For best compatibility, you may use **Python 3.7.9** from the official releases:
- [Python 3.7.9 installer](https://www.python.org/downloads/release/python-379/)
- [Build Python 3.7.11 from source](https://www.python.org/downloads/release/python-3711/)

## Setup
To set up your environment and install dependencies:

```sh
# Create a virtual environment with Python 3.7.9
python3.7 -m venv env
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

Ensure that all required packages are installed before running the notebooks.

