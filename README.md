# RunningFastAndSlow

## Overview
This repository contains cleaned-up code to process and generate the figures for the preprint [Running, Fast and Slow: The Dorsal Striatum Sets the Cost of Movement During Foraging](https://www.biorxiv.org/content/10.1101/2024.05.31.596850v1).

In this repository, you'll find:
- One `.ipynb` notebook for each main figure and each supplementary figure.
- `.py` files containing reusable functions used throughout the analysis.

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

