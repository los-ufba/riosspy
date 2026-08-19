# RiossPy

**RiossPy** is a Python framework for processing and analyzing Sentinel-1 Synthetic Aperture Radar (SAR) imagery, with a focus on oil-spill detection and remote-sensing applications.

The project provides tools for working with Sentinel-1 data, preparing datasets, training deep-learning models, and performing inference.

## Features

* Sentinel-1 SAR data acquisition
* Dataset preparation and preprocessing
* Deep-learning models for SAR image analysis
* Model training callbacks
* Inference utilities

## Project Structure

```text
riosspy/
├── riosspy/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── callbacks.py
│   ├── download_sentinel.py
│   └── __init__.py
│
├── graphs/
├── tests/
├── setup.py
├── requiriments.txt
└── README.md
```

### `riosspy.data`

Utilities for loading, preparing, and processing the datasets used by the project.

### `riosspy.models`

Deep-learning model implementations used for SAR image analysis.

### `riosspy.inference`

Tools for running trained models on new data and generating predictions.

### `riosspy.download_sentinel.py`

Utilities for downloading Sentinel-1 data required for the processing pipeline.

### `riosspy.callbacks.py`

Training callbacks used during model development and experimentation.

## Installation

Clone the repository:

```bash
git clone https://github.com/los-ufba/riosspy.git
cd riosspy
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requiriments.txt
```

Install RiossPy:

```bash
pip install -e .
```

## Usage

After installation, the package can be imported directly:

```python
import riosspy
```

The individual modules can then be used according to the desired workflow:

```python
from riosspy import data
from riosspy import models
from riosspy import inference
```

## Typical Workflow

A typical RiossPy workflow consists of:

```text
Sentinel-1 Data
      │
      ▼
Data Acquisition
      │
      ▼
Preprocessing
      │
      ▼
Dataset Preparation
      │
      ▼
Model Training
      │
      ▼
Trained Model
      │
      ▼
Inference
      │
      ▼
SAR Image Analysis
```

## Research Context

RiossPy was developed as part of research at the **Laboratório de Óleo e Gás (LOS) at Universidade Federal da Bahia (UFBA)**.

The project is intended to facilitate experimentation with remote sensing and machine learning applied to Sentinel-1 SAR imagery.

## Tests

Tests are located in the `tests/` directory.


## License

This project is distributed under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) for the complete license text.

## Repository

https://github.com/los-ufba/riosspy
