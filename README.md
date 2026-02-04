# Fantasy Premier League Team Optimization

---

This repository contains the codebase developed as part of a diploma thesis on **player performance prediction and decision support for Fantasy Premier League (FPL)**.

The project implements an end-to-end pipeline that:
- collects historical and current-season FPL data,
- performs feature engineering on player- and team-level statistics,
- trains machine learning models to predict player points, and
- applies optimization algorithms to support squad selection and transfer decisions  
  (Wildcard, Free Hit, and transfer planning).

The emphasis of this repository is on the implementation of the **full system pipeline**. 
Methodological details, experimental evaluation, and analytical discussion are documented in the associated diploma thesis and are not reproduced here.

---

## Project Structure

The repository is organized as a single Python package:

```

fpl/
├── data/            # Data fetching and preprocessing logic
├── features/        # Feature engineering pipelines
├── models/          # Machine learning models (RF, GBM, LightGBM)
├── optimization/    # Team selection and transfer optimization
├── utils/           # Shared utilities
└── main.py          # Pipeline entry point

````

All source code resides under the `fpl/` package.  


## Data Sources

The project uses data from the following sources:

- **Official Fantasy Premier League (FPL) API**, which provides live and historical information on players, teams, fixtures, and gameweeks.
- **Vaastav’s Fantasy Premier League Dataset** https://github.com/vaastav/Fantasy-Premier-League/

  This repository is used as a historical data source for past seasons and is gratefully acknowledged.


## Data and Outputs (Runtime Structure)

During execution, the project creates additional directories automatically.
These directories do not need to exist prior to running the pipeline.

Typical runtime generated folders include:

- `data/`
  - season-specific subfolders (e.g. `2023-2024/`, `2024-2025/`)
  - raw and processed datasets fetched from the FPL API
- `models/`
  - trained machine learning models
- `predictions/`
  - generated player total point predictions per gameweek
- `results/`
  - model evaluation results (file with error metrics across runs, optional)

All such folders are excluded via `.gitignore`.


## Important Note: Expected Statistics in Older Seasons

Seasons **"21-22" and earlier** do **not** include `expected_*` statistics.

If such seasons are included in the training data, all features derived from `expected_*` columns must be excluded prior to model training.
Alternatively, older seasons can be removed from the `past_seasons` list in `config.yaml`.

This constraint is discussed in more detail in the exploratory analysis provided in `fpl.ipynb`.


## Configuration and Model Selection

The project behavior is controlled via `config.yaml`.

The current configuration reflects the best-performing setup identified during the experimental phase of the thesis.

### Default configuration (recommended)
- **Model**: Random Forest
- **Purpose**: Best overall predictive performance
- **Usage**: Recommended for final evaluations and optimization runs

### Alternative configuration (faster)

A faster Gradient Boosting configuration is also recommended and was evaluated during experimentation.

This option:
- trains faster,
- is suitable for quick iteration or limited computational resources,
- performs slightly worse than the default Random Forest setup, but it also has very good predictive perfomance.

To use this alternative configuration:
1. Select the Gradient Boosting profile (`model_type: "gbm", rolling_window: 8, difficulty_window: 4, team_window: 4, form_window: 3`) in `config.yaml`.
2. Exclude `code` and `season` from the categorical feature list.


## Optimization Logic

Optimization is formulated as a constrained selection problem under standard FPL rules (budget, squad size, positional constraints, and team limits).

The optimization behavior depends on both the current gameweek and the configuration:

- When the pipeline is executed at the **start of a season (Gameweek 1)**, it automatically returns the optimal full squad.
- For subsequent gameweeks, if neither the Wildcard nor Free Hit option is enabled in `config.yaml`, the pipeline runs a **transfer optimization engine**.
  In this case, the objective is to identify the best set of transfers relative to the existing squad, rather than selecting an entirely new team.
  

Wildcard and Free Hit behavior can be explicitly controlled via the configuration file.


## Setup

The project was developed using **Python 3.12**.

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate fpl
````

### Using pip

```bash
pip install -r requirements.txt
```


## Running the Project

From the project root:

```bash
python -m fpl.main
```

All runtime options (season selection, model choice, optimization strategy)
are defined in `config.yaml`.


## Notebook Usage

An example notebook is provided for exploratory analysis and experimentation:

```
fpl.ipynb
```

When running notebooks, ensure that the project root is on the Python path.


## Notes

* This repository is intended primarily as a research and thesis codebase.
* Large datasets, trained models and outputs are intentionally excluded.

---

## Author

Dimitra Lekatsa  

This project was developed as part of a diploma thesis at the National Technical University of Athens (NTUA), School of Electrical and Computer Engineering.

---
