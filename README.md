# NSGA-II-TS-QL-GLR

Supporting data, figures, and Python scripts for *A Q-learning Hybrid NSGA-II for Trade-offs Between Makespan, Energy, and First-Pass Yield in High-Mix Low-Volume Flexible Job Shops*.

## Contents

| Path | Description |
| --- | --- |
| `MK02.xlsx`, `MK06.xlsx`, `MK09.xlsx` | Benchmark workbooks with processing time, energy, and first-pass-yield inputs. |
| `Pictures/` | Figures used in the manuscript. |
| `Scripts/` | Experiment, evaluation, and plotting scripts, including the model and sensitivity algorithm. |
| `evaluation_results/` | Final metric summaries, paired tests, and empirical reference fronts. |
| `sensitivity_results/` | MK06 sensitivity runs and their protocol. |
| `supplementary_figures/` | Additional plots and supporting tables. |

## Getting started

Use Python 3.13. From the repository root, install the listed dependencies:

```bash
python -m pip install -r Scripts/requirements-sensitivity.txt
```

The experiment entry points are `Scripts/run_sensitivity_experiments.py` and `Scripts/run_main_experiments.py`. Both provide `--help` and a short `--smoke` mode. The saved evaluation tables and manuscript figures can be viewed directly in the folders above.

## Reproducibility note

The `protocol.json` files document the original run settings, seeds, and source checksums. They retain historical Chinese filenames and paths from the research workspace. This repository includes the MK06 sensitivity runs and evaluation summaries, but not every raw main-experiment run; it is therefore not a complete archive for independently rebuilding every reported result.
