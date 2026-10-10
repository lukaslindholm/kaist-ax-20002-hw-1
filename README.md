# KAIST AX20002 HW1: Regression Beyond the Gaussian Assumption
**Name:** Lukas Lindholm
**Student ID:** 20266508

## Directory Structure & Setup
For the scripts to run correctly in a clean folder, please ensure that the original competition data files (`train_A.csv`, `test_A.csv`, etc.) are placed inside a subdirectory named `data/`.

## Dependencies
The environment requires the following packages to be installed:
`pip install pandas numpy lightgbm xgboost scikit-learn matplotlib seaborn scipy`

## Execution Instructions

### 1. Final Kaggle Submission
To train the final models on 100% of the training data and generate the submission file, run:
`python submission_final.py`
*Output: Generates `submission_final.csv` in the root directory.*

### 2. Local Validation (D2 Scores)
To reproduce the local 80/20 out-of-fold validation scores discussed in the report, run:
`python local_validation.py`
*Output: Prints Pinball Loss and D2 scores for all tasks to the console.*

### 3. Diagnostic Plots
To reproduce the visual diagnostics used to motivate the model architectures in the report, run the following scripts:
`python diagnostic_plots_all.py`
`python plot_residuals_task_A.py`
*Output: Creates a `plots/` directory and saves all figures as PDFs.*

## Reproducibility
All random seeds for LightGBM, XGBoost, Scikit-Learn (K-Means, Logistic Regression, Train-Test Splits), and Optuna trials were strictly fixed to `random_state=42` to ensure fully deterministic and reproducible results.

## AI Usage Declaration
As per the course guidelines, parts of the code and report structuring were generated with AI assistance (Google Gemini). Specific declarations are included as comments at the top of the Python files and as a footnote in the methodology section of the report. The analytical observations and model architectures were independently directed and verified by the author.