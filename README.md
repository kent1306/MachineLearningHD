# MachineLearningHD

SIT307 Task 11.1HD — Machine Learning Research Project

## Project Overview

This project reproduces and critically evaluates the machine learning approach presented in:

**M. Bhagat, A. Sharma, and P. Agarwal, “An Efficient Stacking-Based Ensemble Technique for Early Heart Attack Prediction,” _Multimedia Tools and Applications_, 2024.**

The project contains two main parts:

1. **Part 1 — Reproduction Study:** Reproduce the six classifiers and stacking ensemble used in the selected paper and compare the reproduced results with the published results.
2. **Part 2 — Proposed ML Solution:** Investigate a limitation identified during reproduction and develop a more robust evaluation pipeline.

## Dataset

The project uses the public Heart Disease dataset used by the selected paper.

- **Records:** 1,025
- **Predictor features:** 13
- **Target:** Binary heart disease classification
  - `0` = No heart disease
  - `1` = Heart disease

Dataset file:

```text
heart.csv
```

## Part 1 — Reproduction Study

The following classifiers were reproduced:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gaussian Naive Bayes
- K-Nearest Neighbours
- Stacking Ensemble

The main evaluation metrics are Accuracy, Precision, Recall, F1 Score, and AUC.

### Main Reproduction Result

| Metric | Published Result | Reproduced Result |
|---|---:|---:|
| Accuracy | 0.9853 | 0.9854 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9727 | 0.9709 |
| F1 Score | 0.9861 | 0.9852 |
| AUC | 0.9880 | 1.0000 |

The reproduced stacking accuracy closely matches the published result.

## Critical Analysis

Further analysis identified substantial duplication in the original dataset:

- **723 duplicated rows** were identified.
- Under the reproduced 80:20 train-test split, **199 of 205 test samples** also had an identical observation in the training data.
- This corresponds to approximately **97.07% exact train-test overlap**.

A duplicate-free diagnostic experiment reduced the dataset from 1,025 records to 302 unique observations.

The stacking accuracy decreased from approximately **98.54%** in the main reproduction to **83.61%** in the duplicate-free diagnostic experiment.

This provides strong evidence that duplicate observations influenced the very high performance measured under the original-style random split.

## Part 2 — Robust Duplicate-Aware Stacking Pipeline

To address the limitation identified in Part 1, this project proposes a **Robust Duplicate-Aware Stacking Pipeline**.

The main methodological changes are:

1. Remove exact duplicate observations before model evaluation.
2. Use **Repeated Stratified 5-Fold Cross-Validation** instead of one random train-test split.
3. Repeat the 5-fold process **5 times**, producing **25 outer-fold evaluations**.
4. Place `StandardScaler` inside each model pipeline so preprocessing is fitted only using the corresponding training data.
5. Retain the same six base algorithms and Logistic Regression meta-classifier to isolate the effect of the redesigned evaluation protocol.

### Proposed Method Results

| Metric | Mean ± SD |
|---|---:|
| Accuracy | 0.8284 ± 0.0373 |
| Precision | 0.8240 ± 0.0519 |
| Recall | 0.8768 ± 0.0622 |
| F1 Score | 0.8472 ± 0.0336 |
| AUC | 0.9050 ± 0.0295 |

The proposed method does not aim to increase raw accuracy above the published result. Instead, it provides a more robust estimate of model generalisation after exact duplicate contamination is removed.

## Repository Structure

```text
MachineLearningHD/
├── HD Task paper option 3.pdf   # Selected research paper
├── Task 11.1HD.ipynb            # Complete Part 1 and Part 2 implementation
├── heart.csv                    # Heart Disease dataset
├── report_final.pdf             # Final technical report
└── README.md                    # Project instructions and summary
```

## Requirements

Main Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
jupyter
```

Recommended Python version:

```text
Python 3.10+
```

## How to Run

### 1. Clone or download the repository

```bash
git clone <your-repository-url>
cd MachineLearningHD
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Task 11.1HD.ipynb
```

### 5. Run the notebook

Run all cells from top to bottom.

Keep `heart.csv` in the same directory as the notebook so that:

```python
pd.read_csv("heart.csv")
```

can load the dataset correctly.

The notebook includes:

- dataset inspection and preprocessing
- Part 1 individual classifier reproduction
- stacking ensemble reproduction
- confusion matrices and ROC analysis
- published vs reproduced comparisons
- duplicate and train-test overlap analysis
- duplicate-free diagnostic experiment
- Part 2 duplicate-aware stacking pipeline
- Repeated Stratified K-Fold evaluation
- mean and standard deviation results
- final comparative analysis

## Reproducibility Notes

Some implementation details were not fully specified in the selected paper. The reasonable assumptions used in this reproduction are documented in the notebook and technical report, including:

- 80:20 train-test split
- `random_state=42`
- `StandardScaler`
- Logistic Regression as the stacking meta-classifier
- library defaults where exact hyperparameters were not reported

For Part 2, the outer validation uses:

```python
RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)
```

## Report

The complete methodology, experimental results, critical analysis, limitations, and discussion are available in:

```text
report_final.pdf
```

## Academic Integrity and GenAI Acknowledgement

ChatGPT was used to support brainstorming, experiment planning, identifying suitable Python libraries and functions, and assisting with the creation of graphs, tables, and charts. It was also used to review explanations and improve writing clarity.

All suggestions were critically reviewed and adapted before inclusion. The implementation, results, methodological decisions, and analysis were reviewed and understood by the author.

## Author

**Huynh Quang Vinh (Kent)**  
SIT307 — Task 11.1HD
