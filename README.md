# Decision Tree Classification & Overfitting Analysis

A small, **coursework-inspired** machine learning project exploring how decision-tree depth affects training and testing accuracy. The analysis uses the **public breast cancer dataset included with scikit-learn**, not private course files or instructor-provided datasets.

## Background

In HMB301 Biotechnology coursework at the University of Toronto, I trained decision tree classifiers and explored overfitting through changes to `max_depth`, `min_samples_leaf`, and `min_impurity_decrease`. This repository independently reproduces the *depth comparison* with an accessible public dataset and a self-contained Python script.

## What the code does

1. Loads scikit-learn's breast cancer classification dataset.
2. Splits observations into training (80%) and test (20%) sets with `random_state=42`.
3. Fits decision trees with maximum depths 1 through 6.
4. Compares training and test accuracy for each depth.
5. Saves a CSV and a plot showing accuracy against maximum depth.

## Run

```bash
python -m pip install -r requirements.txt
python decision_tree_analysis.py
```

## Interpretation

As tree depth increases, training accuracy can improve while test accuracy stops improving or declines. That gap is a warning sign of overfitting: the tree may be learning idiosyncrasies of the training data rather than generalizable patterns. The best depth should be selected with appropriate validation rather than by training accuracy alone.

This is an educational classification demonstration, **not a clinical prediction tool**. It is not intended for medical decision-making.

## Scope and limitations

- This repository is based on concepts practiced in coursework, but does not redistribute the original assignment, teaching materials, or course datasets.
- The public dataset and fixed train/test split make the exercise easy to reproduce, but a single split is not sufficient for robust model selection. Cross-validation would be a useful next step.
- The original HMB301 exercises also examined a heart-failure dataset and gene-expression counts; those data and exercises are not included here.
