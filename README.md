# 🏦 Home Credit Default Risk Solution

A machine learning solution for predicting the probability of loan default in the Kaggle Home Credit Default Risk competition.

## 📝 Description

This repository contains a solution for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition on Kaggle. The goal of this competition is to predict how capable each applicant is of repaying a loan. The solution uses LightGBM as the main algorithm and employs model blending to improve prediction accuracy.

## ✨ Features

- **Feature Engineering**: Creates numerous features from multiple data sources to improve model performance
- **LightGBM Model**: Implements a gradient boosting model with optimized hyperparameters
- **K-Fold Cross-Validation**: Uses stratified k-fold cross-validation to ensure robust model evaluation
- **Model Blending**: Combines predictions from multiple models to create a stronger ensemble
- **Feature Importance Analysis**: Visualizes and exports feature importance for model interpretability

## 🛠️ Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - LightGBM
  - scikit-learn
  - matplotlib
  - seaborn

## 📊 Data

The solution expects the following data files (not included in this repository):
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

These files should be placed in the root directory of the project.

## 🚀 Usage

### Main Model Training

To train the main LightGBM model and generate predictions:

```python
python creditDefault.py
```

This will:
1. Process all data files
2. Create features
3. Train a LightGBM model with 5-fold cross-validation
4. Generate a submission file named `HomeCreditDefaultSubmit.csv`
5. Output feature importance visualization

### Model Blending

To blend multiple model predictions:

```python
python blender.py
```

This will:
1. Load prediction files from the `blended/` directory
2. Create a weighted average of predictions
3. Generate a final submission file named `blended.csv`

## 🔍 Model Details

The solution uses a LightGBM classifier with the following key parameters:
- Learning rate: 0.02
- Number of leaves: 34
- Max depth: 8
- Feature subsampling and row subsampling for regularization
- Early stopping to prevent overfitting

Feature engineering includes:
- Ratio features (e.g., credit to income ratio)
- Statistical aggregations (mean, max, min, etc.)
- Temporal features based on days and dates
- One-hot encoding for categorical variables

## 📈 Performance

The model's performance is evaluated using the ROC AUC metric, which is printed during training for each fold and for the overall validation set.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
