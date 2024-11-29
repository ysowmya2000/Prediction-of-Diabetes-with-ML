# Prediction-of-Diabetes-with-ML
## Overview
This project focuses on predicting the likelihood of diabetes in patients using multiple machine learning models. The dataset used contains various medical predictors, such as glucose levels, BMI, and age, as well as a binary outcome variable indicating the presence of diabetes. The project applies data preprocessing, feature engineering, model training, evaluation, and visualization to achieve optimal predictions.

## Features
### 1. Data Preprocessing:
- Handling missing values using median imputation based on the Outcome variable.
- Scaling numerical features using RobustScaler.
- Encoding categorical features using one-hot encoding.
- Removing outliers using Local Outlier Factor (LOF).

### 2. Exploratory Data Analysis (EDA):
- Distribution analysis of all features.
- Correlation matrix and heatmap visualization.
- Age and outcome distributions, density plots, and pair plots for better insights.

### 3. Feature Engineering:
- Categorizing BMI, Insulin, and Glucose into meaningful ranges.
- Creating new categorical features such as NewBMI, NewInsulinScore, and NewGlucose.

### 4. Model Training and Evaluation:
- Multiple machine learning models:
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Decision Tree Classifier (DT)
- Random Forest Classifier (RF)
- Gradient Boosting Classifier (GBDT)
- XGBoost Classifier

- Hyperparameter tuning using GridSearchCV.
- Metrics for evaluation:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Curve

### 5. Performance Comparison:
- Comparative analysis of accuracy and ROC-AUC for all models.
- Visualizations of performance metrics using bar plots and ROC curves.

### 6. Model Deployment:
- Best-performing model (Gradient Boosting Classifier) saved using pickle for future use.

## Results
- Best Model: Gradient Boosting Classifier
### Model Performance:
- Accuracy: 85.4%
- ROC-AUC: 89.7%
### Visualization:
- ROC-AUC curves for all models.
- Bar chart comparing accuracy and ROC-AUC of all models.

### Future Work
- Explore advanced deep learning models for better accuracy.
- Deploy the trained model as a web API using Flask or FastAPI.
- Integrate the solution with healthcare systems for real-time predictions.
## Libraries used
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost
