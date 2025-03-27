# Predicting the success of Bank telemarketing

## Project Overview
This project employs a structured approach to data analysis, integrating both univariate and multivariate analysis techniques to gain insights and build predictive models. The workflow includes data preprocessing, feature engineering, model selection, hyperparameter tuning, and threshold adjustments.

## Steps in the Project

### 1. Data Exploration and Preprocessing
- **Univariate Analysis:** Analyzed each feature independently to identify distribution patterns, skewness, normality, and outliers.
- **Visualization & Summary Statistics:** Used histograms for continuous variables and bar plots for categorical variables. Identified right-skewed features and missing values.
- **Multivariate Analysis:** Explored relationships between features using correlation matrices, box plots, and count plots.

#### Preprocessing Steps:
- **Dropping Unnecessary Columns:** Removed irrelevant columns such as `last contact date` and `age_category`.
- **Handling Missing Data:** Used imputation techniques to fill missing values and maintain data integrity.
- **Feature Encoding:** Converted categorical variables into numerical formats using One-Hot Encoding and Label Encoding.
- **Feature Selection:** Applied `SelectKBest` and Recursive Feature Elimination (RFE) to identify important features.
- **Dimensionality Reduction:** Used PCA and SVD to reduce dimensionality while retaining key variance.

### 2. Model Selection and Training
Experimented with multiple models:
- **SGD (Stochastic Gradient Descent)**
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **LightGBM (LGBM)**

#### Model Performance Comparison:
- Evaluated models using accuracy and other key metrics.
- Implemented **ensemble learning techniques**:
  - **Stacking Classifiers:** Combined Logistic Regression, XGBoost, and LightGBM, achieving a baseline score of **0.70**.
  - **Voting Classifier:** Improved performance to **0.76**.

### 3. Hyperparameter Tuning
- Fine-tuned models using grid search and randomized search.
- Improved LightGBM’s accuracy from **0.72** to **0.75**.

### 4. Threshold Adjustment
- Adjusted probability thresholds using `predict_proba`.
- Achieved a final best score of **0.77** with LightGBM.

## Challenges and Solutions

### 1. Handling Missing Data
- **Challenge:** Missing values could bias results.
- **Solution:** Used imputation techniques instead of dropping data.

### 2. Feature Engineering and Dimensionality
- **Challenge:** Highly skewed and irrelevant features complicated model training.
- **Solution:** Applied PCA and SVD to reduce complexity and improve efficiency.

### 3. Choosing the Right Model
- **Challenge:** Multiple models produced varying results.
- **Solution:** Experimented with different algorithms, focusing on XGBoost and LightGBM for best performance.

### 4. Overfitting and Model Evaluation
- **Challenge:** Overfitting risk with complex models.
- **Solution:** Used cross-validation and feature selection to generalize models better.

### 5. Model Calibration and Threshold Tuning
- **Challenge:** Imbalanced class predictions required fine-tuning.
- **Solution:** Adjusted probability thresholds using `predict_proba` to improve precision.

## Final Results
- Achieved a final accuracy score of **0.77** using LightGBM with threshold tuning.
- Improved performance over baseline models by systematically refining preprocessing, model selection, and hyperparameter tuning.
## Technologies Used
- **Python** (pandas, numpy, scikit-learn, XGBoost, LightGBM)
- **Data Visualization** (matplotlib, seaborn)
- **Feature Engineering** (PCA, SVD, One-Hot Encoding, RFE)
- **Machine Learning Models** (SGD, Random Forest, XGBoost, KNN, LightGBM)
- **Hyperparameter Tuning** (Grid Search, Random Search)

