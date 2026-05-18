
# Project Overview
## Quick Summary

- Problem: Real estate price prediction
- Model: Stacking Ensemble (Ridge, Lasso, Linear Regression, Gradient Boosting, XGBoost, LightGBM, CatBoost)
- Metric: RMSE (log scale)
- Best score: 12,869
- Dataset: Kaggle Housing

## Define the problem and analyze it from a broader perspective

### 1. Define the objective in business terms

The goal of the project is to develop a machine learning model for predicting real estate prices, which will enable more accurate and faster estimation of the value of apartments and houses.

### 2. Describe how your solution will be used
My solution will be used as a real estate price prediction system and can be implemented in several forms, such as:

* a web application
* an API
* an analytical dashboard
* a reporting module
* integration with mobile applications

### 3. Identify existing solutions or workarounds (if any)
#### Traditional valuation methods:

* Manual appraisal by a property valuer
* Comparative method
* Income method

---

#### Existing technological solutions:

* Automated Valuation Models (AVMs)
* Real estate listing portals with price analysis
* AI-based tools in real estate

### 4. In which categories should the problem be defined (unsupervised/supervised, incremental/static, etc.)?

* **Learning:** supervised
* **Prediction type:** regression
* **Learning mode:** initially static, later incremental
* **Type of data:** tabular, optionally temporal and geospatial

### 5. How will the model's performance be measured?

The results are evaluated based on the Root Mean Squared Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sale price. (The use of logarithms means that errors in predicting expensive and inexpensive houses will have the same impact on the score).

### 6. Is the performance measurement linked to the business objective?

The business objective is to develop a model that accurately predicts real estate prices, supporting agencies, investors, and buyers in decision-making.
RMSE is computed on log-transformed prices, which reduces the impact of large price outliers and stabilizes variance.
A lower RMSE indicates better model precision, leading to more reliable valuations and improved business decisions.

### 7. What is the minimum performance required to meet the business objective?

RMSE (Root Mean Squared Error) ≤ 20,000 (preferably below 15,000)

### 8. Are there any comparable problems? Can you leverage existing experience or tools?

Yes, there are many comparable problems that can provide valuable experience and tools for solving the real estate price prediction problem, such as stock price prediction, energy price forecasting, credit risk assessment, and demand prediction for services.

#### Tools:

* **Scikit-learn:** A tool for building classic ML models such as linear regression, XGBoost, and Random Forest.


### 9. How can the problem be solved manually?
Solving the real estate price prediction problem manually would require an approach based on manually collecting and analyzing data, drawing conclusions from experience, and making decisions intuitively.

### 10. Make a list of assumptions established by you (or others)
#### Assumptions regarding the data:

* **Availability of historical data:** It is assumed that there is a sufficient amount of historical data on real estate prices, such as sales transactions, location data, floor area, property type, technical condition, etc.
* **High data quality:** It is assumed that the data does not contain significant gaps, errors, or anomalies. In case of missing values, it is assumed they can be filled using interpolation, medians, or other imputation techniques.
* **Consistency of units:** All data is measured in consistent units (e.g., area in square meters, prices in local currency).

---

#### **Assumptions regarding the model:**

* **Regression model:** It is assumed that the model for predicting real estate prices will be a regression model, as the goal is to predict a numerical value (price).
* **Model simplicity at the beginning:** Initially, simpler models will be used (e.g., linear regression, decision trees) as a baseline. If they prove insufficient, more advanced methods such as XGBoost or neural networks will be applied later.
* **Model stability:** The model is expected to provide sufficient stability and accuracy on new, unseen data, which requires applying proper validation techniques.

---

#### **Assumptions regarding model usage:**

* **Availability of current input data:** The model will operate in real time, assuming that system users (e.g., real estate agents) will provide up-to-date input data (e.g., new property listings).
* **User education:** System users (e.g., investors, real estate agents) will be properly trained in interpreting the model’s results and applying them in decision-making.

---

#### **Assumptions regarding model performance:**

* **Expected accuracy level:** It is assumed that the model will achieve accuracy acceptable in the business context, i.e., RMSE ≤ 20,000.
* **Expected speed of operation:** The model should be fast enough to be used in real time (e.g., for quick property valuation in response to a client’s inquiry).
* **Regular model updates:** The model will require regular retraining on new data to adapt to the changing real estate market.

# Data

## Data acquisition

### 1. Specify the type and amount of data needed
The provided data is in CSV format and contains 1,460 rows and 81 columns.

### 2. Identify the source from which you can obtain the data and document it
https://www.kaggle.com/competitions/home-data-for-ml-course/data

### 3. Check how much storage space will be needed to store the data
The data requires less than 1 MB of disk space.


# Installation and Setup

In this section, detailed instructions are provided on how to set up the project on a local machine. Follow the steps below to ensure a smooth and reproducible environment.

## Codes and Resources Used

This section provides essential information about the software requirements used in this project.

- **Editor Used:** PyCharm  
- **Python Version:** Python 3.12  

It is recommended to use the same versions to avoid compatibility issues.

---

## Python Packages Used

Below is the list of dependencies required to run the project. It is recommended to install them inside a virtual environment.

### General Purpose
- joblib
- PyYAML 
- GreatExpectations
- FastAPI
- gradio

### Data Manipulation
- numpy 
- pandas 

### Data Visualization
- matplotlib
- seaborn

### Machine Learning
- scikit_learn 
- catboost 
- lightgbm 
- xgboost 
- mlflow
- optuna

---

## Installation Steps

### 1. Clone the repository
```bash
cd <your-project-folder>
git clone https://github.com/Airdj/housing_price_prediction.git
```


### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermode -aG docker $USER
```

### 5. Run the container
```bash
docker compose up --build
```
### 6. Check FastAPI and Gradio
```bash
fastapi: 0.0.0.0/8000
gradio: 0.0.0.0/7860
```
# Code structure

The project follows a modular structure:
- `configs/` – configuration files
- `data/` – raw and processed datasets
- `features/` – feature engineering
- `notebooks/` – EDA
- `app/` – gradio + FastAPI
- `models/` – model definitions, training, tuning, evaluating

```bash
├── configs
│   ├── base_models_params.yaml
│   └── tuned_models_params.yaml
├── data
│   ├── processed
│   │   ├── processed_final_eval_df.csv
│   │   ├── processed_final_inference_df.csv
│   │   ├── processed_final_test_df.csv
│   │   ├── processed_final_train_df.csv
│   │   ├── processed_target_eval_feature.csv
│   │   └── processed_target_train_feature.csv
│   └── raw
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── notebooks
│   └── EDA_housing_price_prediction.ipynb
├── src
│   ├── app
│   │   ├── app.py
│   │   └── main.py
│   ├── data
│   │   ├── load_data.py
│   │   └── preprocess_data.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   ├── evaluate.py
│   │   ├── model_factory.py
│   │   ├── stack.py
│   │   ├── train.py
│   │   └── tune.py
│   ├── pipelines
│   │   ├── inference_pipeline.py
│   │   └── train_pipeline.py
│   ├── serving
│   │   ├── best_model_stacked.pkl
│   │   └── inference.py
│   ├── utils
│   │   ├── build_features_pipeline_artifacts.pkl
│   │   ├── columns.csv
│   │   ├── fastapi_test_record_json.json
│   │   ├── helpers.py
│   │   ├── preprocess_pipeline_artifacts.pkl
│   │   └── validate_data.py
│   └── __init__.py
├── tests
│   ├── test_fastapi.py
│   └── test_inference.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── docker_requirements.txt
├── mlflow.db
├── README.md
└── requirements.txt
```

## Notebook

The notebook contains:
- EDA
- Feature analysis
- Model comparison
# Modeling

### Models tested
- Linear_Regression
- KNN Regressor
- Decision Tree
- Random Forest 
- SGD Regression 
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- LinearSVR
- GradientBoosting Regression
- XGBoost Regression
- LightGBM Regression

### Final model
Stacked models(Ridge, Lasso, LinReg, GBR, XGB, LGBM, CatBoost) with hyperparameter tuning

### Feature engineering
- Drop columns (Low correlation with the target variable, High category dominance,
Lack of statistical significance, High proportion of missing values)
- Handling missing values
- Combine new features
- Encoding categorical variables
- Handling outliers
- Feature scaling
- PCA

# Results and evaluation
Scores are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

Actual best score (RMSE): 12869.95219

### Validation
- Train/test split
- Cross-validation (e.g., K-Fold)
- Preventing data leakage

# Future work
- Monitoring model performance over time
- Make few fixes with data validation
- Upgrade UI