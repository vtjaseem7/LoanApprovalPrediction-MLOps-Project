# Loan Approval Prediction MLOps Project

## Project Overview

This project is an end-to-end Machine Learning Operations (MLOps) implementation for predicting loan approval decisions using applicant financial and credit information.

The project demonstrates the complete machine learning lifecycle, including:

- Data preprocessing
- Feature engineering
- Model training
- Experiment tracking
- Model versioning
- API deployment
- Containerization
- CI/CD automation
- Cloud deployment
- Monitoring

---

## Business Problem

Financial institutions receive thousands of loan applications every day. Manually evaluating each application is time-consuming and prone to inconsistencies.

The objective of this project is to build a machine learning model that predicts whether a loan application should be approved or rejected based on applicant information.

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| person_age | Applicant age |
| person_gender | Applicant gender |
| person_education | Education level |
| person_income | Annual income |
| person_emp_exp | Employment experience |
| person_home_ownership | Home ownership status |
| loan_amnt | Loan amount |
| loan_intent | Purpose of loan |
| loan_int_rate | Interest rate |
| loan_percent_income | Loan amount as percentage of income |
| cb_person_cred_hist_length | Credit history length |
| credit_score | Credit score |
| previous_loan_defaults_on_file | Previous default history |

Target Variable:

```text
loan_status

0 = Rejected
1 = Approved
```

---

## Project Architecture

```text
Dataset
    ↓
Git + DVC
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Model Training
    ↓
Model Evaluation
    ↓
MLflow Experiment Tracking
    ↓
Model Registry
    ↓
FastAPI Inference API
    ↓
Docker Containerization
    ↓
GitHub Actions CI/CD
    ↓
Google Cloud Run Deployment
```

---

## Technology Stack

### Programming

- Python 3.11

### Data Science

- Pandas
- NumPy
- Scikit-learn
- XGBoost

### MLOps

- Git
- GitHub
- DVC
- AWS S3
- MLflow
- Docker
- GitHub Actions

### API

- FastAPI
- Uvicorn

### Cloud

- Google Cloud Run
- Google Artifact Registry

---

## Feature Engineering

The following engineered features were created:

- Income Per Age
- Credit History Ratio
- Loan Burden Ratio

Categorical variables were encoded using one-hot encoding.

Numerical variables were scaled using StandardScaler.

---

## Model Training

Algorithm:

```text
XGBoost Classifier
```

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Best model artifacts are stored and versioned using DVC.

---

## Experiment Tracking

MLflow is used for:

- Parameter tracking
- Metric tracking
- Model artifact logging
- Experiment comparison

---

## Data Versioning

DVC is used to version:

- Datasets
- Model artifacts
- Scalers
- Feature metadata

Remote storage:

```text
AWS S3
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Loan Prediction

```http
POST /api/v1/predict
```

Request Example:

```json
{
  "person_age": 33,
  "person_gender": "female",
  "person_income": 42920,
  "loan_amnt": 14125,
  "loan_int_rate": 10.37,
  "loan_percent_income": 0.33,
  "cb_person_cred_hist_length": 8,
  "credit_score": 675,
  "previous_loan_defaults_on_file": "No",
  "person_education": "Master",
  "person_home_ownership": "RENT",
  "loan_intent": "HOMEIMPROVEMENT"
}
```

Response Example:

```json
{
  "prediction": 1,
  "probability": 0.9992,
  "result": "Approved"
}
```

---

## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd LoanApprovalPrediction-MLOps-Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull DVC Artifacts

```bash
dvc pull
```

### Run FastAPI

```bash
uvicorn api.main:app --reload
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Docker

Build image:

```bash
docker build -t loan-prediction-api .
```

Run container:

```bash
docker run -p 8000:8000 loan-prediction-api
```

---

## CI/CD

GitHub Actions pipeline performs:

- Dependency installation
- Artifact retrieval
- Automated testing
- Build validation

Pipeline status can be viewed in:

```text
GitHub → Actions
```

---

## Cloud Deployment

Deployment Platform:

```text
Google Cloud Run
```

Public API Endpoint:

```text
https://loan-prediction-api-284234578893.asia-south1.run.app
```

Swagger Documentation:

```text
https://loan-prediction-api-284234578893.asia-south1.run.app/docs
```

---

## Monitoring

Prediction logging is implemented through:

```text
logs/predictions.csv
```

Monitoring metrics:

- Total Predictions
- Class Distribution
- Average Confidence
- Low Confidence Predictions
- Confidence Threshold Analysis

Run monitoring:

```bash
python src/monitor.py
```

---

## Confidence Threshold

Current threshold:

```text
0.60
```

Interpretation:

| Probability | Confidence |
|-------------|------------|
| < 0.60 | Low Confidence |
| ≥ 0.60 | High Confidence |

---

## Retraining Triggers

The model should be retrained when:

- Prediction confidence drops significantly
- Data distribution changes
- Accuracy decreases below baseline
- New business requirements arise
- Monthly retraining schedule is reached

---

## Future Improvements

- Data Drift Detection
- Model Drift Monitoring
- Automated Retraining Pipeline
- Kubernetes Deployment
- Feature Store Integration
- Batch Prediction Pipeline

---

## Author

**Muhammed Jaseem VT**

Data Analyst | Data Science & MLOps Enthusiast

LinkedIn: https://www.linkedin.com/in/jaseem-vt-835ba330a/

GitHub: https://github.com/vtjaseem7
