# Loan Approval Prediction MLOps Project

## Problem Statement

Predict whether a loan application should be approved or rejected.

## Dataset

Loan approval dataset containing applicant financial and credit information.

## Feature Engineering

- credit_history_ratio
- income_per_age

## Model

XGBoost Classifier

## Experiment Tracking

MLflow used for:
- Parameter logging
- Metric logging
- Artifact tracking

## Data Versioning

DVC with AWS S3 remote storage.

## API

FastAPI endpoint:

/api/v1/predict

## Testing

Pytest automated tests.

## Containerization

Dockerized application.

## CI/CD

GitHub Actions pipeline.

## Deployment

Google Cloud Run.

## Monitoring

Prediction logging and confidence monitoring implemented.

## Future Improvements

- Data drift monitoring
- Automated retraining
- Batch inference pipeline
