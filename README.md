# OPD Waiting Time Prediction – Patna

## Project Overview
This project aims to predict the waiting time (in minutes) for patients visiting an Outpatient Department (OPD) in Patna using machine learning.

## Problem Statement
Predict the time a patient waits before entering the doctor's room.

## Dataset
The project includes:
- train.csv – Training dataset.
- later.csv – Later-period dataset for evaluation and drift analysis.
- holdout_features.csv – Unlabelled dataset for final predictions.

## Target Variable
`wait_min` – Patient waiting time in minutes.

## Evaluation Metric
Mean Absolute Error (MAE).

## Technologies
- Python
- Pandas
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Docker
- Kubernetes

## Team
Group G227 – B.Tech Computer Science and Engineering.

## Project Status

- Stage 1: Repository setup and dataset tracking – Completed
- Stage 2: Data cleaning, model training and MLflow tracking – Completed
- Stage 3: FastAPI, API testing, Docker, CI/CD, Kubernetes and JSON logging – Completed
- Stage 4: Drift monitoring, model retraining and holdout predictions – In Progress
