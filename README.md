# Painel Amanaje
O Painel Amanajé é um projeto de desenvolvimento completo que propõe o uso de um Temporal Fusion Transformer, ou TFT para prever ações da bolsa de valores.

# 🚀 Stock Market Forecasting with Temporal Fusion Transformer (TFT)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Forecasting-red)
![Prefect](https://img.shields.io/badge/Orchestration-Prefect-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)
![Feast](https://img.shields.io/badge/Feature%20Store-Feast-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A **fully containerized end-to-end machine learning pipeline** for **stock price forecasting**, built around **NVIDIA (NVDA)** data from Yahoo Finance.  
This project demonstrates how to integrate modern **MLOps** components — **Airflow**, **MLflow**, **Feast**, **Optuna**, **FastAPI**, **Docker**, and **PyTorch** — into a production-ready architecture.

---

## 🧭 Overview

This project showcases a **complete machine learning workflow** — from data ingestion to monitoring — centered around **Temporal Fusion Transformer (TFT)**, a state-of-the-art deep learning model for time-series forecasting.

The architecture combines:

- **Airflow** → Orchestrates the end-to-end pipeline.  
- **Feast** → Centralized feature store for consistent training and inference features.  
- **PyTorch** → Core deep learning framework for model development.  
- **Optuna** → Hyperparameter optimization integrated with MLflow tracking.  
- **MLflow** → Logs experiments, parameters, and model versions.  
- **FastAPI** → Serves the trained model via REST API.  
- **Docker Compose** → Runs the entire infrastructure locally with reproducibility. 

📈 The main objective is to forecast **NVDA stock prices** using historical data and engineered features such as moving averages, volatility indicators, and time-based signals.

---

## 🧩 System Architecture

```mermaid
graph TD
    A[Data Source: Yahoo Finance] --> B[Prefect ETL Pipeline]
    B --> C[Great Expectations: Data Validation]
    C --> D[Feast: Feature Store]
    D --> E[TFT Model: PyTorch Forecasting]
    E --> F[MLflow: Tracking & Registry]
    F --> G[Streamlit Dashboard]
    E --> H[Prometheus + cAdvisor: Monitoring]
    H --> I[Grafana Dashboard (optional)]