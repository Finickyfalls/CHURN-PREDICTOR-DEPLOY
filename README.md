# Churn Predictor & Retention Agent

A customer churn prediction dashboard combining an XGBoost classifier with an
LLM-powered retention agent, served with FastAPI and deployed via Docker on Railway.

## What it does
- Predicts a customer's churn probability from 18 account/service/billing features
- Explains WHY using per-customer feature contributions (not just global importance)
- Generates a personalized retention email via Groq (openai/gpt-oss-20b) for at-risk customers

## Dataset
IBM Telco Customer Churn -- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Environment variables (set in Railway, never committed)
- `GROQ_API_KEY` -- from https://console.groq.com

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --reload
```
Visit http://localhost:8000

## Run with Docker
```bash
docker build -t churn-predictor .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here churn-predictor
```

## Live Deployment
[Add your Railway deployment link here]
