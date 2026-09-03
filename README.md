# Churn Predictor & Retention Agent

A customer churn prediction dashboard combining an XGBoost classifier with an
LLM-powered retention agent, served with FastAPI and deployed via Docker on Railway.

**🔗 Live App:** [Churn Intelligence Dashboard](https://churn-predictor-deploy-production.up.railway.app/)

## What it does

- Predicts a customer's churn probability from 18 account, service, and billing features
- Explains **why** using per-customer feature contributions (XGBoost's `pred_contribs`), not a static global importance chart repeated for every customer
- Generates **two personalized retention emails** (Warm & Personal vs. Professional & Efficient) via Groq, and tracks which tone users prefer — an A/B test of agent "personalities"

## Dataset

IBM Telco Customer Churn — https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Project Workflow

1. **Data Cleaning** — handled a known quirk where `TotalCharges` is stored as text with blank entries for new customers; dropped the `customerID` identifier
2. **EDA** — churn rate by contract type, tenure/monthly charges distributions, correlation heatmap
3. **Preprocessing** — binary mapping for Yes/No columns, one-hot encoding for multi-category columns, `StandardScaler` on numeric features
4. **Model Training** — compared 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM) using `RandomizedSearchCV` with `StratifiedKFold` cross-validation, scored by ROC-AUC
5. **Class Imbalance Handling** — `class_weight='balanced'` / `scale_pos_weight`, since only ~27% of customers in this dataset churn
6. **Agent** — a Groq-powered (`openai/gpt-oss-20b`) retention email generator, grounded in the customer's actual risk drivers rather than generic templates
7. **Deployment** — FastAPI + Docker on Railway, with a dark-themed interactive dashboard (animated risk gauge, collapsible input sections, live driver breakdown)

## Results & Reflection

### Model performance

| Metric | Score |
|---|---|
| Best model | XGBoost (selected over Random Forest, LightGBM, Logistic Regression after comparison) |
| Cross-validated ROC-AUC | 0.849 |
| Test set ROC-AUC | 0.846 |
| **Churn recall** | **0.81** |
| Churn precision | 0.52 |
| Stay recall | 0.73 |

**Why recall matters more than accuracy here:** in a retention context, missing an at-risk customer (a false negative) means losing them with zero chance to intervene, while flagging a loyal customer as at-risk (a false positive) costs almost nothing — at most, an unnecessary email. The model was deliberately tuned to prioritize recall, and it shows: it catches roughly 81% of customers who actually churn, at the cost of a higher false-alarm rate (52% precision). This is the right trade-off for this business problem.

**Top churn drivers** (from feature importance): contract length dominates by a wide margin — customers on two-year and one-year contracts are far less likely to churn than month-to-month customers. Fiber optic internet, lack of online security, and paying by electronic check were the next strongest signals. This matches well-documented patterns in this dataset and gave the retention agent real, specific material to reference in its emails rather than generic language.

### On the retention agent

The agent is grounded strictly in the customer's actual data — it's instructed never to invent details, always offer exactly one concrete incentive, and adjust tone based on risk level (a 12%-risk loyal customer gets an appreciation-toned email, not an urgent retention plea). Testing confirmed the two personas ("Warm & Personal" vs. "Professional & Efficient") produce genuinely different-sounding output for the same underlying data, which is what makes the A/B comparison meaningful rather than cosmetic.

### Would I trust this in production tomorrow?

**Partially — with clear, specific caveats.**

What's solid: the model's recall on the class that actually matters (churn) is strong, the explanation layer is genuinely per-customer rather than generic, and the agent's output is grounded and appropriately toned.

What I'd fix before real deployment:
- **Precision is only 52%** — half of all "at-risk" flags are false alarms. At scale, that means a lot of unnecessary retention emails going to customers who were never going to leave. Worth tuning the classification threshold or adding a secondary review step for borderline cases rather than auto-triggering outreach on every flagged customer.
- **The A/B vote counter is in-memory only** — it resets to zero on every redeploy or restart. This is fine for a demo, but a real production version would need this persisted in an actual database (e.g. Postgres) before the A/B results could be trusted over time.
- **No real send capability** — by design. The agent drafts emails for human review; it does not autonomously send anything to real customers. Before adding real sending, proper safeguards (rate limiting, human approval step, sender verification) would be required — sending real messages from a model's output without a human in the loop is a meaningful trust and abuse-risk jump from what's built here.
- **Groq's free-tier models can change** — this project already hit one deprecation (Llama models retired mid-project) and one reasoning-token quirk (empty responses until `max_tokens` was raised). A production system would need monitoring for exactly this kind of upstream model change.

## Repository Contents

| File | Description |
|---|---|
| `Churn_Predictor_Training.ipynb` | Full notebook: cleaning, EDA, model training/evaluation, agent prototyping, deployment packaging |
| `app/main.py` | FastAPI app (prediction + email generation + A/B tracking endpoints) |
| `app/model.py` | Model loading, preprocessing, and per-customer explanation logic |
| `app/agent.py` | Retention email generation (two personas) via Groq |
| `app/templates/index.html` | Interactive dashboard UI |
| `models/` | Saved model, scaler, and feature column artifacts |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build instructions |
| `railway.json` | Railway build configuration |

## Environment Variables (set in Railway, never committed)

- `GROQ_API_KEY` — from https://console.groq.com

## Run Locally

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

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn, XGBoost, LightGBM
- **Agent:** Groq (openai/gpt-oss-20b)
- **Backend:** FastAPI
- **Deployment:** Docker, Railway
