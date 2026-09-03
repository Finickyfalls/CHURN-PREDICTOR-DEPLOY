from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.model import load_artifacts, predict_churn
from app.agent import generate_email_variants

app = FastAPI(title="Churn Predictor & Retention Agent")
templates = Jinja2Templates(directory="app/templates")

model, scaler, feature_columns, numeric_cols = load_artifacts()

# In-memory vote tally for the A/B personality test.
# NOTE: this resets to zero every time the app restarts or redeploys -- a
# production version would persist this in a real database instead. Fine
# for a demo/capstone deployment, but worth disclosing as a known limitation.
preference_counts = {"warm": 0, "professional": 0}


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class EmailRequest(BaseModel):
    customer_summary: str
    churn_probability: float
    top_drivers: list[str]


class PreferenceInput(BaseModel):
    persona: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(payload: CustomerInput):
    result = predict_churn(payload.dict(), model, scaler, feature_columns, numeric_cols)
    return result


@app.post("/generate-email")
def generate_email(payload: EmailRequest):
    # Returns BOTH persona variants for the same customer, so they can be
    # compared side by side -- this is the A/B piece of the stretch goal.
    variants = generate_email_variants(
        customer_summary=payload.customer_summary,
        churn_probability=payload.churn_probability,
        top_drivers=payload.top_drivers
    )
    return variants


@app.post("/track-preference")
def track_preference(payload: PreferenceInput):
    if payload.persona not in preference_counts:
        return {"error": "unknown persona"}
    preference_counts[payload.persona] += 1
    return {"counts": preference_counts}


@app.get("/stats")
def stats():
    total = sum(preference_counts.values())
    return {"counts": preference_counts, "total_votes": total}


@app.get("/health")
def health():
    return {"status": "ok"}
