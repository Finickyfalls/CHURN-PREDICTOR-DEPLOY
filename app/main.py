# The FastAPI app. Two JSON endpoints (predict, generate-email) power a
# single-page dashboard that updates without full page reloads.

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.model import load_artifacts, predict_churn
from app.agent import generate_retention_email

app = FastAPI(title="Churn Predictor & Retention Agent")
templates = Jinja2Templates(directory="app/templates")

model, scaler, feature_columns, numeric_cols = load_artifacts()


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(payload: CustomerInput):
    result = predict_churn(payload.dict(), model, scaler, feature_columns, numeric_cols)
    return result


@app.post("/generate-email")
def generate_email(payload: EmailRequest):
    email_text = generate_retention_email(
        customer_summary=payload.customer_summary,
        churn_probability=payload.churn_probability,
        top_drivers=payload.top_drivers
    )
    return {"email": email_text}


@app.get("/health")
def health():
    return {"status": "ok"}
