# 💼 FinBuddy – AI Financial Research & Underwriting Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb6e00.svg)](https://xgboost.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C.svg)](https://langchain.com)

**FinBuddy** is a production-grade Financial AI platform that marries **Agentic LLM multi-tool systems** with **self-trained explainable Machine Learning models (XGBoost + SHAP)** and **robust MLOps & LLMOps observability**.

---

## 🌟 Key Architecture & Capabilities

### 1. 🤖 Multi-Tool Agentic LLM System (LangChain / LangSmith)
- **Credit Risk Underwriting Tool**: Invokes trained XGBoost models to compute default probabilities and explain driving factors.
- **Financial Market Data Tool**: Connects to `yfinance` to retrieve real-time stock quotes, valuation multiples (P/E, EV/EBITDA), analyst consensus, and company news.
- **Financial Mathematics Tool**: Computes Discounted Cash Flow (DCF) intrinsic valuations, Loan EMI amortization schedules, and Sharpe ratios.
- **Full LLMOps Tracing**: Ready for LangSmith / OpenTelemetry tracing, prompt versioning, and latency monitoring.

### 2. 📊 Explainable Credit Risk ML Pipeline (XGBoost + SHAP)
- **XGBoost Classifier**: Trained on financial credit underwriting dynamics (Income, DTI, FICO score, loan-to-income ratio, defaults).
- **SHAP TreeExplainer**: Generates local instance explanations (waterfall attribution charts) highlighting exact risk-increasing and protective factors for regulatory compliance.

### 3. 🛡️ MLOps & Data Drift Observatory
- **Population Stability Index (PSI)** & **Kolmogorov-Smirnov (KS) statistical testing** to track production distribution shifts against baseline training data.
- Macroeconomic stress simulator for scenario testing and model retraining triggers.

### 4. 🖥️ Interactive Web Dashboard & FastAPI Backend
- **Streamlit Frontend**: 4-tab interface featuring an AI Chatbot, Credit Risk & SHAP Studio, Market Intelligence Hub, and MLOps Observatory.
- **FastAPI REST API**: High-performance asynchronous backend exposing endpoints for chat, underwriting, metrics, and drift monitoring.

---

## 📁 Repository Structure

```
FinBuddy/
├── data/                       # Training baseline & synthetic datasets
│   └── credit_baseline.csv
├── models/                     # Serialized XGBoost model, SHAP explainer & metrics
│   ├── credit_risk_xgb.pkl
│   ├── shap_explainer.pkl
│   └── model_metrics.json
├── src/
│   ├── ml/                     # ML & MLOps Pipelines
│   │   ├── dataset.py          # Synthetic realistic credit data generator
│   │   ├── train.py            # XGBoost training & evaluation pipeline
│   │   ├── explain.py          # SHAP local & global attribution engine
│   │   └── drift.py            # PSI & KS-Test drift monitoring
│   ├── tools/                  # Agent tool definitions
│   │   ├── credit_risk_tool.py # Credit underwriting tool
│   │   ├── market_data_tool.py # Live stock & market quotes tool
│   │   └── calculator_tool.py  # Financial math (DCF, EMI, Sharpe)
│   ├── agents/                 # LLM Agent orchestration
│   │   └── financial_agent.py  # Multi-tool agent with fallback router
│   └── api/                    # FastAPI Backend
│       ├── main.py             # REST API server & endpoints
│       └── schemas.py          # Pydantic request/response models
├── frontend/                   # Streamlit Interactive Dashboard
│   └── app.py
├── tests/                      # Automated test suite
│   ├── test_ml.py
│   └── test_agent.py
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup

```bash
# Clone or navigate to the workspace
cd "d:/AI PROJECTS/FinBuddy"

# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env` and configure your API keys (OpenAI / Gemini / LangSmith):
```bash
cp .env.example .env
```
*(Note: FinBuddy includes an intelligent local fallback router, allowing testing even without external API keys!)*

### 3. Train the Credit Risk Model & Fit SHAP Explainer

```bash
python -m src.ml.train
```
This generates the baseline dataset, trains the XGBoost model, fits the SHAP TreeExplainer, and saves serialized artifacts in `models/`.

### 4. Run the FastAPI Backend Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs will be available at `http://localhost:8000/docs`.

### 5. Launch the Streamlit Interactive Dashboard

```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running the Test Suite

```bash
pytest tests/ -v
```

---

## 📜 License
MIT License
