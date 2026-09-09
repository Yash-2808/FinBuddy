# ⚡ FinBuddy – AI Financial Research, Underwriting & MLOps Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-eb6e00.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C.svg)](https://langchain.com)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20Credit%20Risk-20BEFF.svg)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7.svg)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **FinBuddy** is a production-grade Autonomous Financial AI platform combining **Multi-Tool Agentic LLMs**, **High-Accuracy Explainable Machine Learning (XGBoost + SHAP TreeExplainer)** trained on **32,581 Kaggle Consumer Credit Records**, **Real-Time Market Valuation Sandboxes**, and **Continuous MLOps Concept Drift Observability (PSI & KS-Test)**.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Dataset Specifications](#-dataset-specifications)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Interactive Web Dashboard (UI)](#-interactive-web-dashboard-ui)
- [FastAPI REST API Reference](#-fastapi-rest-api-reference)
- [Installation & Local Setup](#-installation--local-setup)
- [Docker & Cloud Deployment (Render)](#-docker--cloud-deployment-render)
- [Testing Suite](#-testing-suite)
- [Tech Stack](#-tech-stack)
- [Governance & Regulatory Compliance](#-governance--regulatory-compliance)
- [License](#-license)

---

## 🌟 Executive Overview

Modern financial underwriting and equity research require combining quantitative modeling, regulatory compliance, macroeconomic awareness, and real-time market data. **FinBuddy** unifies these disparate disciplines into an integrated platform:

1. **Autonomous Financial Research Agent**: A multi-tool LangChain reasoning agent capable of answering complex inquiries, evaluating loan applicants, running intrinsic valuations, and summarizing equity metrics.
2. **Explainable Underwriting Engine**: An optimized **XGBoost 2.0** model trained on **32,581 Kaggle borrower records** achieving **0.9449 ROC-AUC** and **93.06% accuracy**, paired with **SHAP TreeExplainer** for transparent factor attribution.
3. **Multi-Category Credit Underwriting**: Ingests categorical loan parameters including **Home Ownership** (`RENT`, `OWN`, `MORTGAGE`), **Loan Intent** (`EDUCATION`, `VENTURE`, `MEDICAL`, `PERSONAL`, `DEBTCONSOLIDATION`), **Credit Grade** (`A` through `G`), and **Default History**.
4. **Discounted Cash Flow (DCF) Sandbox**: Interactive 5-year intrinsic valuation models with sensitivity sliders for WACC, terminal growth rates, and cash flow projections.
5. **MLOps Concept Drift Radar**: Real-time statistical distribution monitoring using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests to alert against macroeconomic stress.
6. **Ultra-High-Fidelity Cyber-Fintech UI**: Clean, mobile-responsive dark interface inspired by modern fintech design systems with dynamic aurora mesh animations, scroll reveals, and glassmorphic bento cards.

---

## 🚀 Key Features & Capabilities

### 1. 🤖 Autonomous Multi-Tool Financial Agent
- **Natural Language Router**: Parses natural financial language inquiries into tool execution plans.
- **Credit Underwriting Tool**: Evaluates applicant attributes (Income, Age, Loan Amount, Interest Rate, History, Home Ownership, Loan Intent, Grade, Defaults) and outputs risk tiers and decisions.
- **Market Data Tool**: Live integration with `yfinance` to retrieve stock quotes, valuation multiples (Trailing P/E, Forward P/E, Market Cap), and historical price actions.
- **Financial Mathematics Calculator**: Executes Discounted Cash Flow (DCF), Monthly Loan EMI amortization, and Sharpe ratio calculations.
- **Offline Fallback Engine**: Seamless heuristic router for local execution without requiring external API keys.

### 2. 📊 Explainable Credit Risk ML Engine (XGBoost + SHAP)
- **Extreme Gradient Boosting (`XGBClassifier`)**:
  - `350` estimators, `max_depth=6`, `learning_rate=0.06`, `subsample=0.88`, `colsample_bytree=0.88`.
  - Binary default prediction with probability calibration on 32,581 Kaggle consumer credit records.
- **SHAP (SHapley Additive exPlanations) TreeExplainer**:
  - Computes exact local log-odds attributions for each feature across the 26-feature encoded space.
  - Automatically synthesizes plain-language Underwriting Memorandums categorizing **Top Risk Drivers** and **Top Protective Factors**.
- **Visual Diagnostics**:
  - Interactive SHAP waterfall horizontal bar chart.
  - 5-Axis Spider Radar (Loan Grade Strength, Income Power, Credit History, Employment Stability, Low Leverage).

### 3. 📈 Market Intelligence & DCF Valuation Sandbox
- Live technical equity price charts with candlestick rendering and 20-period Moving Average (SMA 20) overlays.
- Real-time valuation multiples dashboard (Trailing P/E, Forward P/E, Market Cap, Current Price).
- Interactive 5-Year DCF model calculating Present Value of Cash Flows, Terminal Value, Enterprise Value, Fair Share Price, and Margin of Safety against current market price.

### 4. 🛡️ MLOps & Distribution Drift Observatory
- **Population Stability Index (PSI)**: Monitors distribution divergence per feature (`PSI < 0.10`: Stable, `0.10 - 0.25`: Moderate Drift, `> 0.25`: Critical Shift).
- **Kolmogorov-Smirnov (KS-Test)**: Computes statistical distances and p-values between baseline training data and live production batches.
- **Macroeconomic Stress Engine**: Interactive slider simulating inflation, rate hikes, and unemployment to stress-test model resilience in real time.
- **Distribution Shift Histogram**: Visual overlay comparing baseline training distributions against live production features.

### 5. 📜 Governance & Live Decision Audit Trail
- Comprehensive Model Governance Card documenting hyperparameters, baseline dataset specs, and regulatory standards.
- Real-time session decision audit log capturing timestamped applicant profiles, default probabilities, and underwriting verdicts.

---

## 📁 Dataset Specifications

FinBuddy is trained on the benchmark **[Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)** containing **32,581 consumer credit records**:

| Feature Name | Type | Categories / Distribution | Description |
| :--- | :---: | :---: | :--- |
| `person_age` | Numeric | 18 – 85 yrs | Age of the borrower |
| `person_income` | Numeric | \$4,000 – \$1,000,000+ | Annual gross income |
| `person_emp_length` | Numeric | 0 – 45 yrs | Employment tenure (median imputed) |
| `person_home_ownership` | Categorical | `RENT`, `OWN`, `MORTGAGE`, `OTHER` | Housing status |
| `loan_intent` | Categorical | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | Loan purpose / intent |
| `loan_grade` | Categorical | `A`, `B`, `C`, `D`, `E`, `F`, `G` | Risk-based credit grade |
| `loan_amnt` | Numeric | \$500 – \$100,000 | Requested loan principal |
| `loan_int_rate` | Numeric | 5.42% – 23.22% | Loan interest rate (grade-imputed) |
| `loan_percent_income` | Numeric | 0.01 – 0.85 | Loan-to-Income (LTI) ratio |
| `cb_person_default_on_file`| Categorical | `N`, `Y` | Historical default record |
| `cb_person_cred_hist_length`| Numeric | 1 – 40 yrs | Length of credit bureau history |
| **`loan_status` (Target)** | Binary | `0` (Non-Default) / `1` (Default) | Binary default classification |

---

## 📊 Model Performance & Evaluation

The XGBoost Credit Risk Classifier was evaluated on a stratified holdout test set of **4,886 applicant records** (from the 32,573 cleaned Kaggle dataset):

| Metric | Score | Status | Description |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.9449** | 🟢 Exceptional | Area under Receiver Operating Characteristic curve |
| **Test Accuracy** | **93.06%** | 🟢 Optimal | Overall correct classification rate on holdout data |
| **Precision** | **93.53%** | 🟢 High Confidence | Proportion of true defaults among positive predictions |
| **Recall (Sensitivity)** | **73.26%** | 🟢 High Coverage | Proportion of actual defaults successfully detected |
| **F1-Score** | **0.8217** | 🟢 Robust | Harmonic mean of precision and recall |
| **PR-AUC** | **0.8912** | 🟢 Calibrated | Area under Precision-Recall curve |
| **Brier Score** | **0.0526** | 🟢 Calibrated | Mean squared probability error (closer to 0 is superior) |

---

## 🏗️ System Architecture

```
                               ┌────────────────────────────────────────┐
                               │       Streamlit Frontend (UI)          │
                               │   (Aurora Mesh, Bento HUD, Charts)     │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────────┐
                               │         FastAPI REST API Layer         │
                               │   (/api/chat, /api/credit, /api/drift) │
                               └──────────────────┬─────────────────────┘
                                                  │
                  ┌───────────────────────────────┼───────────────────────────────┐
                  ▼                               ▼                               ▼
     ┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
     │   FinBuddy Agent Core  │      │  Credit Risk & SHAP    │      │   MLOps Drift Radar    │
     │ (LangChain Orchestrator│      │ (XGBoost 2.0 Engine &  │      │  (PSI & KS Statistical │
     │  & Tool Execution)     │      │  SHAP TreeExplainer)   │      │   Distribution Test)   │
     └────────────┬───────────┘      └────────────┬───────────┘      └────────────┬───────────┘
                  │                               │                               │
     ┌────────────┴───────────┐      ┌────────────┴───────────┐      ┌────────────┴───────────┐
     │ • Credit Risk Tool     │      │ • credit_risk_xgb.pkl  │      │ • Kaggle Baseline Data │
     │ • Market Data (yfinance│      │ • shap_explainer.pkl   │      │ • Stress Simulator     │
     │ • DCF/EMI Calculator   │      │ • model_metrics.json   │      │ • PSI/KS Matrix        │
     └────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 📁 Repository Structure

```
FinBuddy/
├── .streamlit/                 # Streamlit configuration
│   └── config.toml             # Dark theme & headless server config
├── data/                       # Kaggle dataset & processed baseline
│   ├── credit_risk_dataset.csv # 32,581 raw Kaggle credit records
│   ├── credit_baseline.csv     # Cleaned 26-feature encoded baseline
│   └── preprocessor_meta.json  # Imputation medians & categorical levels
├── models/                     # Serialized ML & Explainability artifacts
│   ├── credit_risk_xgb.pkl     # Trained XGBoost binary classifier
│   ├── shap_explainer.pkl      # Pre-fit SHAP TreeExplainer
│   └── model_metrics.json      # Evaluation metrics & validation record
├── src/
│   ├── agents/                 # LLM Agent Orchestration
│   │   └── financial_agent.py  # Multi-tool agent with fallback router
│   ├── api/                    # FastAPI REST API Backend
│   │   ├── main.py             # Server endpoints & startup lifecycle
│   │   └── schemas.py          # Pydantic request & response schemas
│   ├── ml/                     # ML Training & MLOps Pipelines
│   │   ├── dataset.py          # Kaggle dataset loader & preprocessor
│   │   ├── train.py            # XGBoost training & metric evaluation
│   │   ├── explain.py          # SHAP attribution & narrative generator
│   │   └── drift.py            # PSI & KS statistical drift detector
│   └── tools/                  # LangChain Agent Tool Definitions
│       ├── credit_risk_tool.py # Credit risk scoring tool
│       ├── market_data_tool.py # Live stock & market quotes tool
│       └── calculator_tool.py  # Financial math (DCF, EMI, Sharpe)
├── frontend/                   # Interactive Web Application
│   └── app.py                  # Streamlit dashboard with Cyber-Fintech UI
├── tests/                      # Automated Unit Test Suite
│   ├── test_ml.py              # ML pipeline & MLOps tests
│   └── test_agent.py           # Agent orchestrator & tool tests
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Multi-service composition (UI + API)
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git ignore rules for Python & caches
├── requirements.txt            # Python package dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 🖥️ Interactive Web Dashboard (UI)

The frontend is built with **Streamlit** and styled using custom CSS to deliver a premium fintech experience:

1. **💬 AI Copilot**:
   - Conversational assistant with animated waveform indicator.
   - Quick prompt presets (e.g., "$25k Loan Underwriting", "NVDA Valuation", "DCF intrinsic model", "Loan EMI calculation").
   - Step-by-step tool execution telemetry expanders.
2. **📊 Underwriting & SHAP Studio**:
   - Quick-load archetypes (*Prime Executive, Near-Prime Business, Subprime, Young Graduate*).
   - Real-time parameter inputs including **Home Ownership**, **Loan Intent**, **Loan Grade**, and **Default History**.
   - Gauge meter, decision verdict HUD, SHAP waterfall bar chart, and 5-axis spider radar.
3. **📈 Market & DCF Hub**:
   - Ticker selector and historical horizon view (`1mo` to `5y`).
   - Candlestick price action charts with SMA-20 and volume bars.
   - Interactive 5-year DCF model with live valuation comparison.
4. **🛡️ Drift Radar**:
   - Statistical drift scoring (System Health index, Critical vs Moderate drift count).
   - Full PSI and KS test metric table per feature across 32k baseline rows.
   - Interactive histogram overlay comparing training baseline vs live drifted distributions.
5. **📜 Governance & Logs**:
   - Complete model card specifications.
   - Live session underwriting audit table tracking every scored applicant.

---

## 🔌 FastAPI REST API Reference

The backend runs on **FastAPI** on port `8000`. Full interactive documentation is available at `http://localhost:8000/docs`.

### 1. Health Check
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "xgboost_classifier": true,
    "shap_explainer": true
  }
}
```

### 2. Credit Risk Assessment
- **Endpoint**: `POST /api/credit/score`
- **Sample Request**:
```json
{
  "person_age": 34,
  "person_income": 75000,
  "person_emp_length": 5.0,
  "loan_amnt": 15000,
  "loan_int_rate": 10.5,
  "cb_person_cred_hist_length": 7.0,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "cb_person_default_on_file": "N"
}
```
- **Sample Response**:
```json
{
  "default_probability": 0.0346,
  "approval_score": 96.5,
  "recommendation": "APPROVE",
  "risk_tier": "LOW RISK (Prime)",
  "narrative_explanation": "Applicant default probability is 3.5% (LOW RISK). Underwriting Decision: APPROVE.",
  "top_risk_drivers": [],
  "top_protective_factors": [
    { "feature": "person_income", "value": 75000, "shap_value": -1.2131 },
    { "feature": "loan_percent_income", "value": 0.20, "shap_value": -0.8052 }
  ]
}
```

---

## ⚡ Installation & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Yash-2808/FinBuddy.git
cd FinBuddy
```

### 3. Create & Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Train Model & Generate Artifacts (Optional)
```bash
python -m src.ml.train
```

### 6. Run the FastAPI Backend Server
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the Swagger UI.

### 7. Run the Streamlit Dashboard
```bash
python -m streamlit run frontend/app.py --server.port 8501
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 🐳 Docker & Cloud Deployment (Render)

### 1. Run with Docker Locally
```bash
# Build the Docker image
docker build -t finbuddy .

# Run the container
docker run -p 8501:8501 finbuddy
```
Open **[http://localhost:8501](http://localhost:8501)**.

### 2. Run with Docker Compose (Frontend + Backend)
```bash
docker-compose up --build
```
- Streamlit UI: `http://localhost:8501`
- FastAPI REST API: `http://localhost:8000`

### 3. Deploy to Render (Cloud)
1. Fork or push this repository to GitHub: `https://github.com/Yash-2808/FinBuddy.git`
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** as the Environment.
5. Click **Deploy Web Service**!

---

## 🧪 Testing Suite

FinBuddy includes a unit test suite verifying ML pipelines, SHAP explainability, drift detection, tools, and agent fallback routers:

```bash
# Run tests via Python's built-in unittest runner
python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Coverage Summary:
- `test_dataset_generation`: Validates distributions, types, and Kaggle baseline constraints.
- `test_model_training_and_artifacts`: Validates XGBoost training and artifact serialization.
- `test_shap_explainer_inference`: Tests SHAP feature contribution calculations.
- `test_drift_detection_psi`: Verifies PSI and KS statistical alerts under baseline vs drifted distributions.
- `test_credit_risk_tool_execution`: Validates LangChain underwriting tool outputs.
- `test_calculator_tool_dcf`: Validates DCF valuation mathematical correctness.
- `test_calculator_tool_emi`: Validates loan amortization formula calculations.
- `test_agent_orchestrator_fallback`: Validates autonomous agent fallback routing.

---

## 💻 Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **AI & Agent Orchestration** | LangChain, LangSmith / OpenTelemetry tracing ready, Heuristic Fallback Router |
| **Machine Learning** | XGBoost 2.0, Scikit-Learn, NumPy, Pandas |
| **Explainability (XAI)** | SHAP (SHapley Additive exPlanations TreeExplainer) |
| **MLOps & Observability** | Population Stability Index (PSI), SciPy (Kolmogorov-Smirnov Test) |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit, Plotly Express & Graph Objects, Custom CSS/JS Animations |
| **Market Intelligence** | yfinance |
| **Container & Cloud** | Docker, Docker Compose, Render |
| **Serialization & Storage** | Joblib, JSON |

---

## ⚖️ Governance & Regulatory Compliance

In regulated consumer credit underwriting (e.g., US **FCRA** - Fair Credit Reporting Act and **ECOA** - Equal Credit Opportunity Act):
- **Adverse Action Transparency**: FinBuddy utilizes SHAP attribution values to explicitly state why an applicant was declined or approved, isolating top risk drivers (e.g., high debt-to-income ratio, loan grade, interest burden).
- **Auditability**: Every decision executed in a session is logged in the `session_audit` trail with complete feature payloads and probability outputs.
- **Fairness & Non-Discrimination**: Protected attributes (race, gender, marital status) are excluded from model training to adhere to fair lending standards.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>FinBuddy AI</b> • Built with ❤️ for Next-Generation Autonomous Financial Intelligence
</p>
