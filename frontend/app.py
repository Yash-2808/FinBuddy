"""FinBuddy – Ultra-High-Fidelity AI Fintech Terminal & Underwriting Platform.
Features:
- Dynamic Aurora Gradient Mesh Background Animation
- Flowing Ambient Nebula & Floating Glow Particles
- Premium Cyber-Fintech Color Grading (Electric Cyan, Violet Indigo, Neon Mint, Golden Amber)
- Universal Glassmorphic Bento Containers (Streamlit Border Containers)
- Clean, Centered Segmented Navigation Capsule
- Interactive 5-Axis Radar, SHAP Waterfall, Live Candlesticks, and MLOps Drift Engine.
"""

import sys
import os

# Ensure the project root is in sys.path for cloud platforms (Render, Streamlit Cloud, Docker)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# Configure Streamlit
st.set_page_config(
    page_title="FinBuddy | AI Financial Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# Premium Aurora & Cyber-Fintech Animated CSS Design System
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    * {
        box-sizing: border-box;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background-color: #030712 !important;
        color: #FFFFFF !important;
    }

    /* Main container padding for spaciousness */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }
    
    /* Global text visibility & contrast guarantees */
    p, span, div, li, a, h1, h2, h3, h4, h5, h6, label, td, th {
        color: #F8FAFC !important;
    }

    .stMarkdown p {
        color: #E2E8F0 !important;
        font-size: 0.98rem !important;
        line-height: 1.75 !important;
        margin-bottom: 0.75rem !important;
    }

    .stMarkdown strong {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: #111827 !important;
        color: #38BDF8 !important;
        border: 1px solid #1F2937 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
    }

    /* ---------------------------------------------------- */
    /* DYNAMIC AURORA & NEBULA BACKGROUND ANIMATION         */
    /* ---------------------------------------------------- */
    @keyframes auroraMesh {
        0% {
            background-position: 0% 0%, 100% 100%, 0% 100%, 50% 50%;
            filter: hue-rotate(0deg);
        }
        50% {
            background-position: 100% 50%, 0% 50%, 100% 0%, 50% 100%;
            filter: hue-rotate(15deg);
        }
        100% {
            background-position: 0% 0%, 100% 100%, 0% 100%, 50% 50%;
            filter: hue-rotate(0deg);
        }
    }

    @keyframes pulseLive {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.8); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(0, 242, 254, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }
    }

    @keyframes textShine {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes laserNav {
        0%, 100% {
            border-color: rgba(0, 242, 254, 0.35);
            box-shadow: 0 10px 35px -10px rgba(0, 242, 254, 0.25), inset 0 0 15px rgba(0, 242, 254, 0.08);
        }
        50% {
            border-color: rgba(139, 92, 246, 0.6);
            box-shadow: 0 15px 45px -10px rgba(139, 92, 246, 0.35), inset 0 0 25px rgba(139, 92, 246, 0.12);
        }
    }

    @keyframes wavePulse {
        0%, 100% { height: 6px; }
        50% { height: 22px; }
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Living Ambient Background Mesh */
    .stApp {
        background: 
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.22) 0%, transparent 60%),
            radial-gradient(circle 600px at 10% 20%, rgba(0, 242, 254, 0.12) 0%, transparent 50%),
            radial-gradient(circle 650px at 90% 65%, rgba(139, 92, 246, 0.12) 0%, transparent 50%),
            radial-gradient(circle 500px at 50% 90%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
            radial-gradient(#1E293B 1px, transparent 1px) !important;
        background-size: 200% 200%, 150% 150%, 150% 150%, 150% 150%, 32px 32px !important;
        background-attachment: fixed !important;
        animation: auroraMesh 18s ease-in-out infinite !important;
    }

    /* ---------------------------------------------------- */
    /* UI COMPONENTS & COLOR GRADING                        */
    /* ---------------------------------------------------- */
    
    /* Top Capsule Nav */
    .fin-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 15, 30, 0.88);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 100px;
        padding: 16px 36px;
        margin-bottom: 36px;
        box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.85);
        animation: laserNav 7s infinite ease-in-out, fadeInUp 0.8s ease-out;
    }
    
    .fin-brand {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .fin-logo-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #6366F1 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        color: #04060A;
        font-size: 24px;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.7);
    }

    .fin-logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: #FFFFFF !important;
    }

    .fin-tagline {
        font-size: 0.78rem;
        background: rgba(255, 255, 255, 0.08);
        color: #CBD5E1 !important;
        padding: 6px 16px;
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .fin-status-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.5);
        padding: 8px 22px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #34D399 !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    }

    .status-dot {
        width: 9px;
        height: 9px;
        background: #10B981;
        border-radius: 50%;
        animation: pulseLive 2s infinite ease-in-out;
    }

    /* Hero Section with Generous Spacing */
    .fin-hero {
        text-align: center;
        padding: 24px 20px 36px 20px;
        margin-bottom: 24px;
        animation: fadeInUp 0.9s ease-out;
    }

    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.15), rgba(99, 102, 241, 0.15));
        border: 1px solid rgba(0, 242, 254, 0.45);
        padding: 8px 26px;
        border-radius: 50px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #00F2FE !important;
        margin-bottom: 18px;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.25);
    }

    .fin-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.18;
        letter-spacing: -1.5px;
        margin-bottom: 16px;
        color: #FFFFFF !important;
    }

    .fin-subtext {
        color: #CBD5E1 !important;
        font-size: 1.1rem;
        max-width: 820px;
        margin: 0 auto;
        line-height: 1.7;
    }

    /* Metric Badges with Spacious Margins */
    .fin-metric-badge {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 22px 26px;
        text-align: left;
        margin-bottom: 24px;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .fin-metric-badge:hover {
        border-color: #00F2FE;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.28);
        transform: translateY(-3px);
    }

    .fin-metric-label {
        font-size: 0.75rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }

    .fin-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 6px;
        letter-spacing: -0.5px;
    }

    /* Waveform Frequency Animation */
    .waveform-container {
        display: inline-flex;
        align-items: flex-end;
        gap: 4px;
        height: 24px;
        margin-left: 12px;
    }
    
    .wave-bar {
        width: 3.5px;
        background: linear-gradient(180deg, #00F2FE, #8B5CF6);
        border-radius: 3px;
        animation: wavePulse 1.2s ease-in-out infinite alternate;
    }
    .wave-bar:nth-child(1) { animation-delay: 0.1s; height: 10px; }
    .wave-bar:nth-child(2) { animation-delay: 0.3s; height: 20px; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; height: 15px; }
    .wave-bar:nth-child(4) { animation-delay: 0.4s; height: 22px; }
    .wave-bar:nth-child(5) { animation-delay: 0.15s; height: 9px; }

    /* Decision HUD Badges with Dynamic Glow */
    .fin-decision-approved {
        background: radial-gradient(circle at center, rgba(16, 185, 129, 0.35) 0%, rgba(6, 78, 59, 0.7) 100%);
        border: 1px solid #10B981;
        border-radius: 18px;
        padding: 24px 28px;
        text-align: center;
        color: #D1FAE5 !important;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.4);
    }

    .fin-decision-review {
        background: radial-gradient(circle at center, rgba(245, 158, 11, 0.35) 0%, rgba(120, 53, 15, 0.7) 100%);
        border: 1px solid #F59E0B;
        border-radius: 18px;
        padding: 24px 28px;
        text-align: center;
        color: #FEF3C7 !important;
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.4);
    }

    .fin-decision-rejected {
        background: radial-gradient(circle at center, rgba(244, 63, 94, 0.35) 0%, rgba(136, 19, 55, 0.7) 100%);
        border: 1px solid #F43F5E;
        border-radius: 18px;
        padding: 24px 28px;
        text-align: center;
        color: #FFE4E6 !important;
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.4);
    }

    /* Chat Messages - High Contrast & Roomy */
    [data-testid="stChatMessage"] {
        background: rgba(17, 24, 39, 0.92) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.5) !important;
    }

    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {
        color: #F1F5F9 !important;
        font-size: 0.98rem !important;
        line-height: 1.65 !important;
    }

    /* ---------------------------------------------------- */
    /* UNIVERSAL TAB CAPSULE FOR ALL BROWSERS & CLOUD ENVS  */
    /* ---------------------------------------------------- */
    div[data-testid="stTabs"] {
        width: 100% !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: rgba(10, 15, 30, 0.94) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        padding: 6px 10px !important;
        border-radius: 100px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        gap: 8px !important;
        width: fit-content !important;
        margin: 12px auto 36px auto !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.7) !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 50px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        color: #94A3B8 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.25s ease !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #6366F1 100%) !important;
        color: #04060A !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.45) !important;
    }

    /* ---------------------------------------------------- */
    /* STREAMLIT BORDERED CONTAINERS AS BENTO CARDS         */
    /* ---------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(13, 20, 36, 0.90) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 24px !important;
        padding: 28px 32px !important;
        margin-bottom: 28px !important;
        box-shadow: 0 20px 45px -15px rgba(0, 0, 0, 0.7) !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 242, 254, 0.45) !important;
        box-shadow: 0 30px 65px -15px rgba(0, 242, 254, 0.22) !important;
    }

    .stButton button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        color: #F8FAFC !important;
        border-radius: 14px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        margin-top: 4px !important;
        margin-bottom: 4px !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.3), rgba(99, 102, 241, 0.35)) !important;
        border-color: #00F2FE !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Streamlit widget labels with clean margin */
    .stNumberInput label, .stSlider label, .stSelectbox label, .stTextInput label {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 8px !important;
    }

    /* Generous spacing between inputs */
    .stNumberInput, .stSlider, .stSelectbox, .stTextInput {
        margin-bottom: 16px !important;
    }

    /* Expanders styling with clean padding */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        padding: 14px 20px !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-top: 14px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(13, 20, 36, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        margin-top: 16px !important;
        margin-bottom: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Module Loaders
# -------------------------------------------------------------
from src.ml.explain import CreditRiskExplainer
from src.ml.drift import DriftDetector
from src.ml.dataset import generate_credit_dataset, FEATURE_NAMES
from src.agents.financial_agent import FinBuddyAgent


@st.cache_resource
def get_agent():
    return FinBuddyAgent()


@st.cache_resource
def get_explainer():
    return CreditRiskExplainer()


@st.cache_resource
def get_drift_detector():
    return DriftDetector()


# State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "⚡ **FinBuddy AI Engine Online.**\n\n"
                "I am your autonomous financial copilot powered by multi-tool LangChain agents, "
                "self-trained XGBoost credit risk scoring with SHAP explainability, and live market intelligence. "
                "Select a prompt below or type your inquiry to begin."
            ),
            "tool_calls": []
        }
    ]

if "session_audit" not in st.session_state:
    st.session_state.session_audit = []


# -------------------------------------------------------------
# Top Floating Capsule Navigation
# -------------------------------------------------------------
st.markdown("""
<div class="fin-nav">
    <div class="fin-brand">
        <div class="fin-logo-icon">⚡</div>
        <div class="fin-logo-text">FINBUDDY <span style="font-weight: 300; opacity: 0.7;">AI</span></div>
        <div class="fin-tagline">Autonomous Financial Intelligence</div>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <div class="fin-status-pill">
            <span class="status-dot"></span>
            <span>Agentic Systems Live</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Hero Section with Shimmer & Bento Stats
# -------------------------------------------------------------
st.markdown("""
<div class="fin-hero">
    <div class="hero-badge-pill">
        <span>⚡</span> Next-Gen Financial AI Platform • v2.0 Production
    </div>
    <div class="fin-headline">Intelligent Underwriting & Market Analytics</div>
    <div class="fin-subtext">
        Autonomous multi-tool agent system combining explainable XGBoost + SHAP credit risk models,
        real-time market valuation, and continuous MLOps distribution drift monitoring.
    </div>
</div>
""", unsafe_allow_html=True)

# Top Live KPI Strip (Bento Row) with clean margins
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="fin-metric-badge">
        <div class="fin-metric-label">ML Core Model</div>
        <div class="fin-metric-value" style="color: #00F2FE !important;">XGBoost 2.0</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="fin-metric-badge">
        <div class="fin-metric-label">Validation ROC-AUC</div>
        <div class="fin-metric-value" style="color: #34D399 !important;">0.9787</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="fin-metric-badge">
        <div class="fin-metric-label">Explainability</div>
        <div class="fin-metric-value" style="color: #C084FC !important;">SHAP Tree</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown("""
    <div class="fin-metric-badge">
        <div class="fin-metric-label">MLOps Observability</div>
        <div class="fin-metric-value" style="color: #FBBF24 !important;">PSI & KS-Test</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Main Navigation Tabs (Segmented Capsule Style)
# -------------------------------------------------------------
tab_chat, tab_risk, tab_market, tab_ops, tab_audit = st.tabs([
    "💬 AI Copilot",
    "📊 Underwriting & SHAP",
    "📈 Market & DCF Hub",
    "🛡️ Drift Radar",
    "📜 Governance & Logs"
])


# =============================================================
# TAB 1: AI Copilot (With Animated Waveform Stream)
# =============================================================
with tab_chat:
    with st.container(border=True):
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 10px;">
                🤖 FinBuddy Multi-Tool Financial Agent
                <div class="waveform-container">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>
            </div>
        </div>
        <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 24px;">
            Natural language routing across credit underwriting, market telemetry, and intrinsic valuation engines.
        </div>
        """, unsafe_allow_html=True)

        # Fast Prompt Bar with clean spacing
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #CBD5E1; margin-bottom: 12px;'>⚡ Instant Analysis Prompts:</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            if st.button("📋 Underwrite $25k Loan (740 FICO)", use_container_width=True):
                st.session_state.active_query = "Evaluate credit risk for applicant with income $85,000, loan $25,000, credit score 740, DTI 0.24, age 33"
        with p2:
            if st.button("📈 NVDA Valuation & Multiples", use_container_width=True):
                st.session_state.active_query = "Analyze NVDA stock metrics, valuation multiples, and latest summary"
        with p3:
            if st.button("🏢 Run DCF on $100k FCF", use_container_width=True):
                st.session_state.active_query = "Calculate DCF intrinsic valuation for cash flows [80000, 95000, 110000, 130000] at 9.0% WACC"
        with p4:
            if st.button("🧮 Loan EMI for $40k @ 7.5%", use_container_width=True):
                st.session_state.active_query = "Calculate monthly EMI for $40,000 principal at 7.5% interest for 4 years"

        st.markdown("<div style='margin-top: 24px; margin-bottom: 24px; border-top: 1px solid rgba(255,255,255,0.1);'></div>", unsafe_allow_html=True)

        # Render Messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    with st.expander("🔍 Telemetry & Step-by-Step Tool Reasoning", expanded=False):
                        for tc in msg["tool_calls"]:
                            st.markdown(f"**Tool:** `{tc.get('tool', 'local_engine')}`")
                            if "input" in tc:
                                st.json(tc["input"])
                            st.markdown("**Output:**")
                            st.info(tc.get("output", ""))

        # Query Input
        user_prompt = st.chat_input("Ask a financial question, valuation inquiry, or loan applicant scenario...")
        active_prompt = user_prompt or st.session_state.get("active_query")

        if active_prompt:
            st.session_state.active_query = None

            st.session_state.chat_history.append({"role": "user", "content": active_prompt})
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                with st.spinner("⚡ FinBuddy agent is executing toolchains and synthesizing financial intelligence..."):
                    agent = get_agent()
                    result = agent.run(query=active_prompt, chat_history=st.session_state.chat_history)
                    st.markdown(result["response"])

                    if result.get("tool_calls"):
                        with st.expander("🔍 Telemetry & Step-by-Step Tool Reasoning", expanded=False):
                            for tc in result["tool_calls"]:
                                st.markdown(f"**Tool:** `{tc.get('tool', 'local_engine')}`")
                                if "input" in tc:
                                    st.json(tc["input"])
                                st.markdown("**Output:**")
                                st.info(tc.get("output", ""))

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["response"],
                        "tool_calls": result.get("tool_calls", [])
                    })


# =============================================================
# TAB 2: Underwriting & SHAP Studio
# =============================================================
with tab_risk:
    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 14px;'>👥 Pre-Load Borrower Archetypes</div>", unsafe_allow_html=True)
    bp1, bp2, bp3, bp4 = st.columns(4)
    
    defaults = {"age": 34, "income": 75000.0, "emp": 5.5, "loan": 15000.0, "score": 710, "int_rate": 11.2, "dti": 0.28, "cred_hist": 8.0, "defaults": 0}

    if bp1.button("🌟 Prime Executive", use_container_width=True):
        defaults = {"age": 38, "income": 150000.0, "emp": 9.0, "loan": 20000.0, "score": 790, "int_rate": 6.8, "dti": 0.16, "cred_hist": 14.0, "defaults": 0}
    elif bp2.button("⚖️ Near-Prime Business", use_container_width=True):
        defaults = {"age": 42, "income": 68000.0, "emp": 4.0, "loan": 22000.0, "score": 660, "int_rate": 13.5, "dti": 0.38, "cred_hist": 9.0, "defaults": 0}
    elif bp3.button("🚨 High-Leverage Subprime", use_container_width=True):
        defaults = {"age": 29, "income": 34000.0, "emp": 1.5, "loan": 25000.0, "score": 530, "int_rate": 24.5, "dti": 0.65, "cred_hist": 3.0, "defaults": 2}
    elif bp4.button("🎓 Young Graduate", use_container_width=True):
        defaults = {"age": 23, "income": 58000.0, "emp": 1.0, "loan": 10000.0, "score": 690, "int_rate": 10.5, "dti": 0.22, "cred_hist": 2.0, "defaults": 0}

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    col_bento_left, col_bento_right = st.columns([1.1, 1.9], gap="large")

    with col_bento_left:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">📝 Application Parameters</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 18px;">Real-time inference inputs for the XGBoost underwriting model.</div>
            """, unsafe_allow_html=True)

            f_a1, f_a2 = st.columns(2)
            in_age = f_a1.number_input("Age", 18, 85, defaults["age"], 1)
            in_emp = f_a2.number_input("Employment (Yrs)", 0.0, 45.0, float(defaults["emp"]), 0.5)

            in_inc = f_a1.number_input("Annual Income ($)", 5000.0, 1000000.0, float(defaults["income"]), 2500.0)
            in_loan = f_a2.number_input("Requested Loan ($)", 1000.0, 250000.0, float(defaults["loan"]), 1000.0)

            in_score = st.slider("FICO Score", 350, 850, int(defaults["score"]), 5)
            in_dti = st.slider("Debt-to-Income (DTI)", 0.01, 0.90, float(defaults["dti"]), 0.01)

            f_a3, f_a4 = st.columns(2)
            in_rate = f_a3.number_input("Interest Rate (%)", 2.0, 35.0, float(defaults["int_rate"]), 0.25)
            in_hist = f_a4.number_input("Credit History (Yrs)", 1.0, 40.0, float(defaults["cred_hist"]), 0.5)

            in_defaults = st.selectbox("Past Defaults Count", [0, 1, 2, 3, 4], index=defaults["defaults"])
            
            in_lti = in_loan / max(in_inc, 1.0)
            st.caption(f"📌 **Loan-to-Income:** `{in_lti:.1%}` | **Monthly Debt:** `${(in_inc * in_dti)/12:,.0f}/mo`")

    applicant_data = {
        "person_age": in_age,
        "person_income": in_inc,
        "person_emp_length": in_emp,
        "loan_amnt": in_loan,
        "loan_int_rate": in_rate,
        "loan_percent_income": round(in_lti, 3),
        "cb_person_cred_hist_length": in_hist,
        "credit_score": in_score,
        "debt_to_income_ratio": in_dti,
        "previous_defaults_count": in_defaults,
    }

    with col_bento_right:
        try:
            explainer = get_explainer()
            res = explainer.predict_and_explain(applicant_data)
            prob = res["default_probability"]
            approval_score = res["approval_score"]
            rec = res["recommendation"]

            if not any(a["payload"] == applicant_data for a in st.session_state.session_audit):
                st.session_state.session_audit.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "payload": applicant_data,
                    "prob": prob,
                    "rec": rec,
                    "score": approval_score
                })

            with st.container(border=True):
                # Decision HUD Row with enhanced spacing
                r_d1, r_d2 = st.columns([1.2, 1.8], gap="medium")
                with r_d1:
                    if "APPROVE" in rec and "REJECT" not in rec and "REVIEW" not in rec:
                        st.markdown(f"""
                        <div class="fin-decision-approved">
                            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #34D399;">Underwriting Verdict</div>
                            <div style="font-size: 1.6rem; font-weight: 900; margin: 6px 0;">{rec}</div>
                            <div style="font-size: 0.88rem; color: #A7F3D0;">Approval Index: <b>{approval_score}/100</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif "REVIEW" in rec or "CONDITIONAL" in rec:
                        st.markdown(f"""
                        <div class="fin-decision-review">
                            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #FBBF24;">Underwriting Verdict</div>
                            <div style="font-size: 1.6rem; font-weight: 900; margin: 6px 0;">{rec}</div>
                            <div style="font-size: 0.88rem; color: #FDE68A;">Approval Index: <b>{approval_score}/100</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="fin-decision-rejected">
                            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #F43F5E;">Underwriting Verdict</div>
                            <div style="font-size: 1.6rem; font-weight: 900; margin: 6px 0;">{rec}</div>
                            <div style="font-size: 0.88rem; color: #FECDD3;">Approval Index: <b>{approval_score}/100</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                with r_d2:
                    if go is not None:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Default Probability: <b>{prob:.1%}</b><br><span style='font-size:0.8em;color:#94A3B8'>{res['risk_tier']}</span>", 'font': {'color': '#F8FAFC', 'size': 13}},
                            number={'suffix': "%", 'font': {'color': '#F8FAFC', 'size': 25}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': "#475569"},
                                'bar': {'color': "#00F2FE", 'thickness': 0.35},
                                'bgcolor': "#060A14",
                                'steps': [
                                    {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.45)"},
                                    {'range': [20, 45], 'color': "rgba(245, 158, 11, 0.45)"},
                                    {'range': [45, 70], 'color': "rgba(249, 115, 22, 0.45)"},
                                    {'range': [70, 100], 'color': "rgba(244, 63, 94, 0.55)"},
                                ],
                                'threshold': {
                                    'line': {'color': "#F43F5E", 'width': 3},
                                    'thickness': 0.8,
                                    'value': prob * 100
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                # Dual Visuals: SHAP Feature Waterfall + 5-Axis Spider Radar
                sub_tab1, sub_tab2 = st.tabs(["🔍 SHAP Feature Attribution", "🕸️ 5-Axis Financial Health Radar"])
                
                with sub_tab1:
                    shap_df = pd.DataFrame(res["feature_contributions"])
                    if px is not None:
                        shap_df["Color"] = shap_df["shap_value"].apply(lambda x: "#F43F5E" if x > 0 else "#10B981")
                        fig_shap = px.bar(
                            shap_df.sort_values(by="abs_importance", ascending=True),
                            x="shap_value",
                            y="feature",
                            orientation="h",
                            color="Color",
                            color_discrete_map="identity",
                            labels={"shap_value": "SHAP Impact on Default Log-Odds", "feature": "Applicant Feature"},
                        )
                        fig_shap.update_layout(
                            height=280,
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(8, 14, 28, 0.75)',
                            font=dict(color='#94A3B8'),
                            xaxis=dict(gridcolor='#1A253C'),
                            yaxis=dict(gridcolor='#1A253C'),
                            showlegend=False
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                with sub_tab2:
                    if go is not None:
                        categories = ['Credit Score', 'Income Strength', 'Debt Capacity', 'Employment', 'Coverage']
                        app_radar = [
                            min(100, max(0, (in_score - 350) / 5)),
                            min(100, max(0, (in_inc / 150000) * 100)),
                            min(100, max(0, (1.0 - in_dti) * 100)),
                            min(100, max(0, (in_emp / 10) * 100)),
                            min(100, max(0, (1.0 - min(1.0, in_lti)) * 100)),
                        ]
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=app_radar,
                            theta=categories,
                            fill='toself',
                            name='Applicant Profile',
                            fillcolor='rgba(0, 242, 254, 0.32)',
                            line=dict(color='#00F2FE', width=2.2)
                        ))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[85, 80, 80, 75, 85],
                            theta=categories,
                            fill='toself',
                            name='Prime Standard',
                            fillcolor='rgba(16, 185, 129, 0.15)',
                            line=dict(color='#10B981', width=1.6, dash='dot')
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1A253C', tickfont=dict(color='#64748B')),
                                angularaxis=dict(gridcolor='#1A253C', tickfont=dict(color='#F8FAFC'))
                            ),
                            height=280,
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=True,
                            legend=dict(font=dict(color='#94A3B8'))
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                with st.expander("📄 Auto-Generated Underwriting Memorandum", expanded=False):
                    st.markdown(f"**Executive Narrative:**\n> *{res['narrative_explanation']}*")
                    m_c1, m_c2 = st.columns(2)
                    with m_c1:
                        st.markdown("**Key Risk Drivers (SHAP):**")
                        for d in res["top_risk_drivers"]:
                            st.markdown(f"- 🔴 `{d['feature']}` = **{d['value']}** (SHAP: `+{d['shap_value']}`)")
                    with m_c2:
                        st.markdown("**Key Protective Strengths:**")
                        for p in res["top_protective_factors"]:
                            st.markdown(f"- 🟢 `{p['feature']}` = **{p['value']}** (SHAP: `{p['shap_value']}`)")

        except Exception as e:
            st.error(f"Credit evaluation failed: {e}")


# =============================================================
# TAB 3: Market & DCF Sandbox
# =============================================================
with tab_market:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">📈 Live Financial Intelligence & DCF Valuation Sandbox</div>
        <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 24px;">Interactive 5-year discounted cash flow enterprise modeling and technical price action.</div>
        """, unsafe_allow_html=True)

        col_m_left, col_m_right = st.columns([1, 2.5], gap="large")

        with col_m_left:
            t_symbol = st.text_input("Ticker Symbol", value="NVDA").upper()
            t_period = st.selectbox("Price Horizon", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            
            st.markdown("<div style='margin: 20px 0; border-top: 1px solid rgba(255,255,255,0.1);'></div>", unsafe_allow_html=True)
            st.markdown("###### 🧮 DCF Sensitivity Controls")
            dcf_wacc = st.slider("WACC Discount Rate (%)", 5.0, 18.0, 9.5, 0.25)
            dcf_g = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, 0.25)
            dcf_shares = st.number_input("Shares Outstanding (M)", 1.0, 20000.0, 2450.0, 50.0)
            dcf_base = st.number_input("Base FCF ($M)", 10.0, 200000.0, 25000.0, 1000.0)
            dcf_growth = st.slider("Annual FCF Growth (%)", 0.0, 50.0, 15.0, 1.0)

        # Calculate DCF
        proj_fcfs = [dcf_base * ((1 + dcf_growth / 100.0) ** i) for i in range(1, 6)]
        w_dec, g_dec = dcf_wacc / 100.0, dcf_g / 100.0
        pv_cfs = [cf / ((1 + w_dec) ** t) for t, cf in enumerate(proj_fcfs, start=1)]
        pv_exp_sum = sum(pv_cfs)
        tv_val = (proj_fcfs[-1] * (1 + g_dec)) / max((w_dec - g_dec), 0.001)
        pv_tv_val = tv_val / ((1 + w_dec) ** 5)
        ev_val = pv_exp_sum + pv_tv_val
        fair_share_price = ev_val / dcf_shares

        with col_m_right:
            if yf is not None:
                try:
                    s_obj = yf.Ticker(t_symbol)
                    s_info = s_obj.info
                    s_hist = s_obj.history(period=t_period)

                    if not s_hist.empty:
                        c_price = s_info.get("currentPrice") or s_hist["Close"].iloc[-1]
                        s_mcap = s_info.get("marketCap", 0)
                        s_pe = s_info.get("trailingPE", "N/A")
                        s_fpe = s_info.get("forwardPE", "N/A")

                        km1, km2, km3, km4 = st.columns(4)
                        km1.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Current Price</div><div class="fin-metric-value">${c_price:,.2f}</div></div>', unsafe_allow_html=True)
                        km2.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Market Cap</div><div class="fin-metric-value">${s_mcap/1e9:,.1f}B</div></div>', unsafe_allow_html=True)
                        km3.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Trailing P/E</div><div class="fin-metric-value">{s_pe if isinstance(s_pe, str) else f"{s_pe:.1f}"}</div></div>', unsafe_allow_html=True)
                        km4.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Forward P/E</div><div class="fin-metric-value">{s_fpe if isinstance(s_fpe, str) else f"{s_fpe:.1f}"}</div></div>', unsafe_allow_html=True)

                        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                        if go is not None:
                            fig_mkt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
                            fig_mkt.add_trace(go.Candlestick(
                                x=s_hist.index,
                                open=s_hist['Open'], high=s_hist['High'],
                                low=s_hist['Low'], close=s_hist['Close'],
                                name="Price"
                            ), row=1, col=1)

                            if len(s_hist) > 20:
                                s_hist['SMA20'] = s_hist['Close'].rolling(window=20).mean()
                                fig_mkt.add_trace(go.Scatter(x=s_hist.index, y=s_hist['SMA20'], line=dict(color='#F59E0B', width=1.5), name="SMA 20"), row=1, col=1)

                            fig_mkt.add_trace(go.Bar(
                                x=s_hist.index, y=s_hist['Volume'], name="Volume", marker_color='rgba(0, 242, 254, 0.45)'
                            ), row=2, col=1)

                            fig_mkt.update_layout(
                                title=f"{t_symbol} Price Trajectory & Volume Profile ({t_period})",
                                height=360,
                                margin=dict(l=10, r=10, t=35, b=10),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(8, 14, 28, 0.75)',
                                font=dict(color='#94A3B8'),
                                xaxis_rangeslider_visible=False,
                                xaxis=dict(gridcolor='#1A253C'),
                                yaxis=dict(gridcolor='#1A253C'),
                            )
                            st.plotly_chart(fig_mkt, use_container_width=True)

                    else:
                        st.info(f"No price data available for '{t_symbol}'.")

                except Exception as e:
                    st.warning(f"Market fetch notice: {e}")
            else:
                st.info("Market feed in simulated benchmark mode.")

            st.markdown("<div style='margin: 24px 0 16px 0; border-top: 1px solid rgba(255,255,255,0.1);'></div>", unsafe_allow_html=True)
            st.markdown("###### 🏢 DCF Intrinsic Valuation Output")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("PV of 5-Yr Cash Flows", f"${pv_exp_sum:,.0f}M")
            dc2.metric("PV of Terminal Value", f"${pv_tv_val:,.0f}M")
            dc3.metric("Estimated Fair Share Price", f"${fair_share_price:,.2f}")

            if 'c_price' in locals():
                spread_pct = ((fair_share_price - c_price) / c_price) * 100
                if spread_pct > 0:
                    st.success(f"🟢 **DCF Valuation Assessment:** Undervalued by **{spread_pct:.1f}%** relative to market price (${c_price:.2f}). Margin of safety is favorable.")
                else:
                    st.warning(f"🟠 **DCF Valuation Assessment:** Premium / Overvalued by **{abs(spread_pct):.1f}%** relative to market price (${c_price:.2f}).")


# =============================================================
# TAB 4: Drift Radar (MLOps)
# =============================================================
with tab_ops:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">🛡️ MLOps Observability & Concept Drift Monitoring</div>
        <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 24px;">Statistical distribution shifts tracking Population Stability Index (PSI) and Kolmogorov-Smirnov tests.</div>
        """, unsafe_allow_html=True)

        c_dr1, c_dr2 = st.columns([1, 2.5], gap="large")

        with c_dr1:
            st.markdown("###### ⚙️ Macroeconomic Stress Engine")
            stress_val = st.slider("Economic Stress Multiplier", 0.0, 1.5, 0.0, 0.1)
            batch_cnt = st.select_slider("Test Batch Size", [500, 1000, 2500, 5000], value=1000)
            run_drift_btn = st.button("⚡ Run Statistical Drift Suite", use_container_width=True)

        drift_det = get_drift_detector()
        prod_data = generate_credit_dataset(
            n_samples=batch_cnt,
            random_state=101,
            drift_factor=stress_val
        )
        d_report = drift_det.evaluate_drift(prod_data)

        with c_dr2:
            h_score = d_report["system_health_score"]
            dh1, dh2, dh3 = st.columns(3)
            dh1.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">System Health</div><div class="fin-metric-value" style="color: {"#34D399" if h_score > 80 else "#F43F5E"};">{h_score}/100</div></div>', unsafe_allow_html=True)
            dh2.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Critical Drift Features</div><div class="fin-metric-value" style="color: {"#F43F5E" if d_report["high_drift_features"] > 0 else "#34D399"};">{d_report["high_drift_features"]}</div></div>', unsafe_allow_html=True)
            dh3.markdown(f'<div class="fin-metric-badge"><div class="fin-metric-label">Moderate Drift Features</div><div class="fin-metric-value" style="color: #FBBF24;">{d_report["moderate_drift_features"]}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("###### 📋 Feature-by-Feature Statistical Drift Matrix")
            df_d = pd.DataFrame(d_report["feature_metrics"])
            st.dataframe(
                df_d[["feature", "psi_score", "ks_statistic", "status", "baseline_mean", "production_mean"]],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("<div style='margin: 28px 0 20px 0; border-top: 1px solid rgba(255,255,255,0.1);'></div>", unsafe_allow_html=True)
        st.markdown("###### 🔬 Interactive Distribution Shift Overlay")
        target_f = st.selectbox("Inspect Feature Distribution", FEATURE_NAMES, index=7)

        if px is not None:
            b_series = np.asarray(drift_det.baseline_df[target_f])
            p_series = np.asarray(prod_data[target_f])
            n_pts = min(1000, len(b_series), len(p_series))

            plot_d = pd.DataFrame({
                "Value": np.concatenate([b_series[:n_pts], p_series[:n_pts]]),
                "Dataset": ["Baseline (Training)"] * n_pts + ["Production (Current)"] * n_pts
            })

            fig_distr = px.histogram(
                plot_d,
                x="Value",
                color="Dataset",
                barmode="overlay",
                marginal="box",
                color_discrete_map={"Baseline (Training)": "#00F2FE", "Production (Current)": "#F43F5E"},
                title=f"Distribution Shift: {target_f}"
            )
            fig_distr.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8, 14, 28, 0.75)',
                font=dict(color='#94A3B8'),
                xaxis=dict(gridcolor='#1A253C'),
                yaxis=dict(gridcolor='#1A253C'),
            )
            st.plotly_chart(fig_distr, use_container_width=True)


# =============================================================
# TAB 5: Governance & Audit Log
# =============================================================
with tab_audit:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">📜 Model Governance Card & Underwriting Audit Log</div>
        <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 24px;">Traceability, validation metrics, and live applicant decision trail.</div>
        """, unsafe_allow_html=True)

        c_g1, c_g2 = st.columns([1.2, 1.8], gap="large")

        with c_g1:
            st.markdown("###### 🏷️ Model Governance Card")
            st.markdown("""
            - **Model Architecture:** Extreme Gradient Boosting (`XGBClassifier`)
            - **Tree Depth / Estimators:** `max_depth=6`, `n_estimators=300`
            - **Objective:** Logistic Loss (Binary Classification on Default)
            - **Explainability Standard:** SHAP TreeExplainer (`shap.TreeExplainer`)
            - **Data Baseline:** 25,000 synthetic underwriting records
            - **Features:** 10 core financial ratios (DTI, FICO, Income, Loan/Inc, History)
            """)

            if os.path.exists("models/model_metrics.json"):
                with open("models/model_metrics.json", "r") as f:
                    s_metrics = json.load(f)
                st.json(s_metrics)

        with c_g2:
            st.markdown("###### 📝 Live Underwriting Decision Audit Log")
            if st.session_state.session_audit:
                log_data = []
                for item in st.session_state.session_audit:
                    p = item["payload"]
                    log_data.append({
                        "Time": item["time"],
                        "Income ($)": f"${p['person_income']:,.0f}",
                        "Loan ($)": f"${p['loan_amnt']:,.0f}",
                        "Score": p['credit_score'],
                        "DTI": f"{p['debt_to_income_ratio']:.2f}",
                        "Default Prob": f"{item['prob']:.1%}",
                        "Decision": item["rec"]
                    })
                st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
            else:
                st.info("No applications scored in this session yet. Run an evaluation in the Underwriting Studio!")
