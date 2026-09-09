"""FinBuddy Multi-Tool Financial Research Agent with LangChain & LangSmith Tracing."""

import os
import re
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.tools.credit_risk_tool import CreditRiskTool
from src.tools.market_data_tool import MarketDataTool
from src.tools.calculator_tool import FinancialCalculatorTool


SYSTEM_PROMPT = """You are FinBuddy, an elite AI Financial Research & Underwriting Assistant.
You assist financial analysts, portfolio managers, and loan underwriters with:
1. Credit Risk Underwriting: Running the trained XGBoost + SHAP model to evaluate default risks and explain factors.
2. Market Research: Pulling live stock metrics, valuation multiples, and recent news.
3. Financial Mathematics: Computing DCF valuations, loan EMIs, and Sharpe ratios.

Guidelines:
- Provide clear, professional, structured financial insights.
- When evaluating credit risk or stock valuations, cite specific figures and risk drivers.
- Always include actionable advice and risk disclosures.
"""


class FinBuddyAgent:
    """Orchestrates multi-tool execution and LLM reasoning."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tools = [
            CreditRiskTool(),
            MarketDataTool(),
            FinancialCalculatorTool(),
        ]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the LLM provider based on available environment credentials."""
        openai_key = os.getenv("OPENAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        if openai_key and not openai_key.startswith("your_"):
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model=self.model_name, temperature=0.1)
            except Exception as e:
                print(f"Warning: Could not initialize ChatOpenAI: {e}")

        if google_key and not google_key.startswith("your_"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
            except Exception as e:
                print(f"Warning: Could not initialize ChatGoogleGenerativeAI: {e}")

        return None

    def _rule_based_router(self, query: str) -> Optional[str]:
        """
        Intelligent deterministic router when running offline or without API keys.
        Parses intent and delegates to corresponding ML / Market / Calculator tools.
        """
        q = query.lower()

        # 1. Market Data Intent
        ticker_match = re.search(r'\b(aapl|msft|nvda|tsla|googl|amzn|meta|jpm|gs|spy|qqq)\b', q)
        if any(w in q for w in ["stock", "ticker", "price", "valuation", "shares", "market cap"]) and ticker_match:
            ticker = ticker_match.group(1).upper()
            tool = self.tool_map["market_data_tool"]
            return tool._run(ticker=ticker)

        # 2. Credit Risk Assessment Intent
        if any(w in q for w in ["credit", "loan risk", "underwrite", "default", "borrower", "fico", "score applicant"]):
            tool = self.tool_map["credit_risk_assessment_tool"]
            
            income = 65000.0
            loan = 15000.0
            score = 680
            dti = 0.30
            age = 32
            
            inc_m = re.search(r'income(?:\s+of|\s*[:=])?\s*\$?([0-9,]+)', q)
            if inc_m:
                income = float(inc_m.group(1).replace(",", ""))
                
            loan_m = re.search(r'loan(?:\s+of|\s*[:=])?\s*\$?([0-9,]+)', q)
            if loan_m:
                loan = float(loan_m.group(1).replace(",", ""))
                
            score_m = re.search(r'(?:credit score|fico|score)(?:\s+of|\s*[:=])?\s*([0-9]{3})', q)
            if score_m:
                score = int(score_m.group(1))

            dti_m = re.search(r'dti(?:\s+of|\s*[:=])?\s*([0-9.]+)', q)
            if dti_m:
                val = float(dti_m.group(1))
                dti = val / 100.0 if val > 1.0 else val

            return tool._run(
                person_age=age,
                person_income=income,
                person_emp_length=4.5,
                loan_amnt=loan,
                loan_int_rate=12.5,
                cb_person_cred_hist_length=6.0,
                credit_score=score,
                debt_to_income_ratio=dti,
                previous_defaults_count=0
            )

        # 3. Financial Calculator Intent (EMI)
        if any(w in q for w in ["emi", "monthly payment", "mortgage", "amortization"]):
            tool = self.tool_map["financial_calculator_tool"]
            return tool._run(
                calculation_type="loan_emi",
                principal=25000,
                annual_interest_rate=8.5,
                tenure_years=5
            )

        # 4. DCF Valuation Intent
        if any(w in q for w in ["dcf", "discounted cash flow", "intrinsic value", "wacc"]):
            tool = self.tool_map["financial_calculator_tool"]
            return tool._run(
                calculation_type="dcf_valuation",
                free_cash_flows=[50000, 60000, 75000, 90000, 110000],
                discount_rate_wacc=9.5,
                terminal_growth_rate=2.5,
                shares_outstanding=10000
            )

        return None

    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Executes user query through LLM with tool binding or offline reasoning router.
        """
        if self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

                llm_with_tools = self.llm.bind_tools(self.tools)
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                
                if chat_history:
                    for msg in chat_history:
                        if msg.get("role") == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg.get("role") == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                            
                messages.append(HumanMessage(content=query))
                
                ai_msg = llm_with_tools.invoke(messages)
                
                tool_calls_executed = []
                if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        if tool_name in self.tool_map:
                            res = self.tool_map[tool_name].invoke(tool_args)
                            tool_calls_executed.append({
                                "tool": tool_name,
                                "input": tool_args,
                                "output": res
                            })
                            
                    summary_prompt = (
                        f"{SYSTEM_PROMPT}\n\nUser Question: {query}\n\n"
                        f"Tool Execution Outputs:\n"
                        + "\n\n".join([f"[{t['tool']}]: {t['output']}" for t in tool_calls_executed])
                        + "\n\nPlease provide a clear, synthesized final response for the financial analyst."
                    )
                    final_response = self.llm.invoke([HumanMessage(content=summary_prompt)]).content
                else:
                    final_response = ai_msg.content

                return {
                    "response": final_response,
                    "tool_calls": tool_calls_executed,
                    "model_used": self.model_name
                }
            except Exception as e:
                print(f"LLM invocation failed: {e}. Falling back to rule-based agent.")

        # Fallback Router Execution
        routed_tool_output = self._rule_based_router(query)
        if routed_tool_output:
            response = (
                f"Here is the financial analysis generated by the **FinBuddy Tool Engine**:\n\n"
                f"{routed_tool_output}\n\n"
                f"*Note: Running with direct local tool execution engine.*"
            )
            return {
                "response": response,
                "tool_calls": [{"tool": "local_router", "output": routed_tool_output}],
                "model_used": "FinBuddy-Local-Engine"
            }

        return {
            "response": (
                "Hello! I am **FinBuddy**, your AI Financial Research & Underwriting Assistant.\n\n"
                "Here is how I can assist you:\n"
                "- **Credit Risk & Underwriting**: Evaluate loan default probabilities and inspect SHAP factors (e.g., *'Evaluate credit risk for income $75,000, loan $20,000, credit score 680'*).\n"
                "- **Live Market Intelligence**: Query real-time quotes, valuation metrics, and news (e.g., *'Analyze NVDA stock metrics'*).\n"
                "- **Financial Calculations**: Calculate Loan EMI amortization, DCF enterprise valuation, or Sharpe ratios (e.g., *'Calculate EMI for $50,000 at 8.5% for 5 years'*)."
            ),
            "tool_calls": [],
            "model_used": "FinBuddy-Local-Engine"
        }
