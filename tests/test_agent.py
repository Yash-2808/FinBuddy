"""Unit tests for FinBuddy Agent and tools."""

import unittest
from src.tools.credit_risk_tool import CreditRiskTool
from src.tools.market_data_tool import MarketDataTool
from src.tools.calculator_tool import FinancialCalculatorTool
from src.agents.financial_agent import FinBuddyAgent


class TestAgentToolsAndOrchestrator(unittest.TestCase):

    def test_credit_risk_tool_execution(self):
        """Verify CreditRiskTool produces structured markdown reports."""
        tool = CreditRiskTool()
        output = tool._run(
            person_age=35,
            person_income=90000.0,
            person_emp_length=7.0,
            loan_amnt=15000.0,
            loan_int_rate=9.0,
            cb_person_cred_hist_length=10.0,
            credit_score=740,
            debt_to_income_ratio=0.22,
            previous_defaults_count=0,
        )
        self.assertIn("Credit Risk & Underwriting Assessment", output)
        self.assertIn("Default Probability", output)
        self.assertIn("Recommendation", output)

    def test_calculator_tool_emi(self):
        """Verify loan EMI calculation math."""
        tool = FinancialCalculatorTool()
        output = tool._run(
            calculation_type="loan_emi",
            principal=10000,
            annual_interest_rate=12.0,
            tenure_years=1
        )
        self.assertIn("Loan EMI & Amortization Result", output)
        self.assertTrue("888" in output or "$888.49" in output)

    def test_calculator_tool_dcf(self):
        """Verify DCF valuation tool calculation."""
        tool = FinancialCalculatorTool()
        output = tool._run(
            calculation_type="dcf_valuation",
            free_cash_flows=[100000, 110000, 120000],
            discount_rate_wacc=10.0,
            terminal_growth_rate=2.5,
            shares_outstanding=10000
        )
        self.assertIn("DCF Valuation Model Result", output)
        self.assertIn("Estimated Enterprise / Equity Value", output)

    def test_agent_orchestrator_fallback(self):
        """Verify FinBuddyAgent returns structured response in offline mode."""
        agent = FinBuddyAgent()
        res = agent.run("Evaluate credit risk for income $80,000, loan $15,000, credit score 720")
        self.assertIn("response", res)
        self.assertGreater(len(res["response"]), 0)
        self.assertIn("model_used", res)


if __name__ == "__main__":
    unittest.main()
