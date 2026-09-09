"""LangChain Tool for Financial Mathematics, DCF Valuation, and Loan Calculations."""

import json
from typing import Optional, Type, List, Any
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        def invoke(self, input_dict: Any) -> Any:
            if isinstance(input_dict, dict):
                return self._run(**input_dict)
            return self._run(input_dict)


class FinancialCalculatorInput(BaseModel):
    calculation_type: str = Field(
        ...,
        description="Type of calculation: 'loan_emi', 'dcf_valuation', 'sharpe_ratio', or 'compound_interest'",
    )
    # Parameters for loan_emi
    principal: Optional[float] = Field(None, description="Principal loan amount in USD")
    annual_interest_rate: Optional[float] = Field(None, description="Annual interest rate percentage (e.g. 7.5)")
    tenure_years: Optional[int] = Field(None, description="Tenure of the loan in years (e.g. 5)")
    
    # Parameters for dcf_valuation
    free_cash_flows: Optional[List[float]] = Field(
        None, description="List of forecasted Free Cash Flows (USD) for next N years, e.g. [100000, 110000, 125000]"
    )
    discount_rate_wacc: Optional[float] = Field(None, description="Discount rate / WACC percentage (e.g. 9.0)")
    terminal_growth_rate: Optional[float] = Field(2.5, description="Perpetual terminal growth rate percentage (e.g. 2.5)")
    shares_outstanding: Optional[float] = Field(None, description="Total shares outstanding count for per-share price")

    # Parameters for sharpe_ratio
    portfolio_return: Optional[float] = Field(None, description="Expected annual portfolio return percentage (e.g. 14.0)")
    risk_free_rate: Optional[float] = Field(4.5, description="Risk-free rate percentage (e.g. 4.5)")
    portfolio_std_dev: Optional[float] = Field(None, description="Annual portfolio volatility / standard deviation (e.g. 18.0)")


class FinancialCalculatorTool(BaseTool):
    name: str = "financial_calculator_tool"
    description: str = (
        "Calculates accurate financial metrics including Loan EMI & amortization, "
        "Discounted Cash Flow (DCF) enterprise valuation, Sharpe ratio, and compound interest."
    )
    args_schema: Type[BaseModel] = FinancialCalculatorInput

    def _run(
        self,
        calculation_type: str,
        principal: Optional[float] = None,
        annual_interest_rate: Optional[float] = None,
        tenure_years: Optional[int] = None,
        free_cash_flows: Optional[List[float]] = None,
        discount_rate_wacc: Optional[float] = None,
        terminal_growth_rate: Optional[float] = 2.5,
        shares_outstanding: Optional[float] = None,
        portfolio_return: Optional[float] = None,
        risk_free_rate: Optional[float] = 4.5,
        portfolio_std_dev: Optional[float] = None,
    ) -> str:
        calc = calculation_type.lower().strip()

        try:
            if "emi" in calc or "loan" in calc:
                if not (principal and annual_interest_rate and tenure_years):
                    return "Error: 'loan_emi' requires `principal`, `annual_interest_rate`, and `tenure_years`."
                
                r = (annual_interest_rate / 100.0) / 12.0
                n = tenure_years * 12
                if r == 0:
                    emi = principal / n
                else:
                    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
                
                total_payment = emi * n
                total_interest = total_payment - principal

                return (
                    f"### 🧮 Loan EMI & Amortization Result\n\n"
                    f"- **Principal Loan**: ${principal:,.2f}\n"
                    f"- **Interest Rate**: {annual_interest_rate:.2f}% APR\n"
                    f"- **Loan Tenure**: {tenure_years} years ({n} months)\n"
                    f"- **Monthly EMI**: **${emi:,.2f}**\n"
                    f"- **Total Interest Payable**: ${total_interest:,.2f}\n"
                    f"- **Total Repayment Amount**: ${total_payment:,.2f}"
                )

            elif "dcf" in calc or "valuation" in calc:
                if not (free_cash_flows and discount_rate_wacc):
                    return "Error: 'dcf_valuation' requires `free_cash_flows` and `discount_rate_wacc`."

                wacc = discount_rate_wacc / 100.0
                g = (terminal_growth_rate or 2.5) / 100.0

                if wacc <= g:
                    return f"Error: Discount rate ({discount_rate_wacc}%) must be higher than terminal growth rate ({terminal_growth_rate}%)."

                # PV of explicit forecast period
                pv_fcfs = []
                for t, fcf in enumerate(free_cash_flows, start=1):
                    pv = fcf / ((1 + wacc) ** t)
                    pv_fcfs.append(pv)

                pv_explicit = sum(pv_fcfs)
                final_fcf = free_cash_flows[-1]
                terminal_value = (final_fcf * (1 + g)) / (wacc - g)
                pv_terminal = terminal_value / ((1 + wacc) ** len(free_cash_flows))
                enterprise_value = pv_explicit + pv_terminal

                res = (
                    f"### 🏢 DCF Valuation Model Result\n\n"
                    f"- **Discount Rate (WACC)**: {discount_rate_wacc:.2f}%\n"
                    f"- **Terminal Growth Rate**: {terminal_growth_rate:.2f}%\n"
                    f"- **PV of Forecasted Cash Flows**: ${pv_explicit:,.2f}\n"
                    f"- **PV of Terminal Value**: ${pv_terminal:,.2f}\n"
                    f"- **Estimated Enterprise / Equity Value**: **${enterprise_value:,.2f}**\n"
                )

                if shares_outstanding and shares_outstanding > 0:
                    fair_share_price = enterprise_value / shares_outstanding
                    res += f"- **Estimated Fair Intrinsic Value Per Share**: **${fair_share_price:,.2f}**\n"

                return res

            elif "sharpe" in calc:
                if portfolio_return is None or portfolio_std_dev is None:
                    return "Error: 'sharpe_ratio' requires `portfolio_return` and `portfolio_std_dev`."

                rf = risk_free_rate or 4.5
                sharpe = (portfolio_return - rf) / max(portfolio_std_dev, 0.0001)

                return (
                    f"### 📊 Sharpe Ratio Analysis\n\n"
                    f"- **Expected Return**: {portfolio_return:.2f}%\n"
                    f"- **Risk-Free Rate**: {rf:.2f}%\n"
                    f"- **Portfolio Volatility (Std Dev)**: {portfolio_std_dev:.2f}%\n"
                    f"- **Sharpe Ratio**: **{sharpe:.2f}**\n"
                    f"- **Assessment**: "
                    + ("Excellent (> 2.0)" if sharpe >= 2.0 else "Good (1.0 - 2.0)" if sharpe >= 1.0 else "Sub-optimal (< 1.0)")
                )

            else:
                return f"Unsupported calculation type '{calculation_type}'. Supported: 'loan_emi', 'dcf_valuation', 'sharpe_ratio'."

        except Exception as e:
            return f"Error executing calculation: {str(e)}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
