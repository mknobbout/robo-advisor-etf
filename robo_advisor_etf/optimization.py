import pandas as pd
import cvxpy as cp
import requests
import yfinance as yf
import pickle
import numpy as np
import scipy.stats as st
from typing import Optional, Dict
from bs4 import BeautifulSoup
from enum import Enum


class TradingPeriod(Enum):
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    YEARLY = 252


class ETFPortfolioOptimizer:
    def __init__(self, etf_data: pd.DataFrame):
        if "symbol" not in etf_data.columns:
            raise ValueError('Required column "symbol" not in DataFrame.')

        self._etf_data = etf_data

        self._returns = None
        self._mean_returns = None
        self._cov_matrix = None

        self.refresh_data()

    def refresh_data(self, period="3y") -> None:
        """
        Retrieve latest finance information for the ETF tickers.
        :param period: Period to use to optimize on.
        :return: None
        """
        # Download ticker data from Yahoo Finance
        data = yf.download(list(self._etf_data.symbol), period=period)

        # Returns is the percentage change of the adjusted close
        # We use adjusted close since this INCLUDES dividends. Hence we
        # do not need to capture that in our optimization objective.
        self._returns = data.pct_change().dropna()["Adj Close"]

        # Define columns to retrieve
        data_to_retrieve = [
            "longName",
            "category",
            "industry",
            "beta3Year",
            "volume",
            "marketcap",
        ]

        result = {column: [] for column in data_to_retrieve}
        for symbol in self._etf_data.symbol:
            # Retrieve information from Yahoo Finance
            info = yf.Ticker(symbol).info

            # Store information for the ETF for later use
            for column in data_to_retrieve:
                result[column].append(info.get(column))

        # Append data to dataset
        for column in data_to_retrieve:
            self._etf_data[column] = result[column]

        # Calculate basic information
        self._mean_returns = self._returns.mean().to_numpy()
        self._cov_matrix = self._returns.cov().to_numpy()

    @staticmethod
    def get_expense_ratio(ticker_symbol: str) -> float:
        """
        Static method to download the expense ratio of a ticker. Unfortunately, we have to do this manually
        since yfinance does not allow it.
        :param ticker_symbol: The ticker symbol
        :return: The (yearly) expense ratio
        """
        # Construct the URL for the Yahoo Finance page of the fund
        url = f"https://finance.yahoo.com/quote/{ticker_symbol}?p={ticker_symbol}"

        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the element that contains the expense ratio
            expense_ratio_elem = soup.find("td", {"data-test": "EXPENSE_RATIO-value"})

            # Check if the expense ratio element was found
            if expense_ratio_elem:
                # Extract the text (expense ratio) from the element
                expense_ratio = expense_ratio_elem.text.strip()

                # Convert to float and divide by 100
                return float(expense_ratio.strip("%")) / 100

    def optimize(
        self,
        portfolio_risk_percentage: float,
        max_percentage_per_etf: float = 1.0,
        leq_constraints: Optional[Dict[str, float]] = None,
        geq_constraints: Optional[Dict[str, float]] = None,
    ) -> np.array:
        """
        Uses a mean/variance optimization framework to find the optimal portfolio.
        :param portfolio_risk_percentage: Percentage of the portfolio with the highest risk to aim for.
        :param max_percentage_per_etf: Maximum percentage allocated to 1 ETF
        :param leq_constraints: LEQ constraints
        :param geq_constraints: GEQ constraints
        :return: List of weights
        """
        # Ensure that the percentage is between 0 and 1
        portfolio_risk_percentage = max(0.0, min(1.0, portfolio_risk_percentage))

        # Ensure leq_constraints and geq_constraints are initiated as dicts
        if leq_constraints is None:
            leq_constraints = {}
        if geq_constraints is None:
            geq_constraints = {}

        # The riskiest optimal portfolio is investing all money in the ETF with the highest return
        # We first compute the standard deviation of that portfolio
        highest_return_idx = self._mean_returns.argmax()
        highest_return_std = (
            self._cov_matrix[highest_return_idx][highest_return_idx] ** 0.5
        )

        # We are now going determine the maximum portfolio std based on the risk percentage
        max_portfolio_std = portfolio_risk_percentage * highest_return_std

        # Assert that our covariance matrix is PSD
        cov_psd = cp.psd_wrap(self._cov_matrix)

        def expected_portfolio_return(weights):
            """
            Main objective function. Note that since we take the adjusted closing price, we simply optimize
            on this statistic. We do not have to take into account the expense ratio, since this is already
            baked into the buying price of the ETF.
            :param weights: Current weights
            :return: Expected return
            """
            return cp.sum(self._mean_returns @ weights)

        def portfolio_std(weights):
            return cp.sqrt(cp.quad_form(weights, cov_psd))

        # Define the optimization problem
        weights = cp.Variable(len(self._etf_data))
        constraints = [
            cp.sum(weights) == 1,
            weights <= max_percentage_per_etf,
            weights >= 0,
            portfolio_std(weights) <= max_portfolio_std,
        ]
        # Add leq constraints
        constraints += [
            cp.sum(self._etf_data[leq_feature].to_numpy() @ weights) <= leq_value
            for leq_feature, leq_value in leq_constraints.items()
        ]
        # Add geq constraints
        constraints += [
            cp.sum(self._etf_data[geq_feature].to_numpy() @ weights) >= geq_value
            for geq_feature, geq_value in geq_constraints.items()
        ]

        problem = cp.Problem(
            cp.Maximize(expected_portfolio_return(weights)), constraints
        )

        #  Solve the optimization problem (note that it is qcp)
        problem.solve(qcp=True, solver=cp.CVXOPT)

        # Return portfolio weights and standard deviation
        return weights.value

    def calculate_var(
        self,
        portfolio_weights: np.array,
        period: TradingPeriod = TradingPeriod.DAILY,
        alpha: float = 0.01,
    ):
        """
        Method to calculate the Value of Risk of a given portfolio. Portfolio is specified as a np.array of weights.
        It uses the variance/covariance method and assumes the returns are normally distributed.
        :param portfolio_weights: The weights assigned to each Ticker.
        :param period: Period to compute the VaR over
        :param alpha: The percentile we are interested in.
        :return: The computed Value at Risk using the variance/covariance method.
        """
        # Compute expected return
        portfolio_return = period.value * np.sum(self._mean_returns * portfolio_weights)

        # Compute standard deviation.
        portfolio_stddev = np.sqrt(
            period.value * portfolio_weights.T @ self._cov_matrix @ portfolio_weights
        )

        # Compute z score using the standard Guassian distribution
        z_score = st.norm.ppf(alpha)

        # Calculate the var
        var = portfolio_return + z_score * portfolio_stddev
        return var

    def get_statistics(self, portfolio_weights) -> pd.DataFrame:
        """
        Method for computing some basic portfolio statistics based on the weights.
        :param portfolio_weights: The weights
        :return: Several statistics (beta, VaRs and returns)
        """
        return pd.DataFrame(
            [
                {
                    "description": "3-year Beta",
                    "value": np.sum(
                        self._etf_data["beta3Year"].to_numpy() @ portfolio_weights
                    ),
                },
                {
                    "description": "Monthly value at risk (1%)",
                    "value": self.calculate_var(
                        portfolio_weights, period=TradingPeriod.MONTHLY, alpha=0.01
                    ),
                },
                {
                    "description": "Yearly value at risk (2%)",
                    "value": self.calculate_var(
                        portfolio_weights, period=TradingPeriod.YEARLY, alpha=0.01
                    ),
                },
                {
                    "description": "Monthly expected return",
                    "value": TradingPeriod.MONTHLY.value
                    * np.sum(self._mean_returns @ portfolio_weights),
                },
                {
                    "description": "Yearly expected return",
                    "value": TradingPeriod.YEARLY.value
                    * np.sum(self._mean_returns @ portfolio_weights),
                },
            ]
        )

    def create_portfolio_overview(self, portfolio_weights) -> pd.DataFrame:
        """
        Creates a pandas DataFrame that can be outputted/visualized, with an overview of the portfolio
        :param portfolio_weights:
        :return:
        """
        result = self._etf_data.copy()
        result["weight"] = portfolio_weights
        result["return"] = self._mean_returns

        return pd.DataFrame(result).sort_values("weight", ascending=False)

    @classmethod
    def load(cls, path: str) -> "ETFPortfolioOptimizer":
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
