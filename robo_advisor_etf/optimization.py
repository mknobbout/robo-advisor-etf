import pandas as pd
import cvxpy as cp
import yfinance as yf
import pickle
import numpy as np
import scipy.stats as st


class ETFPortfolioOptimizer:
    def __init__(self, symbols=None):
        # Start with a basic list of very popular ETF's. The ETF's to consider can be changed to your liking.
        if symbols is None:
            symbols = [
                "SPY", "IVV", "VOO", "VTI", "QQQ", "VEA", "VTV", "BND", "IEFA", "AGG", "VUG",
                "VWO", "IJH", "IWF", "IEMG", "VIG", "IJR", "VXUS", "GLD", "VO", "BNDX", "IWM",
                "VGT", "XLK", "VYM", "IWD", "SCHD", "EFA", "ITOT", "VB", "RSP", "XLE", "XLV",
                "VCIT", "TLT", "VCSH", "MUB", "IVW", "VEU", "BSV", "SCHX", "BIL", "QUAL", "IXUS",
                "SCHF", "XLF", "JEPI", "IWB", "LQD", "VNQ", "VT", "USMV", "DIA", "VV", "IEF",
                "VTEB", "IWR", "SHY", "MBB", "IAU", "IVE", "VBR", "GOVT", "JPST", "DGRO", "VGSH",
                "SCHB", "IGSB", "IUSB", "DFAC", "TIP", "SHV", "SDY", "SPLG", "SCHG", "SPYG",
                "USFR", "VGIT", "MDY", "EEM", "DVY", "VGK", "ACWI", "XLY", "EFV", "VHT", "SGOV",
                "XLP", "TQQQ", "SPDW", "SPYV", "VMBS", "BIV", "VXF", "COWZ", "VOE", "QQQM", "XLI",
                "VTIP", "MGK"
            ]

        self._symbols = symbols
        self._information = pd.DataFrame(index=symbols)

        self._returns = pd.DataFrame()
        self._mean_returns = None
        self._cov_matrix = None

        self._portfolios = pd.DataFrame()
        self.refresh_data()

    def refresh_data(self, period="3y") -> None:
        """
        Retrieve latest finance information for the ETF tickers.
        :param period: Period to use to optimize on.
        :return: None
        """
        # Download ticker data from Yahoo Finance
        data = yf.download(list(self._information.index), period=period)

        # Returns is the percentage change of the adjusted close
        # We use adjusted close since this INCLUDES dividends. Hence we
        # do not need to capture that in our optimization objective.
        self._returns = data.pct_change().dropna()["Adj Close"]

        information = []
        for symbol in self._information.index:
            # Retrieve information from Yahoo Finance
            info = yf.Ticker(symbol).info

            # Store information for the ETF for later use
            information.append(
                {
                    "symbol": symbol,
                    "longName": info.get("longName"),
                    "category": info.get("category"),
                    "industry": info.get("industry"),
                    "beta3Year": info.get("beta3Year"),
                    "volume": info.get("volume"),
                    "marketcap": info.get("marketcap"),
                    "trailingAnnualDividendYield": info.get("trailingAnnualDividendYield"),
                    "annualReportExpenseRatio": info.get("annualReportExpenseRatio"),
                }
            )
        self._information = pd.DataFrame(information).set_index("symbol")

        # Fill important missing values with 0
        self._information["annualReportExpenseRatio"].fillna(0, inplace=True)
        self._information["trailingAnnualDividendYield"].fillna(0, inplace=True)

        # Calculate basic information
        self._mean_returns = self._returns.mean().to_numpy()
        self._cov_matrix = self._returns.cov().to_numpy()

    def optimize(self, portfolio_risk_percentage: float) -> np.array:
        """
        Uses a mean/variance optimization framework to find the optimal portfolio.
        :param portfolio_risk_percentage: Percentage of the portfolio with the highest risk to aim for.
        :return: List of weights
        """
        # Ensure that the percentage is between 0 and 1
        portfolio_risk_percentage = max(0.0, min(1.0, portfolio_risk_percentage))

        # The riskiest optimal portfolio is investing all money in the ETF with the highest return
        # We first compute the standard deviation of that portfolio
        highest_return_idx = self._mean_returns.argmax()
        highest_return_std = self._cov_matrix[highest_return_idx][highest_return_idx] ** 0.5

        # We are now going determine the maximum portfolio std based on the risk percentage
        max_portfolio_std = portfolio_risk_percentage * highest_return_std

        def expected_portfolio_return(weights):
            return cp.sum(self._mean_returns @ weights)

        def portfolio_std(weights):
            return cp.sqrt(cp.quad_form(weights, self._cov_matrix))

        # Define the optimization problem
        weights = cp.Variable(len(self._symbols))
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            portfolio_std(weights) <= max_portfolio_std
        ]

        problem = cp.Problem(cp.Maximize(expected_portfolio_return(weights)), constraints)

        #  Solve the optimization problem (note that it is qcp)
        problem.solve(qcp=True, solver=cp.CVXOPT)

        # Return portfolio weights and standard deviation
        return weights.value

    def calculate_var(self, portfolio_weights: np.array, period: str = 'daily', alpha: float = 0.01):
        """
        Method to calculate the Value of Risk of a given portfolio. Portfolio is specified as a np.array of weights.
        It uses the variance/covariance method and assumes the returns are normally distributed.
        :param portfolio_weights: The weights assigned to each Ticker.
        :param period: Period to compute the VaR over (daily/weekly/monthly/yearly)
        :param alpha: The percentile we are interested in.
        :return: The computed Value at Risk using the variance/covariance method.
        """
        period_to_trading_days = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'yearly': 252
        }
        if period not in period_to_trading_days:
            return ValueError('Period must either be "daily", "weekly", "monthly" or "yearly"')

        # Compute expected return
        portfolio_return = period_to_trading_days[period] * np.sum(self._mean_returns * portfolio_weights)

        # Compute standard deviation.
        portfolio_stddev = np.sqrt(period_to_trading_days[period] * portfolio_weights.T @ self._cov_matrix @ portfolio_weights)

        # Compute z score using the standard Guassian distribution
        z_score = st.norm.ppf(alpha)

        # Calculate the var
        var = portfolio_return + z_score * portfolio_stddev
        return var

    def create_portfolio_overview(self, portfolio_weights) -> pd.DataFrame:
        """
        Creates a pandas DataFrame that can be outputted/visualized, with an overview of the portfolio
        :param portfolio_weights:
        :return:
        """
        result = []
        for symbol, weight in zip(self._symbols, portfolio_weights):
            symbol_info = self._information.loc[symbol]
            result.append({
                "Symbol": symbol,
                "Full Name": symbol_info["longName"],
                "Weight": weight,
                "Category": symbol_info["category"],
                "Industry": symbol_info["industry"],
                "Expense Ratio": symbol_info["annualReportExpenseRatio"],
            })

        return pd.DataFrame(result).sort_values("Weight", ascending=False)

    @classmethod
    def load(cls, path: str) -> "ETFPortfolioOptimizer":
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

