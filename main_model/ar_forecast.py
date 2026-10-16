import numpy as np
import pandas as pd


# Estimating AR Persistence and Intercept constants
def fit_ar_coef(df: pd.DataFrame, h: int) -> pd.DataFrame:
    # Args:
    # df: rows = month-end dates, columns = beta1, beta2, beta3
    # h: forecast horizon in months (e.g. 12 pairs Jan 2026 with Jan 2025)


    # Returns:
    # table with rows ["c", "gamma"] and one column per beta
    #  c = the constant 
    #  gamma = the coefficient constant in front of lagged period observation
    results_df = pd.DataFrame(index=["c", "gamma"])

    for i in range(len(df.columns)):          # i = 0, 1, 2 -> beta1, beta2, beta3
        beta = df.iloc[:, i]                  # column for beta i (from input df)

        # build the paired table: beta at month j (x) and beta h months later (y)
        #  ex. Jan 2025 and Jan 2026
        pairs = []
        for j in range(len(df) - h):
            x = beta.iloc[j] # calling month j row
            y = beta.iloc[j + h] # calling month j + h row
            if not (np.isnan(x) or np.isnan(y)):   # skip pairs with a missing value
                pairs.append([x, y])
        paired = pd.DataFrame(pairs, columns=["x", "y"])

        # OLS of y on [1, x] gives intercept c and persistence gamma
        X = np.column_stack([np.ones(len(paired)), paired["x"]]) # x is the month j column
        (temp_c, temp_gamma), *_ = np.linalg.lstsq(X, paired["y"], rcond=None) # OLS estimate

        # add a new column for beta i with its estimated coefficients
        results_df[df.columns[i]] = [temp_c, temp_gamma]

    return results_df


# Table to store forecasted betas (one row per forecast date)
Forec_beta = pd.DataFrame(columns=["beta1", "beta2", "beta3"])


# Producing the Forecasted Beta (single value) (for test purpose)
def forecast_beta(coef_est: pd.DataFrame, beta: float, num: int) -> float:
    # Args:
    # coef_est: output of fit_ar_coef (rows = c, gamma; columns = beta1, beta2, beta3)
    # beta: this month's value of the beta
    # num: which beta this is (1, 2, 3)

    # Returns:
    # forecasted beta h months ahead (h is whatever coef_est was fit with)

    # extract the coefficients for beta num (column num - 1, since columns start at 0)
    c = coef_est.iloc[0, num - 1]
    gamma = coef_est.iloc[1, num - 1]

    # AR model: beta(t+h) = c + gamma * beta(t)
    return c + gamma * beta


# Forecasting all three betas for one month and adding them to the forecast table
def forecast_betalist(coef_est: pd.DataFrame, beta_list: list, forec_beta: pd.DataFrame,
                      date) -> pd.DataFrame:
    # Args:
    # coef_est: output of fit_ar_coef (rows = c, gamma; columns = beta1, beta2, beta3)
    # beta_list: this month's betas [beta1, beta2, beta3]
    # forec_beta: table to add the forecasts to (e.g. Forec_beta)
    # date: row label for the forecasts (e.g. the target month)

    # Returns:
    # forec_beta with a new row of forecasted betas
    forecasts = []
    for i in range(3):                        # i = 0, 1, 2 -> beta1, beta2, beta3
        forecasts.append(forecast_beta(coef_est, beta_list[i], i + 1))

    forec_beta.loc[date] = forecasts
    return forec_beta


# Forecasted yield from the Nelson-Siegel formula
def forecast_yield(forec_beta: pd.DataFrame, date, tau: float, lam: float = 0.0609) -> float:
    # Args:
    # forec_beta: table of forecasted betas (columns = beta1, beta2, beta3)
    # date: which forecast row to use
    # tau: maturity in months (e.g. 120 for 10 years)
    # lam: lambda, fixed at 0.0609 as in Diebold-Li

    # Returns:
    # forecasted yield at maturity tau
    b1, b2, b3 = forec_beta.loc[date, ["beta1", "beta2", "beta3"]]

    slope_load = (1 - np.exp(-lam * tau)) / (lam * tau)
    curve_load = slope_load - np.exp(-lam * tau)

    # y(tau) = beta1 + beta2 * slope loading + beta3 * curvature loading
    return b1 + b2 * slope_load + b3 * curve_load

