import numpy as np
import pandas as pd


# Estimating AR Persistence and Intercept constants
def fit_ar_coef(df: pd.DataFrame, h: int) -> pd.DataFrame:
    # Inputs:
    # df: rows = month-end dates, columns = beta1, beta2, beta3
    # h: forecast horizon in months (e.g. 12 pairs Jan 2026 with Jan 2025)


    # Outputs:
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



# Forecasting all three betas for each of the next h months from one base month
# Assumption: the history data frame's last row is the base month we are using to forecast
def forecast_betas(history: pd.DataFrame, h: int) -> pd.DataFrame:
    # Inputs:
    # history: betas up to and including the base month
    #          (rows = month-end dates, columns = est beta1, est beta2, est beta3)
    #          the last row is the base month ("today")
    # h: number of months ahead to forecast (e.g. h=12 then forecasts for base + 1 ... base + 12)
    # min_months: minimum rows of history needed to fit the AR(1)

    # Returns:
    # table with one row per target month (base month + j), columns = beta1, beta2, beta3
    min_months = h + 1  # minimum months of history needed to fit the AR(1)


    # Test: check if there is enough history to estimate c and gamma
    if len(history) < min_months:
        raise ValueError(f"Not enough historical data, require at least {min_months}")

    base_date = history.index[-1]             # the base month-end date (last row of history)
    base_beta = history.iloc[-1]              # the base month's betas (pull the last row values)

    # The empty Dataframe to store the forecated betas for each of the next h months
    # columns = beta1, beta2, beta3; rows = base month + 1, ... base month + h
    beta_foreclist = pd.DataFrame(columns=history.columns, dtype=float)

    # Loop for h months ahead
    for j in range(1, h + 1):                 
        # Esimating the AR coefficients (c and gamma) based on the forecast length
        # Returns a table with rows = ["c", "gamma"] and columns = beta1, beta2, beta3
        coef_est = fit_ar_coef(history, j)

        beta_forec = [] #temp list to store month j forecasted beta
        # Loop through the three betas
        for i in range(3):                   
            c = coef_est.iloc[0, i] #pulling the estimated c constant for beta i
            gamma = coef_est.iloc[1, i] #pulling the estimated gamma constant for beta i

            # Using AR model: beta(t+j) = c + gamma * beta(t)
            # calculate beta(t+j) for beta i and adding to temp list beta_forec
            beta_forec.append(c + gamma * base_beta.iloc[i])

        # label the row with the target month: j month-ends after the base month
        target_date = base_date + pd.offsets.MonthEnd(j)
        beta_foreclist.loc[target_date] = beta_forec #adding the row to the forecasted beta table

    return beta_foreclist

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

