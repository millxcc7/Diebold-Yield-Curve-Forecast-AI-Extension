import numpy as np
import pandas as pd

# Fixed lambda used in the Diebold-Li paper
LAMBDA = 0.0609

# The 8 Treasury maturities we use, in months
MATURITIES = np.array([3, 6, 12, 24, 36, 60, 84, 120])

# Convert maturities into strings to match CSV column names
COLUMNS = [str(m) for m in MATURITIES]


# 1. Process yield data:

def load_yields():

    # Read monthly yield data already prepared by pull_yields.py
    yields = pd.read_csv("data/treasury_yields_1996_2026.csv", index_col="date", parse_dates=True)


    yields = yields[COLUMNS]

    # Remove months with missing observations
    yields = yields.dropna()

    return yields


# 2. Nelson-Siegel factor loadings

def ns_loadings(maturities, lam):

    maturities = np.asarray(maturities, dtype=float)

    # Calculate lambda multiplied by each maturity
    x = lam * maturities

    # Level loading: always 1 for every maturity
    level = np.ones(len(maturities))

    # Slope loading:
    # (1 - exp(-lambda * maturity)) / (lambda * maturity)
    # expm1 improves numerical accuracy for small values
    slope = -np.expm1(-x) / x

    # Curvature loading:
    # Slope loading minus exp(-lambda * maturity)
    curvature = slope - np.exp(-x)

    # Combine the three loadings into an 8 x 3 matrix
    # Each row represents one maturity
    # Each column represents one factor loading
    X = np.column_stack([level, slope, curvature])

    return X


# 3. Betas estimation (OLS)


def estimate_betas(yields):

    # Construct the Nelson-Siegel factor loading matrix
    # X is 8 x 3
    X = ns_loadings(MATURITIES, LAMBDA)

    # Create lists to store monthly results
    beta_results = []
    fitted_results = []

    # Loop through each month of observed Treasury yields
    for date, row in yields.iterrows():

        y = row.to_numpy(dtype=float)

        # Estimate beta1, beta2, beta3 using OLS
        # Minimize the sum of squared differences between
        # actual yields (y) and fitted yields (X @ betas)
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Reconstruct the 8 fitted yields from the 3 betas
        fitted_yields = X @ betas

        # Save this month's 3 estimated betas
        beta_results.append(betas)

        # Save this month's 8 fitted yields
        fitted_results.append(fitted_yields)

    # Convert estimated betas into a DataFrame
    # Rows = months, columns = beta1, beta2, beta3
    factors = pd.DataFrame(
        beta_results,
        index=yields.index,
        columns=["beta1", "beta2", "beta3"]
    )

    # Convert fitted yields into a DataFrame
    # Rows = months, columns = 8 maturities
    fitted = pd.DataFrame(
        fitted_results,
        index=yields.index,
        columns=COLUMNS
    )

    return factors, fitted


# 4. Calculate fitting errors & RMSE

def calculate_errors(actual, fitted):

    # Calculate fitting errors:
    # Error = Actual Yield - Fitted Yield
    errors = actual - fitted

    # FRED yields are quoted in percentage points
    # Multiply by 100 to convert errors into basis points
    errors_bps = errors * 100

    # Square each error, average across the 8 maturities,
    # then take the square root to obtain monthly RMSE
    rmse = np.sqrt((errors_bps ** 2).mean(axis=1))

    return errors, rmse


# 5. Pipeline & Save Results

def main():

    # Load monthly Treasury yields data
    yields = load_yields()

    # Estimate the 3 betas and reconstruct fitted yields
    factors, fitted = estimate_betas(yields)

    # Calculate fitting errors and monthly RMSE
    errors, rmse = calculate_errors(yields, fitted)

    # Create a diagnostics table with monthly fitting RMSE
    diagnostics = pd.DataFrame({"observation_date": observation_dates.strftime("%Y-%m-%d"),"rmse_bps": rmse}, index=yields.index)

    # Save monthly beta estimates for the AR(1) model
    factors.to_csv("data/ns_factors.csv")

    # Save observed and fitted yields for comparison
    yields.to_csv("data/ns_observed_yields.csv")
    fitted.to_csv("data/ns_fitted_yields.csv")

    # Save fitting errors and monthly RMSE
    errors.to_csv("data/ns_residuals.csv")
    diagnostics.to_csv("data/ns_diagnostics.csv")

    # Check for missing months in our historical dataset
    expected = pd.period_range("1996-01", "2026-09", freq="M") 
    actual = factors.index.to_period("M")
    missing = expected.difference(actual)

    # Display the first 5 estimated factor observations
    print("First 5 estimated betas:")
    print(factors.head())

    # Display average fitting RMSE across all months
    print("\nAverage fitting RMSE (bps):", rmse.mean())

    # Display any missing months
    print("\nMissing months:", list(missing))

    # Check for missing months before AR forecasting
    if len(missing) > 0:
        print("WARNING: Missing months may affect AR forecasts.")

    print("\nNelson-Siegel OLS completed successfully!")