import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm

# ==========================================
# 1. PRE-PROCESSING (Same as starter.py)
# ==========================================
def design_matrix(train, test):
    both = pd.concat([train, test], keys=["tr", "te"])
    both = pd.get_dummies(both, drop_first=True, dtype=float)
    Xtr, Xte = both.loc["tr"].to_numpy(), both.loc["te"].to_numpy()
    
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    
    def add_bias(X):
        return np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    
    return add_bias(Xtr), add_bias(Xte)

# Load data
train_df = pd.read_csv('data/train_A.csv')
test_df = pd.read_csv('data/test_A.csv')

y_train = train_df['y'].to_numpy(dtype=float)
X_train, X_test = design_matrix(train_df.drop(columns='y'), test_df.drop(columns='row_id'))

# ==========================================
# 2. TOBIT LIKELIHOOD FUNCTION
# ==========================================
CENSORED_VALUE = 5.0

# We define the negative log-likelihood (so we can minimize it)
def tobit_neg_loglik(params, X, y):
    # params contains the weights (w) and the standard deviation (sigma)
    w = params[:-1]
    sigma = params[-1]
    
    # Sigma must be positive
    if sigma <= 0:
        return np.inf
        
    y_pred = X @ w
    
    # Separate data into uncensored (y < 5) and censored (y == 5)
    uncensored_mask = y < CENSORED_VALUE
    censored_mask = y >= CENSORED_VALUE
    
    # 1. Likelihood for uncensored data (Standard Normal PDF)
    # Using scipy.stats.norm.logpdf is safer numerically
    ll_uncensored = np.sum(norm.logpdf(y[uncensored_mask], loc=y_pred[uncensored_mask], scale=sigma))
    
    # 2. Likelihood for censored data (1 - CDF)
    # We use logsf (log survival function) which is log(1 - cdf) for numerical stability
    ll_censored = np.sum(norm.logsf(CENSORED_VALUE, loc=y_pred[censored_mask], scale=sigma))
    
    # Return negative total log-likelihood
    total_ll = ll_uncensored + ll_censored
    return -total_ll

# ==========================================
# 3. OPTIMIZATION
# ==========================================
print("Training Tobit Model for Task A...")

# Initialize weights using OLS to give the optimizer a good starting point
w_init, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
sigma_init = np.std(y_train - X_train @ w_init)

# Combine initial parameters
initial_params = np.append(w_init, sigma_init)

# Run L-BFGS-B optimizer
result = minimize(
    tobit_neg_loglik, 
    initial_params, 
    args=(X_train, y_train),
    method='L-BFGS-B',
    bounds=[(None, None)] * len(w_init) + [(1e-6, None)] # sigma must be > 0
)

best_w = result.x[:-1]
best_sigma = result.x[-1]

print(f"Optimization Success: {result.success}")
print(f"Fitted Sigma: {best_sigma:.4f} (OLS sigma was {sigma_init:.4f})")

# ==========================================
# 4. PREDICT QUANTILES
# ==========================================
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

y_pred_test = X_test @ best_w
predictions = []

# Because the data is capped at 5.0, the test predictions shouldn't exceed 5.0.
# We extract quantiles from a Truncated Normal Distribution (capped at 5.0)
for mu in y_pred_test:
    # truncnorm takes standardized bounds (a, b)
    # a = lower bound (none, so -inf), b = upper bound (5.0 - mu)/sigma
    b_std = (CENSORED_VALUE - mu) / best_sigma
    
    row_quantiles = truncnorm.ppf(QUANTILES, a=-np.inf, b=b_std, loc=mu, scale=best_sigma)
    predictions.append(row_quantiles)

predictions = np.array(predictions)

print("Quantiles calculated! Ready to format for submission.")