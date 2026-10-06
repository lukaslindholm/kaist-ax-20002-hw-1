import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm
import os

# ==========================================
# 1. PRE-PROCESSING
# ==========================================
def design_matrix(train, test):
    both = pd.concat([train, test], keys=["tr", "te"])
    both = pd.get_dummies(both, drop_first=True, dtype=float)
    Xtr, Xte = both.loc["tr"].to_numpy(), both.loc["te"].to_numpy()
    
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    
    def add_bias(X):
        return np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    
    return add_bias(Xtr), add_bias(Xte)

print("Loading data for Task D...")
train_df = pd.read_csv('data/train_D.csv')
test_df = pd.read_csv('data/test_D.csv')

y_train = train_df['y'].to_numpy(dtype=float)
X_train, X_test = design_matrix(train_df.drop(columns='y'), test_df.drop(columns='row_id'))

# ==========================================
# 2. LEFT-CENSORED TOBIT LIKELIHOOD FUNCTION
# ==========================================
CENSORED_VALUE = 0.0

def tobit_left_neg_loglik(params, X, y):
    w = params[:-1]
    sigma = params[-1]
    
    if sigma <= 0:
        return np.inf
        
    y_pred = X @ w
    
    uncensored_mask = y > CENSORED_VALUE
    censored_mask = y <= CENSORED_VALUE
    
    # 1. Likelihood for uncensored data (y > 0)
    ll_uncensored = np.sum(norm.logpdf(y[uncensored_mask], loc=y_pred[uncensored_mask], scale=sigma))
    
    # 2. Likelihood for left-censored data (y == 0) using CDF
    ll_censored = np.sum(norm.logcdf(CENSORED_VALUE, loc=y_pred[censored_mask], scale=sigma))
    
    return -(ll_uncensored + ll_censored)

# ==========================================
# 3. OPTIMIZATION
# ==========================================
print("Training Left-Censored Tobit Model for Task D...")

# Initialize with OLS
w_init, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
sigma_init = np.std(y_train - X_train @ w_init)

initial_params = np.append(w_init, sigma_init)

result = minimize(
    tobit_left_neg_loglik, 
    initial_params, 
    args=(X_train, y_train),
    method='L-BFGS-B',
    bounds=[(None, None)] * len(w_init) + [(1e-6, None)]
)

best_w = result.x[:-1]
best_sigma = result.x[-1]

print(f"Optimization Success: {result.success}")
print(f"Fitted Sigma: {best_sigma:.4f} (OLS sigma was {sigma_init:.4f})")

# ==========================================
# 4. PREDICT QUANTILES
# ==========================================
print("Calculating Quantiles from Truncated Normal...")
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

y_pred_test = X_test @ best_w
predictions = []

for mu in y_pred_test:
    # truncnorm takes standardized bounds (a, b)
    # Left bound is 0.0, right bound is infinity
    a_std = (CENSORED_VALUE - mu) / best_sigma
    b_std = np.inf
    
    row_quantiles = truncnorm.ppf(QUANTILES, a=a_std, b=b_std, loc=mu, scale=best_sigma)
    predictions.append(row_quantiles)

final_predictions = np.array(predictions)
print("Task D quantiles generated and ready for final submission assembly!")


import pandas as pd
import numpy as np

# ==========================================
# 5. KAGGLE EXPORT FORMATTING
# ==========================================
# IMPORTANT: CHANGE THIS VARIABLE TO 'B', 'C', or 'D' in the respective files!
TASK = 'D' 

print(f"Formatting final predictions for Task {TASK} export...")

# Extract the correct predictions variable depending on the task
preds = final_predictions if 'final_predictions' in locals() else predictions

# Requirement 1: Sort the quantiles to ensure they are strictly non-decreasing
preds_sorted = np.sort(preds, axis=1)

# Requirement 2: Correct ID formatting (q10, q20 ... q90)
q_labels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
ids = []
values = []

# Iterate through test rows and all 9 quantiles
for i, row_id in enumerate(test_df['row_id']):
    for j, q in enumerate(q_labels):
        # Create the ID string exactly as requested (e.g., A_00000_q10)
        ids.append(f"{row_id}_q{q}")
        
        # Requirement 3: Sanitize data (handle missing/inf values if they appear)
        val = np.nan_to_num(preds_sorted[i, j], posinf=1e6, neginf=-1e6)
        values.append(val)

# Create the final single-task DataFrame and save to CSV
df_out = pd.DataFrame({'id': ids, 'value': values})
df_out.to_csv(f'sub_{TASK}.csv', index=False)

print(f"Successfully saved formatted file: sub_{TASK}.csv!")