import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

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
print("Loading data for Task C...")
train_df = pd.read_csv('data/train_C.csv')
test_df = pd.read_csv('data/test_C.csv')

y_train = train_df['y'].to_numpy(dtype=float)
X_train, X_test = design_matrix(train_df.drop(columns='y'), test_df.drop(columns='row_id'))

# ==========================================
# 2. FIT BASE MODEL & EXTRACT RESIDUALS
# ==========================================
print("Fitting base OLS model to find y_hat...")
w_base, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
y_pred_train = X_train @ w_base
residuals = y_train - y_pred_train

# ==========================================
# 3. MODEL THE HETEROSCEDASTICITY (DYNAMIC SIGMA)
# ==========================================
print("Modeling the dynamic variance (sigma as a function of y_hat)...")
abs_residuals = np.abs(residuals)

# Fit a simple linear regression: |residual| = a + b * y_hat
sigma_model = LinearRegression()
sigma_model.fit(y_pred_train.reshape(-1, 1), abs_residuals)

# Predict the dynamic sigma for the training set
# We use np.maximum to ensure sigma never mathematically drops below a tiny positive number
sigma_train = np.maximum(sigma_model.predict(y_pred_train.reshape(-1, 1)), 1e-6)

print(f"Variance model coefficients: Intercept = {sigma_model.intercept_:.3f}, Slope = {sigma_model.coef_[0]:.3f}")

# ==========================================
# 4. CALCULATE EMPIRICAL QUANTILES
# ==========================================
print("Standardizing residuals to calculate empirical quantiles...")
# Divide residuals by their respective dynamic sigma to "flatten the funnel"
standardized_residuals = residuals / sigma_train

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# We use empirical quantiles to capture any remaining skewness natively
empirical_quantiles = np.quantile(standardized_residuals, QUANTILES)

for q, eq in zip(QUANTILES, empirical_quantiles):
    print(f"q={q}: Standardized offset = {eq:.3f}")

# ==========================================
# 5. PREDICT QUANTILES FOR TEST SET
# ==========================================
print("Generating final predictive distributions for Test Set C...")
y_pred_test = X_test @ w_base

# Predict dynamic sigma for the test set
sigma_test = np.maximum(sigma_model.predict(y_pred_test.reshape(-1, 1)), 1e-6)

# Generate final predictions: y_hat + (dynamic_sigma * empirical_quantile)
# y_pred_test is (N,), sigma_test is (N,), empirical_quantiles is (9,)
final_predictions = y_pred_test[:, None] + sigma_test[:, None] * empirical_quantiles[None, :]

print("Task C quantiles generated and ready for submission!")


import pandas as pd
import numpy as np

# ==========================================
# 5. KAGGLE EXPORT FORMATTING
# ==========================================
# IMPORTANT: CHANGE THIS VARIABLE TO 'B', 'C', or 'D' in the respective files!
TASK = 'C' 

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