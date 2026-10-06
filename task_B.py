import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import brentq
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
print("Loading data for Task B...")
train_df = pd.read_csv('data/train_B.csv')
test_df = pd.read_csv('data/test_B.csv')

y_train = train_df['y'].to_numpy(dtype=float)
X_train, X_test = design_matrix(train_df.drop(columns='y'), test_df.drop(columns='row_id'))

# ==========================================
# 2. FIT BASE MODEL & EXTRACT RESIDUALS
# ==========================================
print("Fitting base OLS model...")
w_base, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
y_pred_train = X_train @ w_base
residuals = y_train - y_pred_train

# ==========================================
# 3. FIT GAUSSIAN MIXTURE MODEL (EM Algorithm)
# ==========================================
print("Fitting Gaussian Mixture Model (4 components) to residuals via EM...")
# We observed 4 distinct peaks/clouds in the diagnostics
gmm = GaussianMixture(n_components=4, random_state=42, max_iter=500)
gmm.fit(residuals.reshape(-1, 1))
print(f"BIC score: {gmm.bic(residuals.reshape(-1, 1)):.1f}")

# Extract the learned parameters for the 4 sub-populations
weights = gmm.weights_
means = gmm.means_.flatten()
sigmas = np.sqrt(gmm.covariances_.flatten())

for k in range(4):
    print(f"Component {k+1}: Weight = {weights[k]:.3f}, Mean = {means[k]:.1f}, Sigma = {sigmas[k]:.1f}")

# ==========================================
# 4. ROBUST NUMERICAL QUANTILE SOLVER FOR GMM
# ==========================================
print("Calculating exact quantiles for the GMM error distribution...")
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Create a dense grid of possible residual values (based on our observed data bounds)
x_grid = np.linspace(-50000, 50000, 10000)

# Calculate the CDF for every point in the grid
def gmm_cdf_grid(x_array):
    cdf_array = np.zeros_like(x_array)
    for k in range(gmm.n_components):
        cdf_array += weights[k] * norm.cdf(x_array, loc=means[k], scale=sigmas[k])
    return cdf_array

# Evaluate the CDF over the entire grid
cdf_evaluated = gmm_cdf_grid(x_grid)

# Find the closest x_value for each target quantile
residual_quantiles = []
for q in QUANTILES:
    # Find the index where the CDF is closest to the desired quantile 'q'
    idx = np.argmin(np.abs(cdf_evaluated - q))
    rq = x_grid[idx]
    residual_quantiles.append(rq)
    print(f"q={q}: Residual offset = {rq:.1f}")

residual_quantiles = np.array(residual_quantiles)

# ==========================================
# 5. PREDICT QUANTILES FOR TEST SET
# ==========================================
print("Generating final predictive distributions for Test Set B...")
y_pred_test = X_test @ w_base

# Add the GMM residual quantiles to the base predictions
# y_pred_test is shape (N,), residual_quantiles is shape (9,)
# Resulting array is shape (N, 9)
final_predictions = y_pred_test[:, None] + residual_quantiles[None, :]

print("Task B quantiles generated and ready for submission!")


import pandas as pd
import numpy as np

# ==========================================
# 5. KAGGLE EXPORT FORMATTING
# ==========================================
# IMPORTANT: CHANGE THIS VARIABLE TO 'B', 'C', or 'D' in the respective files!
TASK = 'B' 

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