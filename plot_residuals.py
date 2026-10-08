import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("--- Skapar residual-diagnostik för Task D ---")

# Ladda data för Task D
train = pd.read_csv('data/train_D.csv')
y = train['y'].to_numpy(dtype=float)
X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True)
feature_names = X_df.columns
X = X_df.to_numpy(dtype=float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vi tränar modellen bara för medianen (q=0.5) för att analysera det genomsnittliga felet
model = LGBMRegressor(
    objective='quantile',
    alpha=0.5,
    n_estimators=1000,
    num_leaves=127,
    learning_rate=0.02,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Räkna ut residualerna på valideringsdatan
val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0.0, 20.0) # Vår kända domängräns
residuals = y_val - val_preds

# --- PLOTTNING ---
num_features = len(feature_names)
cols = 3
rows = (num_features + cols - 1) // cols

plt.figure(figsize=(15, 4 * rows))
plt.suptitle('Residualer vs Features för Task D (LightGBM)', fontsize=16)

for i, feature in enumerate(feature_names):
    plt.subplot(rows, cols, i + 1)
    # Plotta residualerna mot respektive kolumn i X_val
    plt.scatter(X_val[:, i], residuals, alpha=0.1, color='blue', s=2)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(feature)
    plt.ylabel('Residual (True - Predicted)')
    plt.title(f'Residual vs {feature}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('residuals_vs_features.png')
plt.show()