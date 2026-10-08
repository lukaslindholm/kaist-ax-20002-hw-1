import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

TASKS = ['A', 'B', 'C']

for task in TASKS:
    print(f"--- Skapar residual-diagnostik för Task {task} ---")

    # Load data
    train = pd.read_csv(f'data/train_{task}.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True)
    feature_names = X_df.columns
    X = X_df.to_numpy(dtype=float)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a median model (q=0.5) to capture average error
    model = LGBMRegressor(
        objective='quantile',
        alpha=0.5,
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    
    # Clip constraints if known
    if task == 'A':
        val_preds = np.clip(val_preds, a_min=None, a_max=5.0)
        
    residuals = y_val - val_preds

    # --- PLOTTING ---
    num_features = len(feature_names)
    cols = 3
    rows = (num_features + cols - 1) // cols

    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle(f'Residualer vs Features för Task {task} (LGBM Median)', fontsize=16)

    for i, feature in enumerate(feature_names):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(X_val[:, i], residuals, alpha=0.3, color='blue', s=2)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(feature)
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'Residual vs {feature}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f'residuals_vs_features_task_{task}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Sparade {filename}")

print("Klart!")