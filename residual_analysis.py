import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

tasks = ['A', 'B', 'C', 'D']
files = ['data/train_A.csv', 'data/train_B.csv', 'data/train_C.csv', 'data/train_D.csv'] 

# Skapa mappen 'plots' om den inte redan finns
os.makedirs('plots', exist_ok=True)

for i, (task, file) in enumerate(zip(tasks, files)):
    if not os.path.exists(file):
        print(f"Error: Could not find {file}")
        continue
        
    df = pd.read_csv(file)
    y = df['y'].values
    X_df = df.drop(columns=['y'])
    
    # OLS Pre-processing
    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    X = X_df.values
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    X_std = (X - mu) / sd
    
    # Add bias
    X_design = np.hstack([X_std, np.ones((len(X_std), 1))])
    
    # Calculate OLS weights and residuals
    w, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ w
    residuals = y - y_pred
    
    # Skapa en ny figur för varje task: 4 rader, 1 kolumn
    fig, axes = plt.subplots(4, 1, figsize=(8, 16))
    
    # 1: Target distribution
    sns.histplot(y, kde=True, ax=axes[0])
    axes[0].set_title(f'Task {task}: Target (y) Dist')
    
    # 2: Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title(f'Task {task}: Residuals Histogram')
    
    # 3: QQ-plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f'Task {task}: QQ-Plot (Normality)')
    
    # 4: Residuals vs Predicted
    axes[3].scatter(y_pred, residuals, alpha=0.3, s=5)
    axes[3].axhline(0, color='red', linestyle='--')
    axes[3].set_title(f'Task {task}: Residuals vs Pred')
    axes[3].set_xlabel('Predicted y')
    axes[3].set_ylabel('Residual')

    plt.tight_layout()
    
    # Spara som PDF i mappen 'plots'
    save_path = f'plots/diagnostics_task_{task}.pdf'
    plt.savefig(save_path)
    plt.close() # Stänger figuren för att frigöra minne inför nästa loop
    
    print(f"Sparade diagnostik för Task {task} i {save_path}")

print("Alla figurer är genererade!")