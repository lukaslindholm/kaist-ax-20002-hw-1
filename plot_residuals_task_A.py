import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import math

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

os.makedirs('plots', exist_ok=True)

file = 'data/train_A.csv'
task = 'A'

if not os.path.exists(file):
    print(f"Error: Could not find {file}")
else:
    df = pd.read_csv(file)
    y = df['y'].values
    X_raw = df.drop(columns=['y'])
    
    # Encode categorical variables
    X_encoded = pd.get_dummies(X_raw, drop_first=True, dtype=float)
    X = X_encoded.values
    
    # Train a Baseline LightGBM for the median (q50)
    model = lgb.LGBMRegressor(objective='quantile', alpha=0.5, random_state=42, verbose=-1)
    model.fit(X, y)
    
    # Calculate residuals: Actual y - model's median prediction
    y_pred_median = model.predict(X)
    residuals = y - y_pred_median
    
    features = list(X_raw.columns)
    n_features = len(features)
    
    # Split features into two equal halves
    mid_point = math.ceil(n_features / 2)
    feature_splits = [features[:mid_point], features[mid_point:]]
    
    for i, feature_subset in enumerate(feature_splits):
        n_sub = len(feature_subset)
        
        cols = 2 
        rows = math.ceil(n_sub / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_subset):
            axes[idx].scatter(X_raw[feature], residuals, alpha=0.3, s=5, color='royalblue')
            axes[idx].axhline(0, color='red', linestyle='--')
            axes[idx].set_title(f'LGBM Median Residuals vs {feature}', fontsize=12)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Residual')
            
        # Hide any empty subplots in the grid
        for idx in range(n_sub, len(axes)):
            fig.delaxes(axes[idx])
            
        # ONLY add main title to part 1
        if i == 0:
            plt.suptitle(f'Task {task}: Baseline LGBM Residuals vs Features', fontsize=14, y=1.02)
            
        plt.tight_layout()
        
        save_path = f'plots/baseline_lgbm_residuals_task_{task}_part{i+1}.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Part {i+1} plot to {save_path}")

print("Task A split feature diagnostic plots generated successfully!")