import pandas as pd
import os

# ==========================================
# KAGGLE SUBMISSION MERGER
# ==========================================
print("Merging individual task files into final submission...")

# List of the single-task output files we just generated
files_to_merge = ['sub_A.csv', 'sub_B.csv', 'sub_C.csv', 'sub_D.csv']

dfs = []
for file in files_to_merge:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file} with {len(df)} rows.")
    else:
        print(f"ERROR: {file} not found. Did you run the task script?")

# Concatenate all dataframes vertically
if dfs:
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Save the final submission file
    df_final.to_csv('submission_final.csv', index=False)
    
    print("\n--- MERGE COMPLETE ---")
    print(f"Total rows in submission: {len(df_final)} (Expected: 201087)")
    print("Ready for Kaggle upload: submission_final.csv")