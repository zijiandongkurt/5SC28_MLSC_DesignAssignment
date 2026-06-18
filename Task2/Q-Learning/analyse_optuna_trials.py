import os
import optuna
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'rbf_tuning.db').replace(os.sep, '/')

# We only care about these 5 simulation parameters
TARGET_PARAMS = ['n_bins', 'sigma', 'alpha', 'gamma', 'epsilon_decay']

def main():
    if not os.path.exists(os.path.join(BASE_DIR, 'rbf_tuning.db')):
        print("Database not found!")
        return

    print("Loading study...")
    study = optuna.load_study(
        study_name="rbf_swingup",
        storage=f"sqlite:///{DB_PATH}",
    )

    print(f"Total trials found: {len(study.trials)}")
    
    # 1. Extract dataframe
    df = study.trials_dataframe()
    
    # Filter only completed trials
    df = df[df['state'] == 'COMPLETE']
    print(f"Completed trials: {len(df)}")
    if len(df) == 0:
        print("No completed trials found.")
        return

    # Extract target param columns
    param_cols = [f"params_{p}" for p in TARGET_PARAMS]
    available_cols = [c for c in param_cols if c in df.columns]
    
    if not available_cols:
        print("No target parameters found in the study.")
        return

    # Sort by objective value (higher is better)
    df = df.sort_values(by="value", ascending=False)
    
    print("\n--- TOP 10 TRIALS ---")
    top_10 = df.head(10)[['number', 'value'] + available_cols]
    
    # Clean up column names for printing
    top_10.columns = ['Trial', 'Score'] + [c.replace('params_', '') for c in available_cols]
    print(top_10.to_string(index=False))

    # 2. Best Parameter Ranges
    top_percentile = 0.10
    top_n = max(1, int(len(df) * top_percentile))
    df_top = df.head(top_n)
    
    print(f"\n--- BEST RANGES (From Top {top_n} Trials) ---")
    ranges_data = []
    for col in available_cols:
        param_name = col.replace('params_', '')
        min_val = df_top[col].min()
        max_val = df_top[col].max()
        mean_val = df_top[col].mean()
        ranges_data.append({
            "Parameter": param_name,
            "Min": min_val,
            "Mean": mean_val,
            "Max": max_val
        })
    ranges_df = pd.DataFrame(ranges_data)
    print(ranges_df.to_string(index=False))

    # 3. Variable Correlations in Top Trials
    print(f"\n--- CORRELATIONS (From Top {top_n} Trials) ---")
    corr_df = df_top[available_cols].rename(columns=lambda x: x.replace('params_', ''))
    corr_matrix = corr_df.corr(method='spearman') # Spearman is better for non-linear/mixed monotonic relationships
    
    # Print the correlation matrix cleanly
    print("Spearman Correlation Matrix:")
    print(corr_matrix.round(2))
    
    # Extract strong correlations
    print("\nStrong Correlations (|r| >= 0.4):")
    found_strong = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.4 and not np.isnan(corr_val):
                found_strong = True
                print(f"  {var1} & {var2}: {corr_val:.2f} " + 
                      ("(Move in same direction)" if corr_val > 0 else "(Move in opposite directions)"))
    if not found_strong:
        print("  No strong correlations found among the target parameters.")

    # 4. Parameter Importance (relative to all parameters)
    print("\n--- PARAMETER IMPORTANCE ---")
    try:
        importances = optuna.importance.get_param_importances(study)
        # Filter to only show our target params, but show their relative importance out of the total
        total_importance = sum(importances.values())
        print("Importance of target simulation parameters relative to the entire tuning space:")
        for p in TARGET_PARAMS:
            if p in importances:
                print(f"  {p}: {importances[p]:.3f}")
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")

if __name__ == "__main__":
    main()
