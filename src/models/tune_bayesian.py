"""
Bayesian Hyperparameter Optimization for PathSnap Champion Model (LightGBM)
Uses Optuna with Tree-structured Parzen Estimator (TPE) sampler.
Optimizes 5-fold GroupKFold Macro F1 / Top-1 Accuracy across the 80 training trips.
Saves optimization history, parameter importance plots, and tuned model artifacts.
"""

import os
import time
import pickle
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, log_loss
from src.models.trainer import FEATURE_COLS, load_network_arcs, compute_route_continuity, compute_observation_top1

# Suppress optuna verbosity for clean logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials=35):
    print("Loading feature matrices for Bayesian Hyperparameter Optimization...")
    train_df = pd.read_parquet('artifacts/train_features.parquet')
    test_df = pd.read_parquet('artifacts/student_test_features.parquet')

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['is_match'].values
    groups_train = train_df['trip_id'].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['is_match'].values

    test_arcs_map = load_network_arcs()

    # Subsample trips for leak-free, fast 5-fold CV evaluation during search (40 trips)
    cv_sample_trips = np.unique(groups_train)[::2]
    mask_cv = np.isin(groups_train, cv_sample_trips)
    X_cv = X_train[mask_cv]
    y_cv = y_train[mask_cv]
    g_cv = groups_train[mask_cv]

    print(f"Defining Optuna search space across 35 trials on 5-fold GroupKFold...")

    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.20, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 14),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        fold_scores = []
        for trn_idx, val_idx in gkf.split(X_cv, y_cv, g_cv):
            clf = LGBMClassifier(**params)
            clf.fit(X_cv[trn_idx], y_cv[trn_idx])
            preds = clf.predict(X_cv[val_idx])
            score = f1_score(y_cv[val_idx], preds, average='macro', zero_division=0)
            fold_scores.append(score)

        return np.mean(fold_scores)

    t0 = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Running Bayesian optimization trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_time = time.time() - t0

    print(f"\nBayesian Optimization completed in {opt_time:.1f}s ({opt_time/60:.2f} mins).")
    print(f"Best Trial #{study.best_trial.number}: CV Macro F1 = {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save trial results to CSV
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join('artifacts', 'metrics', 'bayesian_trials.csv')
    trials_df.to_csv(trials_csv, index=False)
    print(f"Trials history saved to: {trials_csv}")

    # Plot Optimization History and Parameter Importances
    plot_optimization_results(study)

    # Refit Tuned Model on Full 80 Training Trips
    print("\nRefitting Tuned Champion Model on full 80-trip training set...")
    best_params = study.best_params.copy()
    best_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})

    t_fit_start = time.time()
    tuned_model = LGBMClassifier(**best_params)
    tuned_model.fit(X_train, y_train)
    tuned_fit_time = time.time() - t_fit_start
    print(f"Tuned model refit completed in {tuned_fit_time:.2f}s.")

    # Save Tuned Model Artifact
    tuned_path = os.path.join('artifacts', 'models', 'tuned_lightgbm.pkl')
    with open(tuned_path, 'wb') as f:
        pickle.dump(tuned_model, f)
    print(f"Tuned model artifact saved to: {tuned_path}")

    # Evaluate Tuned Model on Student-Testing Trips
    t_inf_start = time.time()
    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    tuned_inf_time = time.time() - t_inf_start
    test_preds = (test_probs >= 0.5).astype(int)
    tuned_inf_us = (tuned_inf_time / len(X_test)) * 1e6

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    ll = log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15))

    sub_test_df = test_df[['trip_id', 'obs_idx', 'arc_id', 'true_arc']].copy()
    sub_test_df['pred_prob'] = test_probs
    top1_acc, best_cands = compute_observation_top1(sub_test_df)
    best_cands = best_cands.rename(columns={'arc_id': 'pred_arc'})
    continuity = compute_route_continuity(best_cands, test_arcs_map)

    # Load Untuned Baseline Metrics for direct comparison
    comp_df = pd.read_csv('artifacts/metrics/model_comparison.csv')
    untuned_row = comp_df[comp_df['Model'] == 'LightGBM'].iloc[0]

    comparison_results = [
        {
            'Version': 'Untuned LightGBM',
            'CV Macro F1': untuned_row['CV Macro F1'],
            'Student-Test Top-1 Acc (%)': untuned_row['Student-Test Top-1 Acc (%)'],
            'Student-Test Macro F1': untuned_row['Student-Test Macro F1'],
            'Route Continuity (%)': untuned_row['Route Continuity (%)'],
            'Log Loss': untuned_row['Log Loss'],
            'Training Time (s)': untuned_row['Training Time (s)'],
            'Inference Latency (µs/obs)': untuned_row['Inference Latency (µs/obs)']
        },
        {
            'Version': 'Tuned LightGBM (Bayesian)',
            'CV Macro F1': round(study.best_value, 4),
            'Student-Test Top-1 Acc (%)': round(top1_acc, 2),
            'Student-Test Macro F1': round(macro_f1, 4),
            'Route Continuity (%)': round(continuity, 2),
            'Log Loss': round(ll, 4),
            'Training Time (s)': round(tuned_fit_time, 2),
            'Inference Latency (µs/obs)': round(tuned_inf_us, 2)
        }
    ]

    opt_comp_df = pd.DataFrame(comparison_results)
    opt_comp_csv = os.path.join('artifacts', 'metrics', 'untuned_vs_tuned_comparison.csv')
    opt_comp_df.to_csv(opt_comp_csv, index=False)

    print("\n=== Untuned vs Tuned Model Performance Comparison ===")
    print(opt_comp_df.to_string(index=False))

    return opt_comp_df

def plot_optimization_results(study):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    trials = study.trials_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Optimization History
    best_vals = [study.trials[0].value]
    for t in study.trials[1:]:
        if t.value is not None:
            best_vals.append(max(best_vals[-1], t.value))
        else:
            best_vals.append(best_vals[-1])

    axes[0].scatter(trials['number'], trials['value'], color='#1f77b4', alpha=0.7, s=40, label='Trial Score')
    axes[0].plot(trials['number'], best_vals, color='#d62728', linewidth=2.5, label='Best Cumulative F1')
    axes[0].set_title('Bayesian Optimization Progression (35 Trials)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Trial Number', fontsize=11)
    axes[0].set_ylabel('5-Fold CV Macro F1', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=10)

    # 2. Key Hyperparameter Relationships (Learning Rate vs Num Leaves)
    sc = axes[1].scatter(trials['params_learning_rate'], trials['params_num_leaves'], 
                         c=trials['value'], cmap='viridis', s=60, alpha=0.85)
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label('CV Macro F1', fontsize=10)
    axes[1].set_title('Hyperparameter Space: Learning Rate vs Num Leaves', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Learning Rate (log scale)', fontsize=11)
    axes[1].set_ylabel('Num Leaves', fontsize=11)
    axes[1].set_xscale('log')

    plt.tight_layout()
    p_out = os.path.join('visualizations', 'plots', 'bayesian_optimization_analysis.png')
    plt.savefig(p_out, dpi=300)
    plt.close()
    print(f"Optimization plots saved to: {p_out}")

if __name__ == '__main__':
    run_bayesian_optimization(n_trials=35)
