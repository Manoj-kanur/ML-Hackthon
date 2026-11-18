"""
Multi-Model Experimentation and Benchmarking Engine for PathSnap
Trains and evaluates 5 models from 5 distinct algorithm families:
1. Logistic Regression (Linear / Probabilistic)
2. Decision Tree Classifier (Tree-based)
3. Random Forest Classifier (Ensemble / Bagging)
4. LightGBM Classifier (Boosting)
5. MLP Classifier (Neural Network)

Evaluates:
- 5-Fold GroupKFold CV Macro F1 on 80 training trips
- Student-Testing Observation-Level Top-1 Accuracy
- Macro F1, Precision, Recall, Log Loss
- Route Continuity (%)
- Training Time & Inference Latency
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)

FEATURE_COLS = [
    'dist_perp',
    'proj_ratio',
    'heading_diff',
    'speed',
    'arc_length',
    'u_degree',
    'v_degree',
    'dist_rank',
    'dist_gap_to_best',
    'dt',
    'accel',
    'is_connected_to_prev_closest'
]

def load_network_arcs():
    """Loads arcs for all student test trips to measure route continuity."""
    test_arcs_map = {}
    test_dir = os.path.join('data', 'student_test')
    for tid in os.listdir(test_dir):
        tdir = os.path.join(test_dir, tid)
        if os.path.isdir(tdir):
            afile = os.path.join(tdir, f"{tid}.arcs")
            arcs = pd.read_csv(afile, sep=r'\s+', header=None, names=['u', 'v']).values
            test_arcs_map[tid] = arcs
    return test_arcs_map

def compute_route_continuity(test_df, test_arcs_map):
    """
    Computes percentage of consecutive predicted arcs that are topologically connected
    (same arc, direct head-to-tail transition, or shared node).
    """
    total_transitions = 0
    connected_transitions = 0

    for tid, grp in test_df.groupby('trip_id'):
        if tid not in test_arcs_map:
            continue
        arcs = test_arcs_map[tid]
        
        # Sort by observation index
        obs_preds = grp.sort_values('obs_idx').drop_duplicates(subset=['obs_idx'])
        pred_arcs = obs_preds['pred_arc'].values

        for i in range(len(pred_arcs) - 1):
            a1 = pred_arcs[i]
            a2 = pred_arcs[i + 1]
            total_transitions += 1

            if a1 == a2:
                connected_transitions += 1
            else:
                u1, v1 = arcs[a1, 0], arcs[a1, 1]
                u2, v2 = arcs[a2, 0], arcs[a2, 1]
                # Direct transition: v1 == u2, or reverse/shared node
                if v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2:
                    connected_transitions += 1

    return (connected_transitions / max(total_transitions, 1)) * 100.0

def compute_observation_top1(df):
    """
    Selects the candidate with highest predicted probability per observation
    and compares it with ground truth true_arc.
    """
    best_cands = df.loc[df.groupby(['trip_id', 'obs_idx'])['pred_prob'].idxmax()]
    top1_acc = (best_cands['arc_id'] == best_cands['true_arc']).mean() * 100.0
    return top1_acc, best_cands

def get_models():
    """Defines 5 models from 5 distinct algorithm families."""
    models = {
        'Logistic Regression': {
            'family': 'Linear and Probabilistic',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=42))
            ]),
            'key_config': 'L2 penalty, C=1.0, StandardScaler, lbfgs solver'
        },
        'Decision Tree': {
            'family': 'Tree-based',
            'model': DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
            'key_config': 'max_depth=12, min_samples_leaf=20, Gini criterion'
        },
        'Random Forest': {
            'family': 'Ensemble (Bagging)',
            'model': RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=10, n_jobs=-1, random_state=42),
            'key_config': 'n_estimators=100, max_depth=16, min_samples_leaf=10'
        },
        'LightGBM': {
            'family': 'Boosting',
            'model': LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
            'key_config': 'n_estimators=100, lr=0.08, num_leaves=31, colsample=0.8'
        },
        'MLP Neural Network': {
            'family': 'Neural Networks',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001, max_iter=25, batch_size=1024, early_stopping=True, random_state=42))
            ]),
            'key_config': 'hidden=(128, 64), ReLU, alpha=1e-3, Adam, early_stopping'
        }
    }
    return models

def run_experiments():
    print("Loading feature matrices...")
    train_df = pd.read_parquet('artifacts/train_features.parquet')
    test_df = pd.read_parquet('artifacts/student_test_features.parquet')

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['is_match'].values
    groups_train = train_df['trip_id'].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['is_match'].values

    test_arcs_map = load_network_arcs()
    models_dict = get_models()

    results = []
    confusion_matrices = {}

    print(f"\nEvaluating {len(models_dict)} models across 5 algorithm families...")
    print(f"Training data: {len(train_df):,} candidate rows across {len(np.unique(groups_train))} trips.")
    print(f"Testing data:  {len(test_df):,} candidate rows across {len(test_df['trip_id'].unique())} trips.\n")

    for name, info in models_dict.items():
        family = info['family']
        model = info['model']
        config_desc = info['key_config']
        print(f"=== Training {name} ({family}) ===")

        # 1. 5-Fold GroupKFold Cross-Validation on the 80 training trips
        gkf = GroupKFold(n_splits=5)
        cv_macro_f1s = []
        cv_top1s = []

        # Subsample for fast cross-validation fold timing (using 25% of train for CV folds)
        cv_sample_trips = np.unique(groups_train)[::2]  # 40 trips for CV
        mask_cv = np.isin(groups_train, cv_sample_trips)
        X_cv = X_train[mask_cv]
        y_cv = y_train[mask_cv]
        g_cv = groups_train[mask_cv]
        df_cv = train_df[mask_cv].copy()

        t_cv_start = time.time()
        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_cv, y_cv, g_cv)):
            model.fit(X_cv[trn_idx], y_cv[trn_idx])
            val_preds = model.predict(X_cv[val_idx])
            f1 = f1_score(y_cv[val_idx], val_preds, average='macro', zero_division=0)
            cv_macro_f1s.append(f1)
        mean_cv_f1 = np.mean(cv_macro_f1s)
        cv_time = time.time() - t_cv_start
        print(f"  5-Fold GroupKFold CV Macro F1: {mean_cv_f1:.4f} (took {cv_time:.1f}s)")

        # 2. Fit on Full 80-Trip Training Data
        t_fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t_fit_start
        print(f"  Full fit on 80 trips completed in: {fit_time:.2f}s")

        # Save model artifact
        model_path = os.path.join('artifacts', 'models', f"{name.lower().replace(' ', '_')}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 3. Predict on 10 Student-Testing Trips
        t_inf_start = time.time()
        test_probs = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t_inf_start
        test_preds = (test_probs >= 0.5).astype(int)
        inf_per_point_us = (inference_time / len(X_test)) * 1e6

        # Standard Classification Metrics
        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
        
        try:
            ll = log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15))
        except Exception:
            ll = np.nan

        # Observation-level Top-1 Accuracy and Route Continuity
        sub_test_df = test_df[['trip_id', 'obs_idx', 'arc_id', 'true_arc']].copy()
        sub_test_df['pred_prob'] = test_probs
        top1_acc, best_cands = compute_observation_top1(sub_test_df)
        
        best_cands = best_cands.rename(columns={'arc_id': 'pred_arc'})
        continuity = compute_route_continuity(best_cands, test_arcs_map)

        cm = confusion_matrix(y_test, test_preds)
        confusion_matrices[name] = cm

        print(f"  Student Test Results -> Top-1 Acc: {top1_acc:.2f}%, Macro F1: {macro_f1:.4f}, Log Loss: {ll:.4f}")
        print(f"  Route Continuity: {continuity:.2f}%, Inference Latency: {inf_per_point_us:.2f} µs/cand\n")

        results.append({
            'Model': name,
            'Algorithm Family': family,
            'Key Configuration': config_desc,
            'CV Macro F1': round(mean_cv_f1, 4),
            'Student-Test Top-1 Acc (%)': round(top1_acc, 2),
            'Route Continuity (%)': round(continuity, 2),
            'Student-Test Macro F1': round(macro_f1, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'Log Loss': round(ll, 4) if not np.isnan(ll) else 'N/A',
            'Training Time (s)': round(fit_time, 2),
            'Inference Latency (µs/obs)': round(inf_per_point_us, 2)
        })

    res_df = pd.DataFrame(results)
    csv_out = os.path.join('artifacts', 'metrics', 'model_comparison.csv')
    res_df.to_csv(csv_out, index=False)
    print("=== Consolidated Model Comparison Table ===")
    print(res_df.to_string(index=False))

    # Generate Model Comparison Plots
    generate_comparison_plots(res_df, confusion_matrices)
    return res_df

def generate_comparison_plots(res_df, confusion_matrices):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Bar Chart: Top-1 Accuracy & Route Continuity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(res_df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, res_df['Student-Test Top-1 Acc (%)'], width, label='Top-1 Accuracy (%)', color='#1f77b4')
    rects2 = ax1.bar(x + width/2, res_df['Route Continuity (%)'], width, label='Route Continuity (%)', color='#2ca02c')

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('PathSnap: Model Comparison on Student-Testing Partition', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(res_df['Model'], rotation=15, ha='right', fontsize=11)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_ylim(0, 105)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join('visualizations', 'plots', 'model_accuracy_continuity_comparison.png')
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. CV Macro F1 vs Student Test Macro F1
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, res_df['CV Macro F1'], width, label='5-Fold CV Macro F1', color='#ff7f0e')
    rects2 = ax.bar(x + width/2, res_df['Student-Test Macro F1'], width, label='Student-Test Macro F1', color='#9467bd')
    ax.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation vs Test Generalization Macro F1', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Model'], rotation=15, ha='right', fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p2 = os.path.join('visualizations', 'plots', 'model_f1_comparison.png')
    plt.savefig(p2, dpi=300)
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Non-Match', 'Match'], yticklabels=['Non-Match', 'Match'])
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        if i == 0:
            axes[i].set_ylabel('True')
        else:
            axes[i].set_ylabel('')
    plt.tight_layout()
    p3 = os.path.join('visualizations', 'plots', 'confusion_matrices.png')
    plt.savefig(p3, dpi=300)
    plt.close()

    print(f"Saved comparison plots to visualizations/plots/")

if __name__ == '__main__':
    run_experiments()
