import os
import pandas as pd
import numpy as np

from sacred import Experiment
from scipy.stats import wilcoxon, friedmanchisquare
from sacred.observers import FileStorageObserver
from pathlib import Path

ex = Experiment('statistical')
ex.observers.append(FileStorageObserver('results/statistical_tests'))

@ex.config
def config():

    base_path = Path('/home/pedrobouzon/life/skin-type-guardrail/results/ours/')

    experiments = {
        "Margin 28_6º (Agglomerative)": base_path / "ddi_arcface_m28.6_dim128_p20_clusteragglomerative",
        "Margin 45º (Agglomerative)": base_path / "ddi_arcface_m45_dim128_p20_clusteragglomerative",
        "Margin 90º (Agglomerative)": base_path / "ddi_arcface_m90_dim128_p20_clusteragglomerative",
        "Margin 28_6º (KMeans)": base_path / "ddi_arcface_m28.6_dim128_p20_clusterkmeans",
        "Margin 45º (KMeans)": base_path / "ddi_arcface_m45_dim128_p20_clusterkmeans",
        "Margin 90º (KMeans)": base_path / "ddi_arcface_m90_dim128_p20_clusterkmeans",
    }

    print(experiments)

    metric = "f1_score"  
    output_filename = "experiment_comparison_results.csv"

def holm_bonferroni_correction(p_values, alpha=0.05):
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    corrected_p = np.zeros(m)
    
    current_max = 0.0
    for i, idx in enumerate(sorted_indices):
        p_adj = p_values[idx] * (m - i)
        p_adj = max(current_max, p_adj)
        current_max = p_adj
        corrected_p[idx] = min(p_adj, 1.0)
        
    return corrected_p

def load_cross_validation_results(exp_path, metric_col):
    scores_by_id = {}
    target_filename = "cross_validation_results.csv"
    for root, _, files in os.walk(exp_path):
        if target_filename in files:
            csv_path = os.path.join(root, target_filename)
            model_name = os.path.basename(root)
            df = pd.read_csv(csv_path)
            
            if metric_col not in df.columns: 
                continue
                
            for _, row in df.iterrows():
                fold_val = int(row['fold']) if 'fold' in df.columns else np.random.randint(1000)
                unique_id = f"{model_name}_fold_{fold_val}"
                scores_by_id[unique_id] = row[metric_col]
                
    return scores_by_id

@ex.main
def compare_experiments(experiments, metric, output_filename):
    scores_by_experiment = {}
    for experiment_name, experiment_path in experiments.items():
        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f'Experiment not found: {experiment_path}')
        scores_by_experiment[experiment_name] = load_cross_validation_results(experiment_path, metric)

    common_ids = set.intersection(*[set(scores.keys()) for scores in scores_by_experiment.values()])
    common_ids = sorted(list(common_ids))
    
    if not common_ids:
        raise ValueError("No matching backbone/fold combinations found across all experiments")

    print(f"Found {len(common_ids)} aligned evaluations per experiment.")

    aligned_scores = {exp: [scores_by_experiment[exp][uid] for uid in common_ids] for exp in experiments.keys()}
    experiment_names = list(aligned_scores.keys())

    if len(experiment_names) > 2:
        stat, p_friedman = friedmanchisquare(*[aligned_scores[exp] for exp in experiment_names])
        print(f"\nFriedman Test p-value: {p_friedman:.5f}")
    
        if p_friedman >= 0.05:
            print("No statistically significant differences found among any experiments. Stopping post-hoc tests.")
            return

    exp_means = {exp: np.mean(scores) for exp, scores in aligned_scores.items()}
    best_experiment = max(exp_means, key=exp_means.get)
    print(f"Top performing experiment: {best_experiment} (Mean: {exp_means[best_experiment]:.4f})\n")

    # Wilcoxon test (best vs. rest)
    results = []
    best_scores = np.array(aligned_scores[best_experiment])
    
    raw_p_values = []
    comparisons = []
    
    for other_exp in experiment_names:
        if other_exp == best_experiment:
            continue
            
        other_scores = np.array(aligned_scores[other_exp])
        diff = best_scores - other_scores
        
        try:
            _, p_val = wilcoxon(diff, alternative='two-sided')
        except ValueError:
            p_val = 1.0 
            
        raw_p_values.append(p_val)
        comparisons.append(other_exp)
        
    corrected_p_values = holm_bonferroni_correction(raw_p_values)
    
    for other_exp, raw_p, corr_p in zip(comparisons, raw_p_values, corrected_p_values):
        mean_diff = exp_means[best_experiment] - exp_means[other_exp]
        results.append({
            "experiment_a": best_experiment.replace('_', '.'),
            "experiment_b": other_exp.replace('_', '.'),
            "metric": metric,
            "n_samples": len(common_ids),
            "mean_difference": mean_diff,
            "raw_p_value": raw_p,
            "p_value": corr_p,
            "significant": "yes" if corr_p < 0.05 else "no"
        })

    df_stats = pd.DataFrame(results)
    
    out_path = os.path.join(ex.current_run.observers[0].dir, output_filename)
    df_stats.to_csv(out_path, index=False)
    print(f"Stats saved to {out_path}")

if __name__ == "__main__":
    ex.run()