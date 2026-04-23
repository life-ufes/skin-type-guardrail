import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from functools import partial
from sklearn.metrics import (
    balanced_accuracy_score, 
    classification_report, 
    cohen_kappa_score, 
    f1_score, 
    mean_absolute_error,
    confusion_matrix,
)

from test.test import calculate_bootstrap_ci, plot_confusion_matrix
from config import DDI_PATCHES_DIR

PATCHES_ROOT_FOLDER = DDI_PATCHES_DIR
INPUT_CSV = "./results/ours/ddi_arcface_m28.6_dim128_p20_clusteragglomerative/1/test_metadata.csv" 
OUPUT_DIR = "results_test/kinyanjui"

N_BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

def compute_ita_kinyanjui_patch(img_path: str) -> float:
    if not os.path.exists(img_path): return np.nan
    
    img_bgr = cv.imread(img_path)
    if img_bgr is None: return np.nan
        
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)

    pixels_lab = img_lab.reshape(-1, 3)
    pixels_rgb = img_rgb.reshape(-1, 3)
    
    # Filter pure black padding
    valid_mask = np.any(pixels_rgb > 0, axis=1)
    valid_pixels = pixels_lab[valid_mask]
    
    if len(valid_pixels) == 0: return np.nan

    # opencv lab to standard cielab
    L = valid_pixels[:, 0] * (100.0 / 255.0)
    b = valid_pixels[:, 2] - 128.0
    
    # Keep values within one standard deviation
    L_mean, L_std = np.mean(L), np.std(L)
    b_mean, b_std = np.mean(b), np.std(b)
    
    valid_idx = (L >= L_mean - L_std) & (L <= L_mean + L_std) & \
                (b >= b_mean - b_std) & (b <= b_mean + b_std)
    
    L_filtered = L[valid_idx]
    b_filtered = b[valid_idx]
    
    if len(L_filtered) == 0: return np.nan
        
    L_final = np.mean(L_filtered)
    b_final = np.mean(b_filtered)
    
    if b_final == 0: b_final = 0.001 
    ita = np.arctan((L_final - 50.0) / b_final) * (180.0 / np.pi)
    #ita = np.arctan2((L_final - 50.0), b_final) * (180.0 / np.pi)
    return ita

def get_fitzpatrick_type(ita_angle: float, schema: str = 'kinyanjui') -> str:
    if pd.isna(ita_angle): return 'Unknown'
    if schema == 'kinyanjui':
        if ita_angle >= 41: return 'I-IV'
        elif ita_angle >= 19: return 'I-IV'
        else: return 'V-VI'

    elif schema == 'del bino':
        if ita_angle >= 55: return 'I-IV'
        elif ita_angle >= 41: return 'I-IV'
        elif ita_angle >= 28: return 'I-IV'
        elif ita_angle >= 10: return 'I-IV'
        elif ita_angle >= -30: return 'V-VI'
        else: return 'V-VI'

    elif schema == 'groh':
        if ita_angle >= 40: return 'I-IV'
        elif ita_angle >= 23: return 'I-IV'
        elif ita_angle >= 12: return 'I-IV'
        elif ita_angle >= 0: return 'I-IV'
        elif ita_angle >= -25: return 'V-VI'
        else: return 'V-VI'

if __name__ == "__main__":
    if not os.path.exists(OUPUT_DIR): os.makedirs(OUPUT_DIR)

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print("Computing ITA from patches...")
    tqdm.pandas()
    df['patch_ita_angle'] = df['filename'].progress_apply(
        lambda x: compute_ita_kinyanjui_patch(os.path.join(PATCHES_ROOT_FOLDER, str(x)))
    )

    agg_df = df.groupby('original_image').agg({
        'patch_ita_angle': 'mean',
        'skin_tone': 'first'
    }).reset_index()

    agg_df['estimated_fst'] = agg_df['patch_ita_angle'].apply(get_fitzpatrick_type)

    target_mapping = {12: 0, 34: 0, 56: 1, 'I-II': 0, 'III-IV': 0, 'V-VI': 1}
    pred_mapping = {'I-IV': 0, 'V-VI': 1}
    target_names = ['I-IV', 'V-VI']

    y_true = agg_df['skin_tone'].map(target_mapping).values
    y_pred = agg_df['estimated_fst'].map(pred_mapping).values

    metrics_functions = {
        'kappa': partial(cohen_kappa_score, weights='quadratic'),
        'bacc': balanced_accuracy_score,
        'mae': mean_absolute_error,
        'f1_macro': partial(f1_score, average='macro')
    }

    results = {}
    for name, metric_function in metrics_functions.items():
        score, low, high, dist = calculate_bootstrap_ci(
            y_true, y_pred, metric_function, N_BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL, RANDOM_STATE
        )
        results[name] = {'score': score, 'low': low, 'high': high, 'dist': dist}
        print(f"{name.upper()}: {score:.4f} [{low:.4f} - {high:.4f}]")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, OUPUT_DIR, "kinyanjui")

    agg_df.to_csv(os.path.join(OUPUT_DIR, "aggregated_predictions.csv"), index=False)
    pd.DataFrame([
        {'metric': name, 'score': data['score'], 'ci_lower': data['low'], 'ci_upper': data['high']}
        for name, data in results.items()
    ]).to_csv(os.path.join(OUPUT_DIR, "kinyanjui_confidence_intervals.csv"), index=False)
    
    print(f"\nAll plots and metrics saved to {OUPUT_DIR}")