"""
SMOTE-ENC class balancing for the training split.
Implements Eq. (4) from the MAXFUSE paper.

x_syn = x_i + lambda*(x_j - x_i),  lambda ~ U(0,1)

Output CSV format: [filename, label_name, label_id, nca_0, ..., nca_79]
Synthetic samples borrow the filename/label_name of their nearest real neighbor
so MalwareDataset can load the corresponding image.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def balance_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42
) -> tuple:
    """
    Apply SMOTE oversampling to balance the training set.

    Args:
        X_train: Feature matrix (N, 80)
        y_train: Labels (N,)
        k_neighbors: k for SMOTE nearest neighbours
        random_state: Reproducibility seed

    Returns:
        (X_resampled, y_resampled) — balanced versions
    """
    print(f"Before SMOTE: {Counter(y_train)}")
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {Counter(y_res)}")
    return X_res.astype(np.float32), y_res


def run_smote_pipeline(
    features_csv: str,
    train_csv: str,
    labels_csv: str,
    out_balanced_csv: str,
    k_neighbors: int = 5
):
    """
    Load NCA-reduced features, apply SMOTE on train split, save balanced CSV.

    Output columns: [filename, label_name, label_id, nca_0, ..., nca_79]
    Synthetic samples are assigned the filename and label_name of their nearest
    real training neighbor so MalwareDataset can load an image for them.
    """
    feat_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv).set_index('filename')
    train_files_df = pd.read_csv(train_csv)
    train_files = set(train_files_df['filename'])

    feat_cols = [c for c in feat_df.columns if c != 'filename']
    train_df = feat_df[feat_df['filename'].isin(train_files)].copy().reset_index(drop=True)

    filenames_orig = train_df['filename'].values
    label_ids_orig = labels_df.loc[filenames_orig, 'label_id'].values.astype(int)
    label_names_orig = labels_df.loc[filenames_orig, 'label_name'].values

    X_train = train_df[feat_cols].values.astype(np.float32)

    X_res, y_res = balance_train(X_train, label_ids_orig, k_neighbors=k_neighbors)

    n_orig = len(X_train)
    n_syn = len(X_res) - n_orig

    # For synthetic samples, find nearest real neighbor to borrow metadata
    if n_syn > 0:
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_train)
        X_syn = X_res[n_orig:]
        _, nn_indices = nn_model.kneighbors(X_syn)
        nn_indices = nn_indices.flatten()
        syn_filenames = filenames_orig[nn_indices]
        syn_label_names = label_names_orig[nn_indices]
    else:
        syn_filenames = np.array([], dtype=object)
        syn_label_names = np.array([], dtype=object)

    all_filenames = np.concatenate([filenames_orig, syn_filenames])
    all_label_names = np.concatenate([label_names_orig, syn_label_names])

    out_df = pd.DataFrame(X_res, columns=feat_cols)
    out_df.insert(0, 'label_id', y_res)
    out_df.insert(0, 'label_name', all_label_names)
    out_df.insert(0, 'filename', all_filenames)

    out_df.to_csv(out_balanced_csv, index=False)
    print(f"Balanced training data saved to {out_balanced_csv} "
          f"({n_orig} original + {n_syn} synthetic = {len(out_df)} total)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--out_csv', required=True)
    args = parser.parse_args()
    run_smote_pipeline(args.features_csv, args.train_csv, args.labels_csv, args.out_csv)
