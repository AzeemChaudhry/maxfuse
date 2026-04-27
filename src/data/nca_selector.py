"""
NCA (Neighbourhood Component Analysis) feature selection.
Reduces 508-dim PE feature space to 80 dims. Implements Eq. (3).

Reference: Goldberger et al., "Neighbourhood Components Analysis", NeurIPS 2004.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler


def fit_nca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 80,
    save_path: str = 'outputs/nca_model.pkl',
    max_iter: int = 300,
    random_state: int = 42
) -> tuple:
    """
    Fit NCA on training data and save the model.

    NCA objective (Eq. 3):
        L_NCA = sum_i sum_{j: y_j = y_i} [ exp(-||Ax_i - Ax_j||^2) /
                                             sum_{l != i} exp(-||Ax_i - Ax_l||^2) ]

    Args:
        X_train: Feature matrix (N, 508)
        y_train: Class labels (N,)
        n_components: Output dimensions (default 80)
        save_path: Where to save fitted models
        max_iter: NCA optimisation iterations

    Returns:
        (scaler, nca) — fitted sklearn objects
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    nca = NeighborhoodComponentsAnalysis(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    nca.fit(X_scaled, y_train)

    joblib.dump({'scaler': scaler, 'nca': nca}, save_path)
    print(f"NCA model saved to {save_path}")
    return scaler, nca


def transform_nca(
    X: np.ndarray,
    model_path: str = 'outputs/nca_model.pkl'
) -> np.ndarray:
    """
    Transform feature matrix using a saved NCA model.

    Args:
        X: Feature matrix (N, 508) or single sample (508,)
        model_path: Path to saved model dict

    Returns:
        Transformed matrix (N, 80)
    """
    obj = joblib.load(model_path)
    scaler, nca = obj['scaler'], obj['nca']
    X_scaled = scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
    return nca.transform(X_scaled)


def run_nca_pipeline(
    in_csv: str,
    labels_csv: str,
    train_csv: str,
    out_csv: str,
    model_path: str = 'outputs/nca_model.pkl',
    n_components: int = 80
):
    """
    Full NCA pipeline: load features CSV -> fit on train split -> transform all -> save.
    """
    feat_df = pd.read_csv(in_csv)
    labels_df = pd.read_csv(labels_csv)
    train_files = set(pd.read_csv(train_csv)['filename'])

    feat_cols = [c for c in feat_df.columns if c != 'filename']
    X = feat_df[feat_cols].values.astype(np.float32)
    y = labels_df.set_index('filename').loc[feat_df['filename'], 'label_id'].values

    train_mask = feat_df['filename'].isin(train_files).values
    X_train, y_train = X[train_mask], y[train_mask]

    print(f"Fitting NCA on {X_train.shape[0]} train samples, {X.shape[1]} -> {n_components} dims")
    fit_nca(X_train, y_train, n_components=n_components, save_path=model_path)

    X_transformed = transform_nca(X, model_path=model_path)

    out_df = pd.DataFrame(X_transformed, columns=[f'nca_{i}' for i in range(n_components)])
    out_df['filename'] = feat_df['filename'].values
    out_df.to_csv(out_csv, index=False)
    print(f"Transformed features saved to {out_csv}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', required=True)
    parser.add_argument('--labels_csv', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--model_path', default='outputs/nca_model.pkl')
    parser.add_argument('--n_components', type=int, default=80)
    args = parser.parse_args()
    run_nca_pipeline(args.in_csv, args.labels_csv, args.train_csv,
                     args.out_csv, args.model_path, args.n_components)
