"""
SHAP explainability for the numeric branch.
Assigns contribution scores to each PE header feature.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017.
Library: shap (pip install shap)
"""

import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class ShapExplainer:
    """
    SHAP explainer for the MAXFUSE numeric branch.
    Uses KernelSHAP (model-agnostic) to compute feature attributions.
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: list = None,
        device: str = 'cpu'
    ):
        """
        Args:
            model: MAXFUSE model instance
            background_data: (N_background, 80) numpy array
            feature_names: Names of the 80 NCA features (optional)
            device: Device for forward passes
        """
        self.model   = model.to(device).eval()
        self.device  = device
        self.feature_names = feature_names or [f'nca_{i}' for i in range(80)]

        def numeric_predict(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32).to(device)
                img_dummy = torch.zeros(len(x), 1, 224, 224, device=device)
                v_i = model.image_encoder(img_dummy)
                v_n = model.numeric_encoder(x_t)
                if model.use_n1:
                    v_hat_i, v_hat_n = model.cross_attn(v_i, v_n)
                else:
                    v_hat_i = model.proj_i_fallback(v_i)
                    v_hat_n = model.proj_n_fallback(v_n)
                z = 0.5 * v_hat_i + 0.5 * v_hat_n
                logits = model.classifier(z)
                probs  = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()

        self.predict_fn = numeric_predict

        bg_summary = shap.kmeans(background_data, 50)
        self.explainer = shap.KernelExplainer(self.predict_fn, bg_summary)

    def explain(
        self,
        numeric_feats: np.ndarray,
        target_class: int = None,
        save_path: str = None
    ) -> np.ndarray:
        """
        Compute SHAP values for a single sample.

        Args:
            numeric_feats: (80,) or (1, 80) feature array
            target_class: Class to explain (None = predicted class)
            save_path: Optional path for summary bar plot

        Returns:
            shap_values: (80,) attribution array for target_class
        """
        if numeric_feats.ndim == 1:
            numeric_feats = numeric_feats.reshape(1, -1)

        shap_values = self.explainer.shap_values(numeric_feats, nsamples=100)

        if target_class is not None:
            sv = shap_values[target_class][0]
        else:
            probs = self.predict_fn(numeric_feats)
            target_class = int(probs.argmax())
            sv = shap_values[target_class][0]

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_idx = np.argsort(np.abs(sv))[-20:]
            ax.barh(
                [self.feature_names[i] for i in sorted_idx],
                sv[sorted_idx],
                color=['#d73027' if v > 0 else '#4575b4' for v in sv[sorted_idx]]
            )
            ax.set_xlabel('SHAP value (impact on prediction)')
            ax.set_title(f'Top 20 feature attributions for class {target_class}')
            ax.axvline(0, color='black', linewidth=0.5)
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"SHAP plot saved to {save_path}")

        return sv

    def summary_plot(
        self,
        X_test: np.ndarray,
        n_samples: int = 100,
        save_path: str = 'outputs/xai/shap_summary.png'
    ):
        """Generate a SHAP beeswarm summary plot over multiple test samples."""
        X_sample = X_test[:n_samples]
        shap_values = self.explainer.shap_values(X_sample, nsamples=50)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary saved to {save_path}")
