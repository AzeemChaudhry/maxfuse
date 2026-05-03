"""
Baseline multimodal model: simple CNN image branch + RUSBoost numeric branch
with late probability averaging.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from imblearn.ensemble import RUSBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .energy_ood import energy_score, reject_or_classify


class BaselineLateFusion(nn.Module):
    """
    Baseline model used for paper-style comparison.

    - Image branch: lightweight CNN trained with cross-entropy.
    - Numeric branch: RUSBoost classifier fitted on numeric features.
    - Fusion: late averaging of class probabilities (0.5 image + 0.5 numeric).
    """

    def __init__(self, num_classes: int = 25, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.numeric_scaler: Optional[StandardScaler] = None
        self.numeric_clf: Optional[RUSBoostClassifier] = None
        self.numeric_classes_: Optional[np.ndarray] = None

        self.register_buffer('tau', torch.tensor(float('inf')))

    def _image_logits(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(img)
        return self.classifier(feats)

    def fit_numeric_rusboost(self, loader) -> None:
        x_list, y_list = [], []
        for _, num_feats, labels in loader:
            x_list.append(num_feats.cpu().numpy())
            y_list.append(labels.cpu().numpy())

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        self.numeric_scaler = StandardScaler()
        x_scaled = self.numeric_scaler.fit_transform(x)

        self.numeric_clf = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
        )
        self.numeric_clf.fit(x_scaled, y)
        self.numeric_classes_ = np.asarray(self.numeric_clf.classes_)

    def _numeric_proba(self, num_feats: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        if self.numeric_clf is None or self.numeric_scaler is None or self.numeric_classes_ is None:
            return None

        x = num_feats.detach().cpu().numpy()
        x_scaled = self.numeric_scaler.transform(x)
        proba_partial = self.numeric_clf.predict_proba(x_scaled)

        proba_full = np.zeros((len(x), self.num_classes), dtype=np.float32)
        proba_full[:, self.numeric_classes_.astype(int)] = proba_partial.astype(np.float32)
        return torch.tensor(proba_full, device=device)

    def forward(self, img: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        logits_img = self._image_logits(img)
        probs_num = self._numeric_proba(num_feats, logits_img.device)

        if probs_num is None:
            return logits_img

        probs_img = torch.softmax(logits_img, dim=-1)
        probs = 0.5 * probs_img + 0.5 * probs_num
        return torch.log(probs.clamp_min(1e-8))

    @torch.no_grad()
    def inference(self, img: torch.Tensor, num_feats: torch.Tensor, tau: float = None) -> dict:
        threshold = tau if tau is not None else self.tau.item()
        logits = self.forward(img, num_feats)
        preds, energies = reject_or_classify(logits, threshold)
        w = torch.full((img.size(0), 1), 0.5, device=img.device)
        return {
            'predictions': preds,
            'logits': logits,
            'energies': energies,
            'weights_i': w,
            'weights_n': w,
        }

    def calibrate_threshold(self, val_loader, device: str = 'cuda') -> float:
        self.eval()
        energies = []
        with torch.no_grad():
            for img, num_feats, _ in val_loader:
                img = img.to(device)
                num_feats = num_feats.to(device)
                logits = self.forward(img, num_feats)
                energies.append(energy_score(logits).cpu())

        all_energies = torch.cat(energies)
        tau = float(torch.quantile(all_energies, 0.95))
        self.tau.fill_(tau)
        print(f"[Calibration] tau set to {tau:.4f}")
        return tau

    def export_non_torch_state(self) -> dict:
        return {
            'numeric_scaler': self.numeric_scaler,
            'numeric_clf': self.numeric_clf,
            'numeric_classes': self.numeric_classes_,
        }

    def load_non_torch_state(self, state: dict) -> None:
        self.numeric_scaler = state.get('numeric_scaler')
        self.numeric_clf = state.get('numeric_clf')
        self.numeric_classes_ = state.get('numeric_classes')

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
