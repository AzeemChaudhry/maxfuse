"""
Energy Margin Loss for MAXFUSE training.
Implements Equation (17) from the MAXFUSE paper.

L = L_CE(z, y) + alpha * [E[max(0, E_in - m_in)] + E[max(0, m_out - E_out)]]

Where:
    L_CE    = standard cross-entropy on in-distribution samples
    E_in    = energy scores of in-distribution samples (should be low / negative)
    E_out   = energy scores of OOD samples (should be high / less negative)
    m_in    = -25  (in-dist energy margin)
    m_out   = -7   (OOD energy margin)
    alpha   = 0.1  (balance coefficient)

Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.energy_ood import energy_score


class EnergyMarginLoss(nn.Module):
    """
    Combined cross-entropy + energy margin loss.

    During training you need two batches per step:
        1. In-distribution batch (images + labels from Malimg)
        2. OOD batch (random noise or held-out families, no labels needed)
    """

    def __init__(
        self,
        m_in: float = -25.0,
        m_out: float = -7.0,
        alpha: float = 0.1,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            m_in: Desired upper energy bound for in-dist samples
            m_out: Desired lower energy bound for OOD samples
            alpha: Weight for energy margin term
            label_smoothing: Smoothing for cross-entropy (reduces overconfidence)
        """
        super().__init__()
        self.m_in  = m_in
        self.m_out = m_out
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_id:  torch.Tensor,
        labels_id:  torch.Tensor,
        logits_ood: torch.Tensor
    ) -> tuple:
        """
        Compute total loss.

        Args:
            logits_id:  (B_id, num_classes) — in-distribution classifier output
            labels_id:  (B_id,) — ground-truth family labels
            logits_ood: (B_ood, num_classes) — OOD classifier output

        Returns:
            total_loss: scalar tensor
            loss_dict:  dict with breakdown (for logging)
        """
        # -- Cross-entropy on in-distribution samples --
        l_ce = self.ce(logits_id, labels_id)

        # -- Energy scores (consistent with inference via energy_score()) --
        E_in  = energy_score(logits_id)
        E_out = energy_score(logits_ood)

        # -- Margin hinge terms (Eq. 17) --
        # Push E_in  below m_in  (in-dist should have low / negative energy)
        l_in  = F.relu(E_in  - self.m_in).mean()

        # Push E_out above m_out (OOD should have high / less negative energy)
        l_out = F.relu(self.m_out - E_out).mean()

        l_energy = l_in + l_out

        total = l_ce + self.alpha * l_energy

        loss_dict = {
            'loss_ce':     l_ce.item(),
            'loss_energy': l_energy.item(),
            'loss_in':     l_in.item(),
            'loss_out':    l_out.item(),
            'loss_total':  total.item(),
            'mean_E_in':   E_in.mean().item(),
            'mean_E_out':  E_out.mean().item(),
        }

        return total, loss_dict
