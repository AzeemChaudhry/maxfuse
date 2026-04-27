"""
MAXFUSE — Full model assembly.
Combines ImageEncoder, NumericEncoder, CrossModalAttention (N1),
DynamicFusionModule (N2), and energy OOD head (N3).

Implements Algorithm 1 from the MAXFUSE paper.
"""

import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .numeric_encoder import NumericEncoder
from .cross_modal_attention import CrossModalAttention
from .dynamic_fusion import mc_dropout_passes, dynamic_weights, fuse
from .energy_ood import energy_score, reject_or_classify, UNKNOWN_LABEL


def _enable_dropout_only(module: nn.Module) -> None:
    """Set only Dropout layers to train mode, leaving BatchNorm in eval mode."""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


class MAXFUSE(nn.Module):
    """
    Full MAXFUSE model.

    Forward pass (training):
        img, num_feats -> [encoders] -> [N1: CrossModalAttn] ->
        [simple avg fusion] -> classifier -> 25-class logits

    Inference:
        Same up to N1, then:
        [N2: MC-Dropout -> dynamic weights] -> weighted fusion -> classifier ->
        [N3: energy score] -> family label or UNKNOWN

    Args:
        num_classes: Number of known malware families (25 for Malimg)
        img_dim: Image encoder output dimension (256)
        num_dim: Numeric encoder output dimension (128)
        shared_dim: Shared attention dimension (128)
        num_heads: Attention heads (4)
        mc_passes: MC-Dropout forward passes for N2 (10)
        dropout: Dropout rate throughout model (0.3)
        use_n1: Enable cross-modal attention
        use_n2: Enable dynamic fusion (MC-Dropout)
        use_n3: Enable energy OOD rejection
    """

    def __init__(
        self,
        num_classes: int = 25,
        img_dim: int = 256,
        num_dim: int = 128,
        shared_dim: int = 128,
        num_heads: int = 4,
        mc_passes: int = 10,
        dropout: float = 0.3,
        use_n1: bool = True,
        use_n2: bool = True,
        use_n3: bool = True,
    ):
        super().__init__()
        self.use_n1 = use_n1
        self.use_n2 = use_n2
        self.use_n3 = use_n3
        self.mc_passes = mc_passes
        self.num_classes = num_classes

        # -- Encoders --
        self.image_encoder   = ImageEncoder(out_dim=img_dim, dropout=dropout)
        self.numeric_encoder = NumericEncoder(out_dim=num_dim, dropout=dropout)

        # -- N1: Cross-Modal Attention --
        if use_n1:
            self.cross_attn = CrossModalAttention(
                img_dim=img_dim, num_dim=num_dim,
                shared_dim=shared_dim, num_heads=num_heads
            )
            fusion_dim = shared_dim
        else:
            self.proj_i_fallback = nn.Linear(img_dim, shared_dim)
            self.proj_n_fallback = nn.Linear(num_dim, shared_dim)
            fusion_dim = shared_dim

        # -- Classifier head --
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # -- OOD threshold (set post-training via calibrate_threshold) --
        self.register_buffer('tau', torch.tensor(float('inf')))

    # ---- Encoding helpers ----

    def encode(self, img: torch.Tensor, num_feats: torch.Tensor) -> tuple:
        """Run both branch encoders. Returns (v_i, v_n)."""
        v_i = self.image_encoder(img)
        v_n = self.numeric_encoder(num_feats)
        return v_i, v_n

    def attend(self, v_i: torch.Tensor, v_n: torch.Tensor) -> tuple:
        """Apply N1 cross-modal attention (or fallback projection)."""
        if self.use_n1:
            return self.cross_attn(v_i, v_n)
        else:
            return self.proj_i_fallback(v_i), self.proj_n_fallback(v_n)

    # ---- Training forward (single pass, no MC sampling) ----

    def forward(self, img: torch.Tensor, num_feats: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        Returns raw logits for CE loss computation.
        """
        v_i, v_n         = self.encode(img, num_feats)
        v_hat_i, v_hat_n = self.attend(v_i, v_n)
        z = 0.5 * v_hat_i + 0.5 * v_hat_n
        return self.classifier(z)

    # ---- Full inference (Algorithm 1) ----

    @torch.no_grad()
    def inference(
        self,
        img: torch.Tensor,
        num_feats: torch.Tensor,
        tau: float = None
    ) -> dict:
        """
        Full MAXFUSE inference: N1 -> N2 (MC-Dropout) -> N3 (energy OOD).
        Implements Algorithm 1.

        Args:
            img: (B, 1, 224, 224) image tensor
            num_feats: (B, 80) numeric feature tensor
            tau: Energy threshold; if None uses self.tau from calibration

        Returns:
            dict with keys:
                'predictions': (B,) int tensor; -1 = UNKNOWN
                'logits':      (B, num_classes) raw classifier outputs
                'energies':    (B,) energy scores
                'weights_i':   (B, 1) image branch fusion weights
                'weights_n':   (B, 1) numeric branch fusion weights
        """
        threshold = tau if tau is not None else self.tau.item()

        v_i, v_n         = self.encode(img, num_feats)
        v_hat_i, v_hat_n = self.attend(v_i, v_n)

        if self.use_n2:
            # Keep BatchNorm in eval mode; only activate Dropout for stochastic passes
            self.image_encoder.eval()
            _enable_dropout_only(self.image_encoder)
            self.numeric_encoder.eval()
            _enable_dropout_only(self.numeric_encoder)

            def img_fn(x):
                vi = self.image_encoder(x)
                vh, _ = self.attend(vi, v_n)
                return vh

            def num_fn(x):
                vn = self.numeric_encoder(x)
                _, vh = self.attend(v_i, vn)
                return vh

            _, var_i = mc_dropout_passes(img_fn, img, T=self.mc_passes)
            _, var_n = mc_dropout_passes(num_fn, num_feats, T=self.mc_passes)

            # Restore full eval mode after MC passes
            self.image_encoder.eval()
            self.numeric_encoder.eval()

            w_i, w_n = dynamic_weights(var_i, var_n)
            z = fuse(v_hat_i, v_hat_n, w_i, w_n)
        else:
            z = 0.5 * v_hat_i + 0.5 * v_hat_n
            w_i = torch.full((img.size(0), 1), 0.5, device=img.device)
            w_n = w_i.clone()

        logits = self.classifier(z)

        if self.use_n3:
            preds, energies = reject_or_classify(logits, threshold)
        else:
            preds    = logits.argmax(dim=-1)
            energies = energy_score(logits)

        return {
            'predictions': preds,
            'logits':      logits,
            'energies':    energies,
            'weights_i':   w_i,
            'weights_n':   w_n,
        }

    # ---- OOD threshold calibration ----

    def calibrate_threshold(self, val_loader, device: str = 'cuda') -> float:
        """
        Calibrate the energy threshold tau on the validation set (FPR95 point).
        Only in-distribution validation samples are used.

        Sets self.tau in-place and returns the value.
        """
        self.eval()
        all_energies = []

        with torch.no_grad():
            for img, num_feats, labels in val_loader:
                img, num_feats = img.to(device), num_feats.to(device)
                v_i, v_n         = self.encode(img, num_feats)
                v_hat_i, v_hat_n = self.attend(v_i, v_n)
                z = 0.5 * v_hat_i + 0.5 * v_hat_n
                logits = self.classifier(z)
                E = energy_score(logits)
                all_energies.append(E.cpu())

        all_energies = torch.cat(all_energies)
        tau = float(torch.quantile(all_energies, 0.95))
        self.tau.fill_(tau)
        print(f"[Calibration] tau set to {tau:.4f}")
        return tau

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
