"""
Grad-CAM explainability for the image branch.
Highlights byte regions that drove the classification decision.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017.
Library: pytorch-grad-cam (pip install grad-cam)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class _ImageOnlyWrapper(torch.nn.Module):
    """Wraps MAXFUSE so Grad-CAM sees a single-input -> logits interface."""

    def __init__(self, maxfuse_model):
        super().__init__()
        self.m = maxfuse_model
        self.stored_num_feats = None

    def forward(self, img):
        num_feats = self.stored_num_feats
        if num_feats is None:
            num_feats = torch.zeros(img.size(0), 80, device=img.device)
        return self.m.forward(img, num_feats)


class GradCAMExplainer:
    """
    Grad-CAM wrapper for the MAXFUSE image branch (EfficientNet-B0).

    The target layer is the last convolutional block of EfficientNet-B0:
        model.image_encoder.backbone.blocks[-1]
    """

    def __init__(self, model, device: str = 'cuda'):
        """
        Args:
            model: MAXFUSE model instance
            device: 'cuda' or 'cpu'
        """
        self.model  = model.to(device)
        self.device = device

        self.wrapper = _ImageOnlyWrapper(model)
        target_layer = model.image_encoder.backbone.blocks[-1]
        self.cam = GradCAM(
            model=self.wrapper,
            target_layers=[target_layer]
        )

    def explain(
        self,
        img_tensor: torch.Tensor,
        num_feats: torch.Tensor,
        target_class: int = None,
        save_path: str = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            img_tensor: (1, 1, 224, 224) float32 tensor
            num_feats: (1, 80) numeric features
            target_class: Class to explain (None = predicted class)
            save_path: Optional path to save visualisation PNG

        Returns:
            cam_image: (224, 224, 3) uint8 RGB heatmap overlay
        """
        img_tensor = img_tensor.to(self.device)
        num_feats  = num_feats.to(self.device)

        self.wrapper.stored_num_feats = num_feats

        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        grayscale_cam = self.cam(
            input_tensor=img_tensor,
            targets=targets
        )[0]

        rgb_img = img_tensor[0, 0].cpu().numpy()
        rgb_img = np.stack([rgb_img] * 3, axis=-1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cam_image).save(save_path)
            print(f"Grad-CAM saved to {save_path}")

        return cam_image

    def explain_batch(
        self,
        images: torch.Tensor,
        num_feats: torch.Tensor,
        labels: list,
        family_names: list,
        save_dir: str = 'outputs/xai/gradcam'
    ):
        """Generate and save Grad-CAM for a batch."""
        for i in range(len(images)):
            img  = images[i:i+1]
            feat = num_feats[i:i+1]
            label = labels[i]
            name = family_names[label] if label >= 0 else 'UNKNOWN'
            save_path = f"{save_dir}/sample_{i:04d}_{name}.png"
            self.explain(img, feat, target_class=label, save_path=save_path)
