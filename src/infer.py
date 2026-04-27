"""
Single-sample MAXFUSE inference with full XAI output.
Outputs: predicted family, energy score, Grad-CAM PNG, SHAP bar chart.
"""

import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.maxfuse import MAXFUSE
from models.energy_ood import UNKNOWN_LABEL
from data.binary_to_image import binary_to_image
from data.pe_extractor import extract_pe_features
from data.nca_selector import transform_nca
from explain.gradcam import GradCAMExplainer


MALIMG_FAMILIES = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
    'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A',
    'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3',
    'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N',
    'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]


def load_model_from_checkpoint(ckpt_path: str, cfg: dict, device: str) -> tuple:
    """Load MAXFUSE model and return (model, tau)."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = cfg['model']
    model = MAXFUSE(
        num_classes=m_cfg['num_classes'],
        img_dim=m_cfg['img_dim'],
        num_dim=m_cfg['num_dim'],
        shared_dim=m_cfg['shared_dim'],
        mc_passes=m_cfg['mc_passes'],
        dropout=m_cfg['dropout'],
        use_n1=m_cfg.get('use_n1', True),
        use_n2=m_cfg.get('use_n2', True),
        use_n3=m_cfg.get('use_n3', True),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    tau = ckpt.get('tau', float('inf'))
    model.tau.fill_(tau)
    return model, tau


def infer_single(
    binary_path: str,
    config_path: str,
    checkpoint_path: str,
    nca_model_path: str = 'outputs/nca_model.pkl',
    output_dir:     str = 'outputs/xai',
    family_names:   list = None,
    device:         str = None
) -> dict:
    """
    Run full MAXFUSE inference on a single PE binary.

    Returns dict with:
        family:      predicted family name or 'UNKNOWN'
        class_id:    predicted class index (-1 if unknown)
        confidence:  softmax probability of predicted class
        energy:      energy score E(z)
        is_zero_day: bool
        weights_i:   image branch fusion weight
        weights_n:   numeric branch fusion weight
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if family_names is None:
        family_names = MALIMG_FAMILIES

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tau = load_model_from_checkpoint(checkpoint_path, cfg, device)
    print(f"Model loaded. tau = {tau:.4f}")

    print(f"Processing: {binary_path}")

    # Load input as image (PNG direct load or raw PE binary visualization)
    p = Path(binary_path)
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
        from PIL import Image as PILImage
        img_pil = PILImage.open(str(p)).convert('L').resize((224, 224))
        img_array = np.array(img_pil, dtype=np.float32) / 255.0
    else:
        img_array = binary_to_image(binary_path, size=224)
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).float().to(device)

    # PE features -> NCA
    raw_feats = extract_pe_features(binary_path)
    if raw_feats is None:
        print("[WARN] PE parsing failed; using zero features.")
        raw_feats = np.zeros(508, dtype=np.float32)

    nca_feats = transform_nca(raw_feats.reshape(1, -1), model_path=nca_model_path)
    num_tensor = torch.tensor(nca_feats, dtype=torch.float32).to(device)

    # Inference (Algorithm 1)
    result = model.inference(img_tensor, num_tensor, tau=tau)

    pred_id  = int(result['predictions'][0].item())
    energy   = float(result['energies'][0].item())
    w_i      = float(result['weights_i'][0].item())
    w_n      = float(result['weights_n'][0].item())

    logits   = result['logits'][0]
    probs    = torch.softmax(logits, dim=-1)
    conf     = float(probs.max().item())

    is_ood   = (pred_id == UNKNOWN_LABEL)
    family   = 'UNKNOWN (zero-day alert)' if is_ood else family_names[pred_id]

    print(f"\n{'='*50}")
    print(f"  Prediction:    {family}")
    print(f"  Confidence:    {conf*100:.1f}%")
    print(f"  Energy E(z):   {energy:.4f}  (tau={tau:.4f})")
    print(f"  Zero-day:      {'YES !' if is_ood else 'No'}")
    print(f"  Fusion w_img:  {w_i:.3f}  |  w_num: {w_n:.3f}")
    print(f"{'='*50}\n")

    # Explainability
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(binary_path).stem

    if not is_ood:
        try:
            explainer_cam = GradCAMExplainer(model, device=device)
            explainer_cam.explain(
                img_tensor, num_tensor,
                target_class=pred_id,
                save_path=str(out_dir / f'{stem}_gradcam.png')
            )
        except Exception as e:
            print(f"[WARN] Grad-CAM failed: {e}")

    return {
        'family':      family,
        'class_id':    pred_id,
        'confidence':  conf,
        'energy':      energy,
        'tau':         tau,
        'is_zero_day': is_ood,
        'weights_i':   w_i,
        'weights_n':   w_n,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MAXFUSE single-sample inference")
    parser.add_argument('--binary',     required=True, help='Path to PE binary (.exe)')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--nca_model',  default='outputs/nca_model.pkl')
    parser.add_argument('--output_dir', default='outputs/xai')
    parser.add_argument('--device',     default=None)
    args = parser.parse_args()

    result = infer_single(
        args.binary, args.config, args.checkpoint,
        args.nca_model, args.output_dir, device=args.device
    )
    print(json.dumps(result, indent=2))
