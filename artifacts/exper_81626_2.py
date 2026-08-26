import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple

# =====================================================================
# 1. SNR Calculation Helper
# =====================================================================
def calculate_snr_and_mse(w_orig: torch.Tensor, w_recon: torch.Tensor) -> Tuple[float, float]:
    """
    Calculates Signal-to-Noise Ratio (SNR) in dB and Mean Squared Error (MSE).

    SNR (dB) = 10 * log10( Var(W_orig) / MSE(W_orig, W_recon) )
    """
    w_orig_f = w_orig.detach().cpu().float()
    w_recon_f = w_recon.detach().cpu().float()

    var_orig = torch.var(w_orig_f).item()
    mse = torch.mean((w_orig_f - w_recon_f) ** 2).item()

    if mse == 0 or np.isnan(mse):
        return float('inf'), 0.0
    if var_orig == 0:
        return 0.0, mse

    snr_db = 10.0 * np.log10(var_orig / (mse + 1e-12))
    return snr_db, mse

# =====================================================================
# 2. Tensor Categorization
# =====================================================================
def classify_projection_type(tensor_name: str) -> str:
    """Categorizes layer weights by functional block."""
    name = tensor_name.lower()
    if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"]):
        return "Attention Projections"
    elif any(k in name for k in ["gate_proj", "up_proj", "down_proj", "mlp"]):
        return "MLP Projections"
    elif any(k in name for k in ["embed_tokens", "lm_head"]):
        return "Embeddings & Head"
    elif any(k in name for k in ["norm", "layernorm"]):
        return "Normalization"
    else:
        return "Other Tensors"

# =====================================================================
# 3. Model SNR Evaluation Loop
# =====================================================================
def run_snr_experiment(
    weight_dict: Dict[str, torch.Tensor],
    reconstruct_fn,
    tile_size: int = 256
) -> Dict[str, Dict[str, list]]:
    """
    Passes weight tensors through the encoding/decoding function and
    computes layer-wise and category-wise SNR statistics.
    """
    metrics = defaultdict(lambda: {"snr_list": [], "mse_list": [], "names": []})

    print("=" * 65)
    print(f"{'Tensor Name':<40} | {'Type':<18} | {'SNR (dB)':<8} | {'MSE':<10}")
    print("=" * 65)

    for name, w_orig in weight_dict.items():
        if w_orig.ndim < 2:
            continue  # Skip 1D biases and layer norm vectors

        category = classify_projection_type(name)

        # Reconstruct full weight matrix using your decoder pipeline
        # (Pass delta delta_W or absolute weights based on system mode)
        with torch.no_grad():
            w_recon = reconstruct_fn(w_orig)

        snr_db, mse = calculate_snr_and_mse(w_orig, w_recon)

        metrics[category]["snr_list"].append(snr_db)
        metrics[category]["mse_list"].append(mse)
        metrics[category]["names"].append(name)

        # Print first few or low SNR warnings
        status = "⚠️ LOW" if snr_db < 20 else " OK "
        print(f"{name[:38]:<40} | {category:<18} | {snr_db:6.2f} dB {status} | {mse:.2e}")

    print("=" * 65)
    return metrics

# =====================================================================
# 4. Visualization & Summary Dashboard
# =====================================================================
def plot_snr_summary(metrics: Dict[str, Dict[str, list]]):
    """Plots category-wise average SNR and per-layer reconstruction fidelity."""
    categories = list(metrics.keys())
    avg_snrs = [np.mean(metrics[cat]["snr_list"]) for cat in categories]
    min_snrs = [np.min(metrics[cat]["snr_list"]) for cat in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar Chart: Mean vs Min SNR by Category
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, avg_snrs, width, label='Mean SNR (dB)', color='#4C72B0')
    ax1.bar(x + width/2, min_snrs, width, label='Min SNR (dB)', color='#DD8452')
    ax1.axhline(y=20.0, color='red', linestyle='--', label='Divergence Threshold (20 dB)')

    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Reconstruction Quality by Matrix Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Scatter Plot: Individual SNR Distribution across Tensors
    for cat in categories:
        snrs = metrics[cat]["snr_list"]
        ax2.scatter([cat] * len(snrs), snrs, alpha=0.7, label=cat)

    ax2.axhline(y=20.0, color='red', linestyle='--', label='20 dB Cutoff')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Per-Tensor SNR Distribution')
    ax2.set_xticklabels(categories, rotation=15)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
