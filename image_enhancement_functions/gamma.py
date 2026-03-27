# ══════════════════════════════════════════════════════════════════
# GAMMA CORRECTION
#
# MATH:
#   I_out = (I_in / 255) ^ γ  × 255
#
#   γ < 1  → power < 1 → curve bends upward → dark pixels brighten
#   γ > 1  → power > 1 → curve bends downward → bright pixels darken
#   γ = 1  → linear (identity, no change)
#
# IMPLEMENTATION: Precompute LUT[i] for i ∈ [0,255] → O(1) per pixel
#   LUT[i] = clip(((i/255)^γ) × 255, 0, 255)
#
# WHY: Models nonlinear camera/display response.
#      Dark factory frames need γ < 1 (0.5–0.8 typical).
#      Glare near fire or windows needs γ > 1.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np

def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction via precomputed lookup table.

    Args:
        gamma: < 1 brightens (night/shadow), > 1 darkens (glare)
               Typical SafeSight values: 0.6–0.8 for dark factories
    """
    if gamma == 1.0:
        return image.copy()

    # Build LUT: one corrected value per input intensity
    lut = np.array([
        np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    # Apply: each pixel value replaced by lut[pixel_value]
    return cv2.LUT(image, lut)