# ══════════════════════════════════════════════════════════════════
# CLAHE — Contrast Limited Adaptive Histogram Equalisation
#
# MATH (3 improvements over global HE):
#
#   a) Per-tile histogram:
#        T_tile(r) = floor(255 · CDF_tile(r))
#
#   b) Clip & redistribute:
#        h_clip(r) = min(h(r), β)
#        E = Σ max(0, h(r) - β)   ← total clipped excess
#        h_final(r) = h_clip(r) + E/L   ← redistribute uniformly
#
#   c) Bilinear interpolation between tiles:
#        T_pixel = (1-a)(1-b)·T_TL + a(1-b)·T_TR
#                + (1-a)b·T_BL  + ab·T_BR
#
# WHY: Fixes two global HE problems:
#        i)  Different lighting zones handled separately
#        ii) Clip limit prevents noise amplification
# WHEN TO USE: Always for SafeSight. Default first choice.
#
# clip_limit: 2.0 = safe default, 4.0+ = aggressive enhancement
# tile_grid:  (8,8) standard, smaller = more local, larger = closer to global
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np

def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_enhanced = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)