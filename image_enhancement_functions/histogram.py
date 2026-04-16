# ══════════════════════════════════════════════════════════════════
# 1. HISTOGRAM EQUALISATION (Global)
#
# MATH:
#   p(r)   = h(r) / N              ← normalised histogram (PDF)
#   CDF(r) = Σ p(j) for j=0..r    ← cumulative distribution
#   T(r)   = round(255 · CDF(r))  ← intensity mapping
#
# WHY: Spreads intensities evenly across [0,255].
#      Dark factory images gain global brightness.
# WHEN TO USE: Image is globally too dark or too bright.
# WHEN NOT: Mixed lighting (bright fire + dark worker) → use CLAHE.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np

def histogram_equalise(image: np.ndarray) -> np.ndarray:
    
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)