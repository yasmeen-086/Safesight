# ══════════════════════════════════════════════════════════════════
# UNSHARP MASK — Sharpening
#
# MATH:
#   detail  = I_original - I_blurred          ← high-frequency edges
#   I_sharp = I_original + λ · detail
#           = (1+λ) · I_original - λ · I_blurred
#
#   At λ=1, equivalent kernel:
#       K = [ 0  -1   0 ]
#           [-1   5  -1 ]
#           [ 0  -1   0 ]
#   (centre = 1+4λ = 5, neighbours = -λ = -1)
#
# WHY: Helmet dome edges, jacket border lines, and safety sign
#      text all need sharp edges for HOG feature extraction and
#      YOLO bounding box accuracy.
#
# ORDER: Always sharpen LAST — after CLAHE and gamma.
#        Sharpening before enhancement amplifies noise.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np

def sharpen(image: np.ndarray,
            amount: float = 1.0,
            blur_ksize: int = 5) -> np.ndarray:
    
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=0)

    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, gamma=0)
