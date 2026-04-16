import os
import argparse
import cv2
import numpy as np

from image_enhancement_functions.gamma import gamma_correction
from image_enhancement_functions.sharpen import sharpen
from image_enhancement_functions.CLAHE import apply_clahe
from image_enhancement_functions.histogram import histogram_equalise

def analyse_frame(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean         = float(np.mean(gray))
    std          = float(np.std(gray))
    shadow_ratio = float(np.sum(gray < 50) / gray.size)

    laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    return {
        "mean":         round(mean, 2),
        "std":          round(std, 2),
        "shadow_ratio": round(shadow_ratio, 3),
        "blur_score":   round(blur_score, 2),
    }

def get_enhancement_trigger(frame: np.ndarray) -> str:
    """
    Analyses the frame statistics and returns which enhancement
    should be applied.

    Thresholds (tune these for your specific factory cameras):
        mean > 190          → "GAMMA_DARKEN"
        blur_score < 100    → "SHARPEN"
        shadow_ratio > 0.30 → "CLAHE"
        std < 40            → "HISTOGRAM_EQ"
        mean < 85           → "GAMMA_BRIGHTEN"
        else                → "NONE"

    """
    stats = analyse_frame(frame)

    if stats["mean"] > 190:
        return "GAMMA_DARKEN"

    if stats["shadow_ratio"] > 0.30:
        return "CLAHE"

    if stats["mean"] < 85:
        return "GAMMA_BRIGHTEN"

    if stats["std"] < 30:
        return "HISTOGRAM_EQ"

    if stats["blur_score"] < 100 and stats["std"] >= 30:
        return "SHARPEN"

    if stats["std"] < 35:
        return "HISTOGRAM_EQ"

    return "NONE"

def apply_trigger(frame: np.ndarray,verbose: bool = False) -> np.ndarray:
    trigger = get_enhancement_trigger(frame)

    enhancement_map = {
        "GAMMA_DARKEN":   lambda f: gamma_correction(f, gamma=2.0),
        "SHARPEN":        lambda f: sharpen(f, amount=1.2, blur_ksize=5),
        "CLAHE":          lambda f: apply_clahe(f, clip_limit=2.0),
        "HISTOGRAM_EQ":   lambda f: histogram_equalise(f),
        "GAMMA_BRIGHTEN": lambda f: gamma_correction(f, gamma=0.65),
        "NONE":           lambda f: f,
    }

    if verbose:
        stats = analyse_frame(frame)
        print(f"[Trigger] {trigger:20s} | "
              f"mean={stats['mean']:.1f}  "
              f"std={stats['std']:.1f}  "
              f"shadow={stats['shadow_ratio']:.2f}  "
              f"blur={stats['blur_score']:.1f}")

    return enhancement_map[trigger](frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str,  help="Path to input image")
    parser.add_argument("--source", type=int,  default=None,
                        help="Webcam index for live demo (e.g. 0)")
    parser.add_argument("--out",    type=str,  default="output/enhancement")
    parser.add_argument("--trigger-demo", action="store_true",
                        help="Run auto-trigger demo on --image")
    args = parser.parse_args()

    if args.trigger_demo and args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {args.image}")

        stats   = analyse_frame(img)
        trigger = get_enhancement_trigger(img)
        result  = apply_trigger(img, verbose=True)

        print(f"\nFrame stats : {stats}")
        print(f"Trigger     : {trigger}")

        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(f"{args.out}/trigger_original.jpg", img)
        cv2.imwrite(f"{args.out}/trigger_enhanced.jpg", result)