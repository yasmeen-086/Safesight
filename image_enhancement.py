import os
import argparse
import cv2
import numpy as np

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