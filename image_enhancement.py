import os
import argparse
import cv2

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
        # ── Auto-trigger demo ──────────────────────────────────────
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