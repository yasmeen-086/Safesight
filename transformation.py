def demo(image_path: str, out_dir: str = "output/transforms"):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="image.png")
    parser.add_argument("--out",   type=str, default="output/transforms")
    args = parser.parse_args()
    demo(args.image, args.out)

