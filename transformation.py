

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out",   type=str, default="output/transforms")
    args = parser.parse_args()
    demo(args.image, args.out)

