if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str,  help="Path to input image")
    parser.add_argument("--source", type=int,  default=None,
                        help="Webcam index for live demo (e.g. 0)")
    parser.add_argument("--out",    type=str,  default="output/enhancement")
    parser.add_argument("--trigger-demo", action="store_true",
                        help="Run auto-trigger demo on --image")
    args = parser.parse_args()