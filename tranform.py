import cv2
import numpy as np

image = cv2.imread("image.png")

src_coords = np.float32([
    [450, 310],  # Top-left
    [820, 310],  # Top-right
    [1150, 750], # Bottom-right
    [120, 750]   # Bottom-left
])

warped_img, H_matrix = get_birdseye_view(image) 

worker_feet = [600, 500]
ground_coord = map_detection_to_ground(worker_feet, H_matrix)

print(f"Worker is at ground position: {ground_coord}")