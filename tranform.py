import cv2
import numpy as np

image = cv2.imread("image.png")

def get_birdseye_view(frame):
    src_pts = np.float32([
        [450, 310],  # Top-left
        [820, 310],  # Top-right
        [1150, 750], # Bottom-right
        [120, 750]   # Bottom-left
    ])

    width, height = 500, 600
    dst_pts = np.float32([
        [0, 0],              # Top-left
        [width, 0],          # Top-right
        [width, height],     # Bottom-right
        [0, height]          # Bottom-left
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_image = cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)

    return warped_image, matrix

def map_detection_to_ground(point, H_matrix):
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed_pt = cv2.perspectiveTransform(pt, H_matrix)
    return transformed_pt[0][0].tolist()

warped_img, H_matrix = get_birdseye_view(image) 

worker_feet = [600, 500]
ground_coord = map_detection_to_ground(worker_feet, H_matrix)

print(f"Worker is at ground position: {ground_coord}")