import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image,k=4):
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)

    label_counts = Counter(labels.flatten())
    dominant_color = centers[label_counts.most_common(1)[0][0]]

    return dominant_color

def get_color_name(rgb_color):
    color_names = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0],
        "cyan": [0, 255, 255],
        "magenta": [255, 0, 255],
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "gray": [128, 128, 128],
        "orange": [255, 165, 0],
        "purple": [128, 0, 128],
        "pink": [255, 192, 203]
    }

    min_distance = float('inf')
    closest_color_name = None

    for color_name, color_value in color_names.items():
        distance = np.linalg.norm(np.array(rgb_color) - np.array(color_value))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name