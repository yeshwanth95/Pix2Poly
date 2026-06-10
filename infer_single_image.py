import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import argparse
import sys

from polygon_inference import PolygonInference
from utils import log

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment_path", help="path to experiment folder to evaluate")
args = parser.parse_args()

def main():
    # Load image from stdin
    image_data = sys.stdin.buffer.read()

    # Initialize inference
    inference = PolygonInference(args.experiment_path)
    
    # Get inference results
    polygons_list = inference.infer(image_data, debug=True)

    # Decode image for visualization
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        log("Failed to load image from stdin", "ERROR")
        return

    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create figure with exact image dimensions
    plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Plot merged result
    vis_image_merged = image.copy()
    formatted_contours = [np.array(cnt).reshape(-1, 1, 2).astype(np.int32) for cnt in polygons_list]
    cv2.drawContours(vis_image_merged, formatted_contours, -1, (0, 255, 0), 1)
    
    # Draw dots at vertices for merged result
    for contour in formatted_contours:
        for point in contour:
            x, y = point[0]
            # Draw 2x2 red square.
            y_min = max(0, y-1)
            y_max = min(height, y+1)
            x_min = max(0, x-1)
            x_max = min(width, x+1)
            vis_image_merged[y_min:y_max, x_min:x_max] = [255, 0, 0]
    
    # Convert BGR to RGB for correct display in matplotlib
    vis_image_merged_rgb = cv2.cvtColor(vis_image_merged, cv2.COLOR_BGR2RGB)
    plt.imshow(vis_image_merged_rgb)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save visualization
    output_path = os.path.join(f"visualization.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    log(f"Saved main visualization to {output_path}")

    # Print polygons to stdout
    print(json.dumps(polygons_list))

if __name__ == "__main__":
    main() 