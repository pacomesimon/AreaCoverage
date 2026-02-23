import numpy as np
import cv2
import pandas as pd
from .shapes import create_fov_mask, create_antenna_mask, place_shape
import random

def run_optimization(target_mask, sensor_type, num_sensors, sensor_range_m, map_width_m, n_experiments):
    """
    Runs N experiments to find the best sensor placement.
    """
    h, w = target_mask.shape
    scale = w / map_width_m  # pixels per meter
    radius_px = int(sensor_range_m * scale)
    
    # Create the base shape template
    if sensor_type == "Camera FOV (90°)":
        shape_template = create_fov_mask(radius_px, 90, 0, w, h)
    elif sensor_type == "Camera FOV (120°)":
        shape_template = create_fov_mask(radius_px, 120, 0, w, h)
    elif sensor_type == "Antenna Lobe":
        shape_template = create_antenna_mask(radius_px, w, h)
    else: # Default Circle/Omni
        shape_template = np.zeros((radius_px*2, radius_px*2), dtype=np.uint8)
        cv2.circle(shape_template, (radius_px, radius_px), radius_px, 255, -1)

    best_coverage_pct = -1
    best_state = None
    best_config = []

    total_target_area = np.sum(target_mask > 0)
    if total_target_area == 0:
        return None, 0, []

    for _ in range(n_experiments):
        current_mask = np.zeros((h, w), dtype=np.uint8)
        current_config = []
        
        for _ in range(num_sensors):
            # Random position and rotation
            target_indices = np.argwhere(target_mask > 0)
            y, x = target_indices[random.randrange(len(target_indices))]
            angle = random.uniform(0, 360)
            
            place_shape(current_mask, shape_template, x, y, angle)
            current_config.append({'x': x, 'y': y, 'angle': angle})
            
        # Calculate coverage: intersection of current_mask and target_mask
        intersection = np.bitwise_and(current_mask, target_mask)
        covered_area = np.sum(intersection > 0)
        coverage_pct = (covered_area / total_target_area) * 100
        
        if coverage_pct > best_coverage_pct:
            best_coverage_pct = coverage_pct
            best_state = current_mask.copy()
            best_config = current_config
            
    # Prepare results for table (convert px to meters)
    results_df = pd.DataFrame(best_config)
    results_df['x_m'] = (results_df['x'] / scale).round(2)
    results_df['y_m'] = (results_df['y'] / scale).round(2)
    results_df['angle_deg'] = results_df['angle'].round(1)
    
    return best_state, best_coverage_pct, results_df[['x_m', 'y_m', 'angle_deg']]
