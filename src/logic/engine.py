import numpy as np
import cv2
import pandas as pd
from .shapes import create_fov_mask, create_antenna_mask, place_shape, process_custom_shape
from .dqn_logic import run_dqn_mlp_optimization
import random

def run_optimization(target_mask, sensor_type, num_sensors, sensor_range_m, map_width_m, n_experiments, method="Monte Carlo"):
    """
    Runs N experiments to find the best sensor placement.
    """
    h, w = target_mask.shape
    scale = w / map_width_m  # pixels per meter
    # Create the base shape template
    if sensor_type == "Custom Shape" and sensor_range_m is not None:
        shape_template = sensor_range_m # Mask passed directly
        radius_px = shape_template.shape[0] // 2
    else:
        radius_px = int(sensor_range_m * scale)
        if sensor_type == "Camera FOV (90°)":
            shape_template = create_fov_mask(radius_px, 90, 0, w, h)
        elif sensor_type == "Camera FOV (120°)":
            shape_template = create_fov_mask(radius_px, 120, 0, w, h)
        elif sensor_type == "Antenna Lobe":
            shape_template = create_antenna_mask(radius_px, w, h)
        else: # Default Circle/Omni
            shape_template = np.zeros((radius_px*2, radius_px*2), dtype=np.uint8)
            cv2.circle(shape_template, (radius_px, radius_px), radius_px, 255, -1)

    if method == "Deep Q-Learning":
        yield from run_dqn_mlp_optimization(target_mask, sensor_type, num_sensors, shape_template, map_width_m, n_experiments)
        return

    best_coverage_pct = -1
    best_state = None
    best_config = []

    total_target_area = np.sum(target_mask > 0)
    if total_target_area == 0:
        return None, 0, []

    for i in range(n_experiments):
        current_mask = np.zeros((h, w), dtype=np.uint8)
        current_config = []
        
        for _ in range(num_sensors):
            # Random position and rotation
            target_indices = np.argwhere(target_mask > 0)
            y, x = target_indices[random.randrange(len(target_indices))]
            centroid = np.mean(target_indices, axis=0)
            angle = np.degrees(np.arctan2(-y + centroid[0], x - centroid[1]))
            
            place_shape(current_mask, shape_template, x, y, angle)
            current_config.append({'x': x, 'y': y, 'angle': angle})
            
        # Calculate coverage: intersection of current_mask and target_mask
        intersection = np.bitwise_and(current_mask, target_mask)
        covered_area = np.sum(intersection > 0)
        coverage_pct = (covered_area / total_target_area) * 100
        
        updated = False
        if coverage_pct > best_coverage_pct:
            best_coverage_pct = coverage_pct
            best_state = current_mask.copy()
            best_config = current_config
            updated = True
        
        # Yield intermediate results frequently (e.g., every 5 iterations or on update)
        if updated or (i % 5 == 0) or (i == n_experiments - 1):
            results_df = pd.DataFrame(best_config)
            if not results_df.empty:
                results_df['x_m'] = (results_df['x'] / scale).round(2)
                results_df['y_m'] = (results_df['y'] / scale).round(2)
                results_df['angle_deg'] = results_df['angle'].round(1)
                yield best_state, best_coverage_pct, results_df[['x_m', 'y_m', 'angle_deg']], i + 1
            else:
                yield best_state, best_coverage_pct, pd.DataFrame(), i + 1
