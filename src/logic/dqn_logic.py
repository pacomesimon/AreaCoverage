import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
import pandas as pd
from .shapes import place_shape

class MLP_QNetwork(nn.Module):
    def __init__(self, input_size=2, num_outputs=2):
        super(MLP_QNetwork, self).__init__()
        features_n = 256
        self.net = nn.Sequential(
            nn.Linear(input_size, features_n),  
            nn.ReLU(),
            nn.Linear(features_n, features_n//2),
            nn.ReLU(),
            nn.Linear(features_n//2, features_n//4),
            nn.ReLU(),
            nn.Linear(features_n//4, features_n//8),
            nn.ReLU(),
            nn.Linear(features_n//8, features_n//16),
            nn.ReLU(),
            nn.Linear(features_n//16, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

def run_dqn_mlp_optimization(target_mask, sensor_type, num_sensors, shape_template, map_width_m, n_experiments):
    """
    Optimizes sensor placement using an MLP-based DQN.
    State: Normalized [x, y, angle] for each sensor.
    Actions: Discrete shifts for each sensor.
    """
    h, w = target_mask.shape
    scale = w / map_width_m
    
    # The model transforms any (x,y) -> (dx,dy)
    state_size = 2
    num_outputs = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_QNetwork(state_size, num_outputs).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-50)
    criterion = nn.MSELoss()
    
    # Initial random configuration
    target_indices = np.argwhere(target_mask > 0)
    if len(target_indices) == 0:
        return
    
    centroid = np.mean(target_indices, axis=0)
        
    current_config = []
    for _ in range(num_sensors):
        std = np.std(target_indices, axis=0)
        eligible = target_indices[np.all(np.abs(target_indices - centroid) <= (std*1), axis=1)]
        y_px, x_px = eligible[np.random.randint(len(eligible))]
        angle = np.degrees(np.arctan2(-y_px + centroid[0], x_px - centroid[1]))
        current_config.append({'x': float(x_px), 'y': float(y_px), 'angle': float(angle)})

    def get_state(config):
        state = [[c['x']/w, c['y']/h] for c in config]
        return torch.FloatTensor(state).to(device) # Shape: (num_sensors, 2)

    def calculate_metrics(config):
        accumulator = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in config:
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            place_shape(temp_mask, shape_template, c['x'], c['y'], c['angle'])
            accumulator += (temp_mask > 0).astype(np.float32)
            mask = np.bitwise_or(mask, temp_mask)
            
        intersection = np.bitwise_and(mask, target_mask)
        covered_area = np.sum(intersection > 0)
        overlap_area = np.sum(np.maximum(0, accumulator - 1))
        return covered_area, overlap_area, mask

    total_target_area = np.sum(target_mask > 0)
    best_coverage_pct = -1
    best_state_mask = None
    best_config = [c.copy() for c in current_config]
    
    epsilon = 0.5
    step_size_px = max(5, int(5 * scale)) # 5 meters or scaled

    for i in range(n_experiments):
        state_t = get_state(current_config)
        
        # Get displacements from model
        outputs = model(state_t)
        
        # Exploration: Add Gaussian noise
        noise = torch.randn_like(outputs) * epsilon
        applied_actions = outputs + noise
        
        # Convert to numpy for application
        deltas = applied_actions.detach().cpu().numpy() * step_size_px # (num_sensors, 2)
        
        # Apply transformations to ALL sensors
        old_config = [c.copy() for c in current_config]
        old_covered_area, old_overlap, _ = calculate_metrics(old_config)
        
        for idx, c in enumerate(current_config):
            dx, dy = deltas[idx]
            c['x'] = float(np.clip(c['x'] + dx, 0, w))
            c['y'] = float(np.clip(c['y'] + dy, 0, h))
            # Automatically update angle based on new position
            c['angle'] = float(np.degrees(np.arctan2(-c['y'] + centroid[0], c['x'] - centroid[1])))
        
        new_covered_area, new_overlap, current_mask = calculate_metrics(current_config)
        
        # Reward: change in coverage
        reward = (new_covered_area - old_covered_area) / total_target_area
        
        # Penalize intersection (overlap)
        # We penalize the relative change in overlap to encourage dispersion
        overlap_penalty = (new_overlap - old_overlap) / total_target_area
        reward -= overlap_penalty * 0.5 # Penalty weight
        
        # Bonus for high coverage
        coverage_pct = (new_covered_area / total_target_area) * 100
        reward += (coverage_pct / 100.0) * 0.1
        
        # Policy update using reward as signal
        # Loss = -reward * similarity(current_outputs, applied_actions)
        # This reinforces applied_actions that yielded positive rewards
        current_outputs = model(state_t)
        loss = -(reward) * torch.sum((current_outputs - applied_actions.detach())**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epsilon = max(0.0, epsilon * 0.995)
        
        # Track best
        if coverage_pct > best_coverage_pct:
            best_coverage_pct = coverage_pct
            best_state_mask = current_mask.copy()
            best_config = [c.copy() for c in current_config]
            updated = True
        else:
            updated = False
            
        # Stream results: show current and best
        results_df_curr = pd.DataFrame(current_config)
        results_df_curr['x_m'] = (results_df_curr['x'] / scale).round(2)
        results_df_curr['y_m'] = (results_df_curr['y'] / scale).round(2)
        results_df_curr['angle_deg'] = results_df_curr['angle'].round(1)
        
        display_metric = f"{round(coverage_pct, 2)}% (Best: {round(best_coverage_pct, 2)}%)"
        yield current_mask, display_metric, results_df_curr[['x_m', 'y_m', 'angle_deg']], i + 1

    # Final yield: ensure the best overall result is the last thing emitted
    results_df_best = pd.DataFrame(best_config)
    results_df_best['x_m'] = (results_df_best['x'] / scale).round(2)
    results_df_best['y_m'] = (results_df_best['y'] / scale).round(2)
    results_df_best['angle_deg'] = results_df_best['angle'].round(1)
    yield best_state_mask, f"{round(best_coverage_pct, 2)}%", results_df_best[['x_m', 'y_m', 'angle_deg']], n_experiments
