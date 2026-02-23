import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
import pandas as pd
from .shapes import place_shape

class MLP_QNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MLP_QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
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
    
    # State size: 3 values per sensor (x, y, angle)
    state_size = num_sensors * 3
    # Actions: 6 possible shifts per sensor (+x, -x, +y, -y, +angle, -angle)
    num_actions_per_sensor = 6
    num_total_actions = num_sensors * num_actions_per_sensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP_QNetwork(state_size, num_total_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # Initial random configuration
    target_indices = np.argwhere(target_mask > 0)
    if len(target_indices) == 0:
        return
        
    current_config = []
    for _ in range(num_sensors):
        y_px, x_px = target_indices[random.randrange(len(target_indices))]
        current_config.append({'x': float(x_px), 'y': float(y_px), 'angle': random.uniform(0, 360)})

    def get_state(config):
        state = []
        for c in config:
            state.extend([c['x']/w, c['y']/h, c['angle']/360.0])
        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def calculate_coverage(config):
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in config:
            place_shape(mask, shape_template, c['x'], c['y'], c['angle'])
        intersection = np.bitwise_and(mask, target_mask)
        return np.sum(intersection > 0), mask

    total_target_area = np.sum(target_mask > 0)
    best_coverage_pct = -1
    best_state_mask = None
    best_config = [c.copy() for c in current_config]
    
    epsilon = 0.5
    step_size_px = max(5, int(5 * scale)) # 5 meters or scaled
    step_size_angle = 15.0

    for i in range(n_experiments):
        state_t = get_state(current_config)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, num_total_actions - 1)
        else:
            with torch.no_grad():
                q_values = model(state_t)
                action_idx = torch.argmax(q_values).item()
        
        sensor_idx = action_idx // num_actions_per_sensor
        move_type = action_idx % num_actions_per_sensor
        
        # Apply action
        old_config = [c.copy() for c in current_config]
        old_covered_area, _ = calculate_coverage(old_config)
        
        c = current_config[sensor_idx]
        if move_type == 0: c['x'] = min(w, c['x'] + step_size_px)
        elif move_type == 1: c['x'] = max(0, c['x'] - step_size_px)
        elif move_type == 2: c['y'] = min(h, c['y'] + step_size_px)
        elif move_type == 3: c['y'] = max(0, c['y'] - step_size_px)
        elif move_type == 4: c['angle'] = (c['angle'] + step_size_angle) % 360
        elif move_type == 5: c['angle'] = (c['angle'] - step_size_angle) % 360
        
        new_covered_area, current_mask = calculate_coverage(current_config)
        
        # Reward: change in coverage
        reward = (new_covered_area - old_covered_area) / total_target_area
        # Bonus for high coverage
        coverage_pct = (new_covered_area / total_target_area) * 100
        reward += (coverage_pct / 100.0) * 0.1
        
        # Simple Q-learning update
        next_state_t = get_state(current_config)
        with torch.no_grad():
            target_q = reward + 0.3 * torch.max(model(next_state_t)).item()
        
        q_values = model(state_t)
        q_target = q_values.clone()
        q_target[0, action_idx] = target_q
        
        loss = criterion(q_values, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epsilon = max(0.1, epsilon * 0.995)
        
        # Track best
        if coverage_pct > best_coverage_pct:
            best_coverage_pct = coverage_pct
            best_state_mask = current_mask.copy()
            best_config = [c.copy() for c in current_config]
            updated = True
        else:
            updated = False
            
        # Stream results
        if updated or (i % 10 == 0) or (i == n_experiments - 1):
            results_df = pd.DataFrame(best_config)
            results_df['x_m'] = (results_df['x'] / scale).round(2)
            results_df['y_m'] = (results_df['y'] / scale).round(2)
            results_df['angle_deg'] = results_df['angle'].round(1)
            yield best_state_mask, best_coverage_pct, results_df[['x_m', 'y_m', 'angle_deg']], i + 1
