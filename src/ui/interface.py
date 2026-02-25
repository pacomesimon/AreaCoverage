import gradio as gr
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from src.logic.engine import run_optimization
from src.logic.shapes import process_custom_shape
from src.logic.em_logic import run_em_optimization

def process_em_coverage(image_data, radii_m_str, map_width, update_radii, perc_thresh):
    if image_data is None:
        return None, "Error: No image provided.", None
    
    background = image_data["background"]
    layers = image_data["layers"]
    
    if not layers:
        return background, "Error: Please draw the target area in a layer.", None

    bg_np = np.array(background.convert("RGB"))
    h, w = bg_np.shape[:2]

    # Target mask
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        layer_np = np.array(layer.convert("RGBA"))
        mask = layer_np[:, :, 3] > 0
        target_mask[mask] = 255
    
    if np.sum(target_mask) == 0:
        return bg_np, "Error: No target area detected.", None

    try:
        radii_m = [float(r.strip()) for r in radii_m_str.split(',') if r.strip()]
    except:
        return bg_np, "Error: Invalid radii format. Use comma-separated numbers.", None

    # Run EM (Streaming)
    max_em_iter = 50
    for best_mask, coverage_pct, results_df, current_iter in run_em_optimization(
        target_mask, radii_m, float(map_width), 
        max_iter=max_em_iter, update_radii=update_radii, perc_thresh=perc_thresh
    ):
        if best_mask is None:
            continue

        # Visualization
        overlay = bg_np.copy()
        coverage_color = np.array([0, 255, 200], dtype=np.uint8) # Neon Teal for EM
        overlay[best_mask > 0] = coverage_color
        
        scale = w / float(map_width)
        for _, row in results_df.iterrows():
            cx, cy = int(row['x_m'] * scale), int(row['y_m'] * scale)
            # Draw sensor
            cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 1)
            
            # Radius circle
            r_px = int(row['radius_m'] * scale)
            cv2.circle(overlay, (cx, cy), r_px, (255, 255, 255), 2)

        # Combine
        alpha = 0.4
        output_img = cv2.addWeighted(overlay, alpha, bg_np, 1 - alpha, 0)
        
        status = f"{round(coverage_pct, 2)}% Coverage achieved (EM Iteration {current_iter}/{max_em_iter})"
        yield output_img, status, results_df

def process_coverage(image_data, sensor_type, num_sensors, sensor_range, map_width, n_experiments, custom_shape_data, optimization_method):
    if image_data is None:
        return None, "Error: No image provided.", None
    
    background = image_data["background"]
    layers = image_data["layers"]
    
    if not layers:
        yield background, "Error: Please draw the target area in a layer.", None
        return

    # Convert background to numpy
    bg_np = np.array(background.convert("RGB"))
    h, w = bg_np.shape[:2]

    # Combine all layers to form the target mask
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        layer_np = np.array(layer.convert("RGBA"))
        mask = layer_np[:, :, 3] > 0
        target_mask[mask] = 255
    
    if np.sum(target_mask) == 0:
        yield bg_np, "Error: No target area detected.", None
        return

    # Handle Custom Shape Logic
    final_sensor_spec = float(sensor_range)
    if sensor_type == "Custom Shape":
        if custom_shape_data is None:
            yield bg_np, "Error: Please upload a custom shape.", None
            return
        
        scale = w / float(map_width)
        custom_range_px = int(float(sensor_range) * scale)
        
        custom_mask = process_custom_shape(custom_shape_data, sensor_range, map_width)
        if custom_mask is None:
             yield bg_np, "Error: Failed to process shape.", None
             return
        
        ch, cw = custom_mask.shape
        target_size = max(custom_range_px * 2, 4)
        custom_mask = cv2.resize(custom_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        final_sensor_spec = custom_mask

    # Run optimization (Streaming)
    for best_mask, best_pct, results_df, current_iter in run_optimization(
        target_mask, sensor_type, int(num_sensors), final_sensor_spec, 
        float(map_width), int(n_experiments), optimization_method
    ):
        if best_mask is None: continue

        overlay = bg_np.copy()
        coverage_color = np.array([0, 150, 255], dtype=np.uint8) 
        overlay[best_mask > 0] = coverage_color
        
        scale = w / map_width
        for _, row in results_df.iterrows():
            cx, cy = int(row['x_m'] * scale), int(row['y_m'] * scale)
            angle = row['angle_deg']
            cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 0), 1)
            rad = np.radians(angle)
            ex, ey = int(cx + 15 * np.cos(-rad)), int(cy + 15 * np.sin(-rad))
            cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 2)

        contours_tuple = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        alpha = 0.5
        output_img = cv2.addWeighted(overlay, alpha, bg_np, 1 - alpha, 0)
        
        pct_display = f"{round(best_pct, 2)}%" if isinstance(best_pct, (int, float)) else str(best_pct)
        status = f"{pct_display} Coverage achieved (Experiment {current_iter}/{n_experiments})"
        yield output_img, status, results_df

# Design the Premium UI with Custom CSS
custom_css = """
#header { text-align: center; margin-bottom: 2rem; }
.gr-tabs { border: none !important; }
.gr-tab-item { font-weight: 600 !important; }
#run-btn, #em-run-btn {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    border: none !important;
    transition: transform 0.2s;
    color: white !important;
}
#run-btn:hover, #em-run-btn:hover { transform: scale(1.02); }
"""

def create_ui():
    with gr.Blocks(css=custom_css) as demo:
        with gr.Row(elem_id="header"):
            gr.Markdown("# 🛰️ Advanced Coverage Analysis System")
        
        with gr.Tabs() as tabs:
            # TAB 1: EM OPTIMIZATION
            with gr.TabItem("EM Cluster Optimization (GMM)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 1. Define Target Area")
                            em_editor = gr.ImageEditor(
                                label=None, type="pil", sources=["upload", "clipboard"],
                                brush=gr.Brush(colors=["#00FF00"], default_size=25),
                                value = "./assets/images/map.jpg", layers=True, canvas_size=(800, 600)
                            )
                    
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 2. Configure Clusters")
                            radii_input = gr.Textbox(value="50, 75, 75, 100", label="Cluster Radii (m)", placeholder="e.g. 50, 75, 100")
                            em_map_width = gr.Number(value=800, label="Map Width (m)")
                            update_radii = gr.Checkbox(label="Optimize Radii during EM", value=False)
                            perc_thresh = gr.Slider(minimum=0, maximum=100, value=80, label="Rejection Threshold (%)")
                            em_run_btn = gr.Button("🚀 Run EM Optimization", variant="primary", elem_id="em-run-btn")

                with gr.Row():
                    with gr.Column(scale=2):
                        em_output_image = gr.Image(label="Cluster Analysis result")
                    with gr.Column(scale=1):
                        em_coverage_result = gr.Textbox(label="Success Metric", interactive=False)
                        em_results_table = gr.Dataframe(label="Cluster Centers", interactive=False)

            # TAB 2: SPATIAL SEARCH (RL & MONTE CARLO)
            with gr.TabItem("Spatial Search (RL / Monte Carlo)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 1. Define Target Area")
                            search_editor = gr.ImageEditor(
                                label=None, type="pil", sources=["upload", "clipboard"],
                                brush=gr.Brush(colors=["#00FF00"], default_size=25),
                                value = "./assets/images/map.jpg", layers=True, canvas_size=(800, 600)
                            )
                    
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 2. Strategy & Parameters")
                            sensor_type = gr.Dropdown(
                                choices=["Omni-directional (Circle)", "Camera FOV (90°)", "Camera FOV (120°)", "Antenna Lobe", "Custom Shape"],
                                value="Camera FOV (90°)", label="Sensor Pattern"
                            )
                            optimization_method = gr.Dropdown(
                                choices=["Monte Carlo", "Deep Q-Learning"],
                                value="Monte Carlo", label="Optimization Method"
                            )
                        
                        with gr.Accordion("🎨 Custom Shape Definition", open=False):
                            custom_shape_editor = gr.ImageEditor(
                                label="Draw/Upload Shape", type="pil", sources=["upload", "clipboard"],
                                value = "./assets/images/polar.jpg", brush=gr.Brush(colors=["#FFFFFF"], default_size=15),
                                layers=True, canvas_size=(300, 300)
                            )

                        num_sensors = gr.Number(value=5, label="Number of Sensors", precision=0)
                        sensor_range = gr.Number(value=20, label="Standard Sensor Range (m)")
                        search_map_width = gr.Number(value=100, label="Map Width (m)")
                        n_experiments = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Iterations (Simulation Quality)")
                        search_run_btn = gr.Button("🚀 Run Spatial Search", variant="primary", elem_id="run-btn")

                with gr.Row():
                    with gr.Column(scale=2):
                        search_output_image = gr.Image(label="Coverage Heatmap")
                    with gr.Column(scale=1):
                        search_coverage_result = gr.Textbox(label="Success Metric", interactive=False)
                        search_results_table = gr.Dataframe(label="Deployment Map", interactive=False)

        # Event Handlers
        em_run_btn.click(
            process_em_coverage,
            inputs=[em_editor, radii_input, em_map_width, update_radii, perc_thresh],
            outputs=[em_output_image, em_coverage_result, em_results_table]
        )

        search_run_btn.click(
            process_coverage,
            inputs=[
                search_editor, sensor_type, num_sensors, sensor_range, search_map_width, 
                n_experiments, custom_shape_editor, optimization_method
            ],
            outputs=[search_output_image, search_coverage_result, search_results_table]
        )
   
    return demo, custom_css
