import gradio as gr
import numpy as np
import cv2
from PIL import Image
from src.logic.engine import run_optimization
from src.logic.shapes import process_custom_shape

def process_coverage(image_data, sensor_type, num_sensors, sensor_range, map_width, n_experiments, custom_shape_data, optimization_method):
    if image_data is None:
        return None, 0, None
    
    background = image_data["background"]
    layers = image_data["layers"]
    composite = image_data.get("composite")
    
    if not layers:
        return background, 0, "Error: Please draw the target area in a layer using the brush."

    # Convert background to numpy
    bg_np = np.array(background.convert("RGB"))
    h, w = bg_np.shape[:2]

    # Combine all layers to form the target mask
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        layer_np = np.array(layer.convert("RGBA"))
        # Use alpha channel to define mask
        mask = layer_np[:, :, 3] > 0
        target_mask[mask] = 255
    
    if np.sum(target_mask) == 0:
        return bg_np, 0, "Error: No target area detected. Use the brush to highlight the zone to cover."

    # Handle Custom Shape Logic
    final_sensor_spec = float(sensor_range)
    if sensor_type == "Custom Shape":
        if custom_shape_data is None:
            return bg_np, 0, "Error: Please upload or draw a custom shape in the accordion."
        
        scale = w / float(map_width)
        custom_range_px = int(float(sensor_range) * scale)
        
        # Process shape to binary mask
        custom_mask = process_custom_shape(custom_shape_data, sensor_range, map_width)
        if custom_mask is None:
             return bg_np, 0, "Error: Failed to process custom shape."
        
        # Resize custom mask to match custom_range_px (radius of influence)
        # We assume the custom mask's larger dimension represents 2 * sensor_range
        ch, cw = custom_mask.shape
        target_size = custom_range_px * 2
        # Avoid zero size
        target_size = max(target_size, 4)
        
        custom_mask = cv2.resize(custom_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        final_sensor_spec = custom_mask # Pass the mask instead of range

    # Run optimization (Streaming)
    for best_mask, best_pct, results_df, current_iter in run_optimization(
        target_mask, 
        sensor_type, 
        int(num_sensors), 
        final_sensor_spec, 
        float(map_width), 
        int(n_experiments),
        optimization_method
    ):
        if best_mask is None:
            continue

        # Visualize result: Overlay best_mask on background
        overlay = bg_np.copy()
        
        # 1. Show the Coverage Area (Blueish Glow)
        coverage_color = np.array([0, 150, 255], dtype=np.uint8) # Premium Blue
        overlay[best_mask > 0] = coverage_color
        
        # 2. Draw Sensor Centers and Directions
        scale = w / map_width
        
        for _, row in results_df.iterrows():
            cx, cy = int(row['x_m'] * scale), int(row['y_m'] * scale)
            angle = row['angle_deg']
            
            # Draw sensor point
            cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 0), 1)
            
            # Draw orientation line
            rad = np.radians(angle)
            ex = int(cx + 15 * np.cos(-rad))
            ey = int(cy + 15 * np.sin(-rad))
            cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 2)

        # 3. Show the Target Area outline in Neon Green
        contours_tuple = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1]
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # Blend
        alpha = 0.5
        output_img = cv2.addWeighted(overlay, alpha, bg_np, 1 - alpha, 0)

        # Yield results for streaming
        pct_display = f"{round(best_pct, 2)}%" if isinstance(best_pct, (int, float)) else str(best_pct)
        status = f"{pct_display} Coverage achieved (Experiment {current_iter}/{n_experiments})"
        yield output_img, status, results_df

# Design the Premium UI with Custom CSS
custom_css = """
#header {
    text-align: center;
}

#run-btn {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    border: none !important;
    transition: transform 0.2s;
}
#run-btn:hover {
    transform: scale(1.02);
}
"""

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row(elem_id="header"):
            gr.Markdown("## Spatial Planning & Coverage Analysis")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("#### 🗺️ Map & Target Definition")
                    editor = gr.ImageEditor(
                        label=None,
                        type="pil",
                        sources=["upload", "clipboard"],
                        brush=gr.Brush(colors=["#00FF00"], default_size=25),
                        value = "./assets/images/map.jpg",
                        layers=True,
                        canvas_size=(800, 600)
                    )
                    gr.Markdown("*Tip: Use the brush to paint the 'Target Coverage Area' on a layer.*")
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### ⚙️ Simulation Settings")
                    sensor_type = gr.Dropdown(
                        choices=["Omni-directional (Circle)", "Camera FOV (90°)", "Camera FOV (120°)", "Antenna Lobe", "Custom Shape"],
                        value="Camera FOV (90°)",
                        label="Sensor Pattern"
                    )
                    
                    optimization_method = gr.Dropdown(
                        choices=["Monte Carlo", "Deep Q-Learning"],
                        value="Monte Carlo",
                        label="Optimization Method"
                    )
                
                with gr.Accordion("🎨 Custom Shape Definition", open=False):
                    gr.Markdown("Define your own radiation pattern. The sensor center is the center of the image.")
                    custom_shape_editor = gr.ImageEditor(
                        label="Draw/Upload Shape",
                        type="pil",
                        sources=["upload", "clipboard"],
                        value = "./assets/images/polar.jpg",
                        brush=gr.Brush(colors=["#FFFFFF"], default_size=15),
                        layers=True,
                        canvas_size=(300, 300)
                    )

                num_sensors = gr.Number(value=5, label="Number of Sensors", precision=0)
                sensor_range = gr.Number(value=20, label="Standard Sensor Range (m)")
                map_width = gr.Number(value=100, label="Total Map Width (m)")
                n_experiments = gr.Slider(minimum=10, maximum=1000, value=100, step=10, label="Simulation Quality (N Experiments)")
                
                run_btn = gr.Button("Run Simulation", variant="primary", elem_id="run-btn")

        with gr.Row():
            with gr.Column(scale=2):
                output_image = gr.Image(label="Coverage Heatmap")
            with gr.Column(scale=1):
                coverage_result = gr.Textbox(label="Success Metric", interactive=False)
                results_table = gr.Dataframe(label="Deployment Map (Meters)", interactive=False)

        run_btn.click(
            process_coverage,
            inputs=[
                editor, sensor_type, num_sensors, sensor_range, map_width, 
                n_experiments, custom_shape_editor, optimization_method
            ],
            outputs=[output_image, coverage_result, results_table]
        )
   
    return demo, custom_css
