import numpy as np
import cv2
from PIL import Image

def create_fov_mask(radius_px, fov_deg, angle_deg, width, height):
    """
    Creates a mask for a Field of View (sector).
    """
    center = (radius_px, radius_px)
    
    # Draw sector on a local small canvas
    canvas_size = radius_px * 2
    temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    start_angle = -fov_deg / 2
    end_angle = fov_deg / 2
    
    cv2.ellipse(temp_mask, center, (radius_px, radius_px), 0, start_angle, end_angle, 255, -1)
    
    return temp_mask

def process_custom_shape(image_data, range_m, map_width_m):
    """
    Processes a user-drawn or uploaded image into a sensor mask.
    The center of the sensor is assumed to be the center of the image.
    """
    if image_data is None:
        return None
    
    # Handle ImageEditor output or simple Image
    if isinstance(image_data, dict):
        # Merge layers if it's from an editor
        layers = image_data.get("layers", [])
        bg = image_data.get("background", None)
        
        if layers:
            # Use the first layer alpha/content as the shape
            img = layers[0].convert("RGBA")
        elif bg:
            img = bg.convert("RGBA")
        else:
            return None
    else:
        img = image_data.convert("RGBA")

    # Convert to grayscale mask
    np_img = np.array(img)
    if np_img.shape[2] == 4:
        # Use alpha channel if present
        mask = np_img[:, :, 3]
    else:
        mask = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)
    
    # Threshold to binary
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Resize mask based on range_m
    # We want the 'width' of the mask image to represent 2*range_m in real world?
    # Actually, let's just resize it so it fits the radius_px scaling logic.
    return mask

def create_antenna_mask(radius_px, width, height):
    """
    Creates a simple antenna radiation pattern (oval/lobed).
    In a real scenario, this would be a more complex beam pattern.
    Here we use an ellipse as a placeholder for a directional antenna lobe.
    """
    canvas_size = radius_px * 2
    temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    center = (radius_px, radius_px)
    
    # Directional lobe: an elongated ellipse
    cv2.ellipse(temp_mask, center, (radius_px, int(radius_px * 0.4)), 0, 0, 360, 255, -1)
    
    return temp_mask

def rotate_image(image, angle):
    """Rotates an image (mask) around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)

def place_shape(background_mask, shape_mask, x, y, angle):
    """
    Places a shape_mask onto background_mask at (x, y) with rotation.
    x, y are center coordinates.
    """
    rotated_shape = rotate_image(shape_mask, angle)
    sh, sw = rotated_shape.shape
    bh, bw = background_mask.shape
    
    # Calculate bounds
    x_start = int(x - sw // 2)
    y_start = int(y - sh // 2)
    x_end = x_start + sw
    y_end = y_start + sh
    
    # Crop to background bounds
    x_s, y_s = max(0, x_start), max(0, y_start)
    x_e, y_e = min(bw, x_end), min(bh, y_end)
    
    # Crop shape relative to background
    sx_s, sy_s = x_s - x_start, y_s - y_start
    sx_e, sy_e = sx_s + (x_e - x_s), sy_s + (y_e - y_s)
    
    if x_e > x_s and y_e > y_s:
        # bitwise_or to combine coverage
        background_mask[y_s:y_e, x_s:x_e] = np.bitwise_or(background_mask[y_s:y_e, x_s:x_e], rotated_shape[sy_s:sy_e, sx_s:sx_e])
    
    return background_mask
