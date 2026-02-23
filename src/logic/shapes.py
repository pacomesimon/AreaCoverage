import numpy as np
import cv2
from PIL import Image

def create_fov_mask(radius_px, fov_deg, angle_deg, width, height):
    """
    Creates a mask for a Field of View (sector).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (radius_px, radius_px)
    
    # Draw sector on a local small canvas to avoid large memory usage for a single shape template
    canvas_size = radius_px * 2
    temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    start_angle = -fov_deg / 2
    end_angle = fov_deg / 2
    
    cv2.ellipse(temp_mask, center, (radius_px, radius_px), 0, start_angle, end_angle, 255, -1)
    
    return temp_mask

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
