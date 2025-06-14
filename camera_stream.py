import cv2
import numpy as np
from typing import Optional

def get_camera_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Capture a single frame from the webcam"""
    ret: bool
    frame: np.ndarray
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
