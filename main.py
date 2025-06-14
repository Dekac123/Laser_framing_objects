import cv2
import numpy as np
from typing import Optional, Tuple
from camera_stream import get_camera_frame
from svg_importer import import_svg

# Global state
overlay_img: Optional[np.ndarray] = None
overlay_pos: Tuple[int, int] = (100, 100)
dragging: bool = False

def mouse_callback(event: int, x: int, y: int, flags: int, param: None) -> None:
    """Handle mouse drag to move the overlay image"""
    global dragging, overlay_pos, overlay_img
    if overlay_img is None:
        return

    img_h: int = overlay_img.shape[0]
    img_w: int = overlay_img.shape[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        if overlay_pos[0] <= x <= overlay_pos[0] + img_w and overlay_pos[1] <= y <= overlay_pos[1] + img_h:
            dragging = True
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        overlay_pos = (x, y)

def draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw the overlay image at current position"""
    if overlay_img is None:
        return frame

    x: int
    y: int
    x, y = overlay_pos
    h: int = overlay_img.shape[0]
    w: int = overlay_img.shape[1]

    if y + h > frame.shape[0] or x + w > frame.shape[1] or x < 0 or y < 0:
        return frame  # Out of bounds

    alpha_overlay: np.ndarray = overlay_img[:, :, 3] / 255.0
    alpha_background: np.ndarray = 1.0 - alpha_overlay

    for c in range(0, 3):
        frame[y:y+h, x:x+w, c] = (
            alpha_overlay * overlay_img[:, :, c] +
            alpha_background * frame[y:y+h, x:x+w, c]
        )

    return frame

def main() -> None:
    """Main function to run the GUI and integrate modules"""
    global overlay_img, overlay_pos

    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 1000, 800)
    cv2.setMouseCallback("Camera", mouse_callback)

    while True:
        frame: Optional[np.ndarray] = get_camera_frame(cap)
        if frame is None:
            break

        if overlay_img is not None:
            frame = draw_overlay(frame)

        cv2.imshow("Camera", frame)

        key: int = cv2.waitKey(1)
        if key == ord("i"):  # Press 'i' to import SVG
            new_overlay: Optional[np.ndarray] = import_svg()
            if new_overlay is not None:
                overlay_img = new_overlay
                overlay_pos = (
                    frame.shape[1] // 2 - overlay_img.shape[1] // 2,
                    frame.shape[0] // 2 - overlay_img.shape[0] // 2,
                )
        elif key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


