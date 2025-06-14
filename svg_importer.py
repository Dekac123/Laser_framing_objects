from tkinter import filedialog, Tk
from PIL import Image
import numpy as np
from typing import Optional
import io
import cv2
import cairosvg

def import_svg(width: int = 200, height: int = 200) -> Optional[np.ndarray]:
    """Open file dialog and convert selected SVG to OpenCV BGRA image"""
    root: Tk = Tk()
    root.withdraw()

    file_path: str = filedialog.askopenfilename(filetypes=[("SVG Files", "*.svg")])
    if not file_path:
        return None

    png_bytes: bytes = cairosvg.svg2png(url=file_path, output_width=width, output_height=height)
    img: Image.Image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img_array: np.ndarray = np.array(img)
    bgra_img: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)

    return bgra_img
