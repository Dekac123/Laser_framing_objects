import sys
import io
import cv2
import numpy as np
import cairosvg
from PIL import Image

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import Qt, QTimer, QPoint

import xml.etree.ElementTree as ET
from typing import Tuple
# def svg_to_qpixmap(path: str, width: int, height: int) -> QPixmap:
#     """Convert an SVG file to a QPixmap using cairosvg."""
#     png_data: bytes = cairosvg.svg2png(url=path, output_width=width, output_height=height)
#     image: Image.Image = Image.open(io.BytesIO(png_data)).convert("RGBA")
#     raw_data: bytes = image.tobytes("raw", "RGBA")
#     qimg: QImage = QImage(raw_data, image.width, image.height, QImage.Format_RGBA8888)
#     return QPixmap.fromImage(qimg)
def get_svg_size(svg_path: str) -> Tuple[int, int]:
    """Extract width and height from SVG file in pixels."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    width_str = root.get("width", "0")
    height_str = root.get("height", "0")

    def parse_dimension(value: str) -> float:
        if value.endswith("px"):
            return float(value[:-2])
        elif value.endswith("mm"):
            # Convert mm to pixels (assuming 96 dpi)
            return float(value[:-2]) * 96 / 25.4
        elif value.endswith("cm"):
            return float(value[:-2]) * 96 / 2.54
        elif value.endswith("pt"):
            return float(value[:-2]) * 96 / 72
        else:
            return float(value)

    width = int(parse_dimension(width_str))
    height = int(parse_dimension(height_str))
    return width, height


def svg_to_qpixmap(svg_path: str) -> QPixmap:
    """Render SVG at its native size and return as QPixmap."""
    width, height = get_svg_size(svg_path)

    # If no size found, fall back to something reasonable
    if width == 0 or height == 0:
        width, height = 400, 400

    png_data = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)
    image = Image.open(io.BytesIO(png_data)).convert("RGBA")
    raw_data = image.tobytes("raw", "RGBA")
    qimg = QImage(raw_data, image.width, image.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

class CameraWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Camera with SVG Overlay")
        self.setGeometry(100, 100, 1200, 900)

        self.label: QLabel = QLabel(self)
        self.label.setFixedSize(1200, 900)
        self.setCentralWidget(self.label)

        # Camera setup
        self.cap: cv2.VideoCapture = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

        self.timer: QTimer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.overlay_pixmap: QPixmap | None = None
        self.overlay_pos: QPoint = QPoint(0, 0)
        self.dragging: bool = False
        self.drag_start_pos: QPoint = QPoint()

    def update_frame(self) -> None:
        """Capture frame and display it with overlay."""
        ret: bool
        frame: np.ndarray
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h: int = frame.shape[0]
        w: int = frame.shape[1]
        ch: int = frame.shape[2]
        bytes_per_line: int = ch * w

        q_img: QImage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        if self.overlay_pixmap:
            painter = QPainter(q_img)
            painter.drawPixmap(self.overlay_pos, self.overlay_pixmap)
            painter.end()

        self.label.setPixmap(QPixmap.fromImage(q_img))

    def keyPressEvent(self, event) -> None:
        """Handle key input for import and quit."""
        if event.key() == Qt.Key_I:
            self.load_svg()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def load_svg(self) -> None:
        """Open file dialog and load SVG as overlay."""
        self.timer.stop()  # Stop camera updates while dialog is open

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open SVG File",
            "",
            "SVG Files (*.svg)"
        )

        if file_name:
            max_w: int = self.label.width()
            max_h: int = self.label.height()
            #self.overlay_pixmap = svg_to_qpixmap(file_name, max_w, max_h)
            self.overlay_pixmap = svg_to_qpixmap(file_name)

            # Center the overlay
            self.overlay_pos = QPoint(
                (max_w - self.overlay_pixmap.width()) // 2,
                (max_h - self.overlay_pixmap.height()) // 2
            )

        self.timer.start(30)  # Resume camera updates

    def mousePressEvent(self, event) -> None:
        if self.overlay_pixmap:
            label_pos = self.label.mapFromGlobal(event.globalPosition().toPoint())
            rect = self.overlay_pixmap.rect().translated(self.overlay_pos)
            if rect.contains(label_pos):
                self.dragging = True
                self.drag_start_pos = label_pos - self.overlay_pos

    def mouseMoveEvent(self, event) -> None:
        if self.dragging:
            label_pos = self.label.mapFromGlobal(event.globalPosition().toPoint())
            self.overlay_pos = label_pos - self.drag_start_pos
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.dragging = False

    def closeEvent(self, event) -> None:
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    window: CameraWindow = CameraWindow()
    window.show()
    sys.exit(app.exec())
