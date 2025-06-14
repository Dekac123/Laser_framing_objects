import cv2
import numpy as np
from typing import List, Tuple

def correct_perspective_from_aruco(
    image: np.ndarray,
    marker_ids: List[int],
    marker_positions: List[Tuple[float, float]],
    marker_length: float,
    aruco_dict_type: int = cv2.aruco.DICT_5X5_50,
    output_size: Tuple[int, int] = (800, 600)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects given ArUco markers and returns perspective-corrected image and homography matrix.

    :param image: Input image (OpenCV format)
    :param marker_ids: List of expected ArUco IDs (e.g. [0,1,2,3,4,5])
    :param marker_positions: Real-world coordinates for each marker (x, y)
    :param marker_length: Marker size in the real world (in meters or mm)
    :param aruco_dict_type: ArUco dictionary type
    :param output_size: Size of output warped image (width, height)
    :return: (Warped image, homography matrix)
    """
    # Load the specified ArUco dictionary and set up detection parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers in the image
    corners, ids, _ = detector.detectMarkers(image)

    # Raise an error if not enough markers were found
    if ids is None or len(ids) < len(marker_ids):
        raise ValueError("Not all required ArUco markers found.")

    # Prepare lists for matching image points to real-world coordinates
    image_points = []
    world_points = []

    # Create a dictionary for quick lookup of corners by ID
    id_to_corner = {int(i): c for i, c in zip(ids.flatten(), corners)}

    # Match detected markers with expected IDs and get their center points
    for marker_id, world_coord in zip(marker_ids, marker_positions):
        if marker_id not in id_to_corner:
            continue
        c = id_to_corner[marker_id][0]  # Get the 4 corners of the marker
        center = np.mean(c, axis=0)     # Calculate the center of the marker
        image_points.append(center)     # Append image coordinates
        world_points.append(world_coord)  # Append real-world coordinates

    # Need at least 4 points to compute homography
    if len(image_points) < 4:
        raise ValueError("At least 4 valid markers required for perspective correction.")

    # Convert to NumPy arrays
    image_pts = np.array(image_points, dtype=np.float32)
    world_pts = np.array(world_points, dtype=np.float32)

    # Compute homography matrix from image points to real-world points
    H, _ = cv2.findHomography(image_pts, world_pts)

    # Apply the perspective transform to the input image
    warped = cv2.warpPerspective(image, H, output_size)

    # Optional: draw detected markers and center points for visualization
    for marker_id in marker_ids:
        if marker_id in id_to_corner:
            cv2.polylines(image, [id_to_corner[marker_id].astype(int)], True, (0, 255, 0), 2)
            center = np.mean(id_to_corner[marker_id][0], axis=0).astype(int)
            cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)

    return warped, H
