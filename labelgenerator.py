import numpy as np
import cv2
import os
from typing import Tuple

class PerspectiveProjectionPseudoLabelGenerator:
    def __init__(self, sequence_folder: str):
        self.sequence_folder = sequence_folder
        self.K_cam2 = np.load(os.path.join(sequence_folder, 'K_cam2.npy'))
        self.T_cam2_velo = np.load(os.path.join(sequence_folder, 'T_cam2_velo.npy'))

    def load_velodyne_points(self, scan_id: str) -> np.ndarray:
        velodyne_path = os.path.join(self.sequence_folder, 'velodyne', f'{scan_id}.bin')
        return np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

    def load_velodyne_labels(self, scan_id: str) -> np.ndarray:
        labels_path = os.path.join(self.sequence_folder, 'velodyne_labels', f'{scan_id}.label')
        return np.fromfile(labels_path, dtype=np.uint32)

    def load_image(self, scan_id: str) -> np.ndarray:
        image_path = os.path.join(self.sequence_folder, 'image_2', f'{scan_id}.png')
        return cv2.imread(image_path)

    def load_sam_segments(self, scan_id: str) -> np.ndarray:
        sam_segments_path = os.path.join(self.sequence_folder, 'sam_segments', f'{scan_id}.png')
        return cv2.imread(sam_segments_path, cv2.IMREAD_GRAYSCALE)

    def project_velodyne_to_image(self, velo_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points_cam = self.T_cam2_velo @ velo_points.T
        projected_points = self.K_cam2 @ points_cam[:3, :]
        projected_points = projected_points / projected_points[2, :]

        # Filter out points that are behind the camera or outside the image
        valid_indices = (projected_points[2, :] > 0)
        projected_points = projected_points[:, valid_indices]

        return projected_points[:2, :].T.astype(np.int32), valid_indices

    def generate_pseudo_labels(self, scan_id: str) -> np.ndarray:
        velo_points = self.load_velodyne_points(scan_id)
        labels = self.load_velodyne_labels(scan_id)
        sam_segments = self.load_sam_segments(scan_id)

        projected_points, valid_indices = self.project_velodyne_to_image(velo_points)
        valid_labels = labels[valid_indices]

        # Create a 2D array to hold label counts for each pixel
        label_counts = np.zeros((sam_segments.shape[0], sam_segments.shape[1], np.max(labels)+1), dtype=np.int32)

        # For each projected point, increment the count for its label in the corresponding pixel
        for point, label in zip(projected_points, valid_labels):
            if 0 <= point[0] < sam_segments.shape[1] and 0 <= point[1] < sam_segments.shape[0]:
                label_counts[point[1], point[0], label] += 1

        # For each segment, determine the most common label
        pseudo_labels = np.zeros_like(sam_segments)
        for segment_id in np.unique(sam_segments):
            if segment_id == 0:
                continue  # Skip background

            segment_mask = sam_segments == segment_id
            segment_label_counts = np.sum(label_counts[segment_mask], axis=0)
            dominant_label = np.argmax(segment_label_counts)
            pseudo_labels[segment_mask] = dominant_label

        return pseudo_labels

    def visualize_pseudo_labels(self, pseudo_labels: np.ndarray) -> np.ndarray:
        cmap = create_semkitti_label_colormap()
        return cmap[pseudo_labels]

def process_all_sequences(data_root: str):
    # Get all sequence folders
    sequence_folders = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    for sequence_folder in sequence_folders:
        print(f"Processing sequence: {sequence_folder}")

        # Initialize the pseudo-label generator for the current sequence
        generator = PerspectiveProjectionPseudoLabelGenerator(sequence_folder)

        # Get all scan IDs in the current sequence
        scan_ids = [f.split('.')[0] for f in os.listdir(os.path.join(sequence_folder, 'velodyne')) if f.endswith('.bin')]

        for scan_id in scan_ids:
            print(f"  Processing scan: {scan_id}")

            # Generate pseudo-labels for the current scan
            pseudo_labels = generator.generate_pseudo_labels(scan_id)

            # Save the pseudo-labels (example: save as PNG)
            pseudo_labels_path = os.path.join(sequence_folder, 'pseudo_labels', f'{scan_id}.png')
            os.makedirs(os.path.join(sequence_folder, 'pseudo_labels'), exist_ok=True)
            cv2.imwrite(pseudo_labels_path, pseudo_labels)

            # Visualize and save the pseudo-labels overlay
            image = generator.load_image(scan_id)
            visualized_labels = generator.visualize_pseudo_labels(pseudo_labels)
            overlay = cv2.addWeighted(image, 0.5, visualized_labels, 0.5, 0)
            overlay_path = os.path.join(sequence_folder, 'pseudo_labels_overlay', f'{scan_id}.png')
            os.makedirs(os.path.join(sequence_folder, 'pseudo_labels_overlay'), exist_ok=True)
            cv2.imwrite(overlay_path, overlay)

def create_semkitti_label_colormap(): 
    """Creates a label colormap used in SEMANTICKITTI segmentation benchmark.

    Returns:
        A colormap for visualizing segmentation results in BGR format.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [245, 150, 100]       # "car"
    colormap[2] = [245, 230, 100]       # "bicycle"
    colormap[3] = [150, 60, 30]         # "motorcycle"
    colormap[4] = [180, 30, 80]         # "truck"
    colormap[5] = [255, 0, 0]           # "other-vehicle"
    colormap[6] = [30, 30, 255]         # "person"
    colormap[7] = [200, 40, 255]        # "bicyclist"
    colormap[8] = [90, 30, 150]         # "motorcyclist"
    colormap[9] = [255, 0, 255]         # "road"
    colormap[10] = [255, 150, 255]      # "parking"
    colormap[11] = [75, 0, 75]          # "sidewalk"
    colormap[12] = [75, 0, 175]         # "other-ground"
    colormap[13] = [0, 200, 255]        # "building"
    colormap[14] = [50, 120, 255]       # "fence"
    colormap[15] = [0, 175, 0]          # "vegetation"
    colormap[16] = [0, 60, 135]         # "trunk"
    colormap[17] = [80, 240, 150]       # "terrain"
    colormap[18] = [150, 240, 255]      # "pole"
    colormap[19] = [0, 0, 255]          # "traffic-sign"
    return colormap

if __name__ == "__main__":
    data_root = "data_semantickitti"  # Root directory containing all sequence folders

    process_all_sequences(data_root)