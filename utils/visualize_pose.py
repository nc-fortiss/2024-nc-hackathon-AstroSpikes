import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import json
import os


def to_float_array(x, expected_len=None):
    """Parse string representation of array from CSV."""
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    arr = np.fromstring(s.replace(",", " "), sep=" ", dtype=np.float32)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {arr.size}: {x}")
    return arr


def project_points(q, r, K):
    """Project 3D coordinate frame to 2D image coordinates."""
    points = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)
    rotV = np.expand_dims(Rotation.from_quat(q).as_rotvec(), axis=1)
    image_points, _ = cv2.projectPoints(points, rotV, r, K, distCoeffs=np.zeros(5))
    
    origin = tuple(map(int, image_points[0].ravel()))
    x_axis = tuple(map(int, image_points[1].ravel()))
    y_axis = tuple(map(int, image_points[2].ravel()))
    z_axis = tuple(map(int, image_points[3].ravel()))
    return origin, x_axis, y_axis, z_axis


def load_pose_from_csv(filename, csv_path):
    """Load pose prediction (r_xyz, q_xyzw) for a specific image."""
    df = pd.read_csv(csv_path)
    row = df.loc[df["filename"] == filename]
    if row.empty:
        raise ValueError(f"Filename '{filename}' not found in {csv_path}")
    row = row.iloc[0]
    r_xyz = to_float_array(row["r_xyz"], expected_len=3)
    q_xyzw = to_float_array(row["q_xyzw"], expected_len=4)
    return r_xyz, q_xyzw


def plot_image_with_pose(
    filename: str,
    image_path: str,
    predictions_dir: str,
    camera_json_path: str,
    output_path: str = None,
    show: bool = True
) -> None:
    """
    Visualize pose estimation axes overlaid on an image.
    
    Args:
        filename: Image filename (e.g., 'img405_RT130.png')
        image_path: Path to directory containing the image
        predictions_dir: Directory containing kpts_predictions_pose_metrics.csv
        camera_json_path: Path to camera.json with camera intrinsics
        output_path: Optional path to save the output image
        show: Whether to display the image with matplotlib
    """
    # Load camera intrinsics
    with open(camera_json_path, "r") as f:
        data = json.load(f)
        K = np.array(data["cameraMatrix"])
    
    # Load pose predictions
    csv_path = os.path.join(predictions_dir, "kpts_predictions_pose_metrics.csv")
    r_xyz, q_xyzw = load_pose_from_csv(filename, csv_path)
    
    # Load image
    img_full_path = os.path.join(image_path, filename)
    if not os.path.exists(img_full_path):
        raise FileNotFoundError(f"Image not found: {img_full_path}")
    
    img_bgr = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {img_full_path}")
    
    # Project 3D axes to 2D
    origin, x_axis, y_axis, z_axis = project_points(q_xyzw, r_xyz, K)
    origin = tuple(np.round(origin).astype(int))
    x_axis = tuple(np.round(x_axis).astype(int))
    y_axis = tuple(np.round(y_axis).astype(int))
    z_axis = tuple(np.round(z_axis).astype(int))
    
    # Draw axes (BGR colors for cv2)
    PINK = (255, 64, 255)   # X-axis
    LIME = (40, 255, 120)   # Y-axis
    CYAN = (255, 255, 0)    # Z-axis
    
    cv2.line(img_bgr, origin, x_axis, PINK, 4, cv2.LINE_AA)
    cv2.line(img_bgr, origin, y_axis, LIME, 4, cv2.LINE_AA)
    cv2.line(img_bgr, origin, z_axis, CYAN, 4, cv2.LINE_AA)
    
    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Display
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Pose Visualization: {filename}")
        plt.tight_layout()
        plt.show()
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, img_bgr)
        print(f"Saved to: {output_path}")


def plot_multiple_images_with_pose(
    filenames: list,
    base_path: str,
    predictions_dir: str,
    camera_json_path: str,
    output_path: str = None,
    show: bool = True,
    event_representation: str = None,
    akida_version: str = None
) -> None:
    """
    Visualize multiple images with pose axes in a 3x3 grid.
    
    Args:
        filenames: List of 9 image filenames (e.g., ['img005_RT258.png', ...])
        base_path: Base directory containing trajectory folders (RT folders)
        predictions_dir: Directory containing kpts_predictions_pose_metrics.csv
        camera_json_path: Path to camera.json with camera intrinsics
        output_path: Optional path to save the output image
        show: Whether to display the image with matplotlib
        event_representation: Name of event representation (e.g., 'LNES', 'Event Frame')
        akida_version: Akida version (e.g., 'V1', 'V2')
    """
    if len(filenames) != 9:
        raise ValueError(f"Expected 9 filenames, got {len(filenames)}")
    
    # Load camera intrinsics
    with open(camera_json_path, "r") as f:
        K = np.array(json.load(f)["cameraMatrix"])
    
    # Load pose predictions CSV once
    csv_path = os.path.join(predictions_dir, "kpts_predictions_pose_metrics.csv")
    df_poses = pd.read_csv(csv_path)
    
    # Create 3x3 subplot with reduced spacing
    fig, axes = plt.subplots(3, 3, figsize=(16, 10.9))
    axes = axes.flatten()
    
    # Add main title
    if event_representation and akida_version:
        fig.suptitle(f"{event_representation} - Akida {akida_version}", 
                     fontsize=18, fontweight='bold', y=0.98)
    
    for idx, filename in enumerate(filenames):
        # Extract trajectory folder name (e.g., "img005_RT258.png" -> "RT258")
        traj_name = filename.split("_")[1].split(".")[0]
        img_path = os.path.join(base_path, traj_name, filename)
        
        # Load pose and speed score
        row = df_poses.loc[df_poses["filename"] == filename]
        if row.empty:
            print(f"Warning: {filename} not found in predictions")
            axes[idx].axis('off')
            continue
        
        r_xyz = to_float_array(row.iloc[0]["r_xyz"], expected_len=3)
        q_xyzw = to_float_array(row.iloc[0]["q_xyzw"], expected_len=4)
        speed_score = row.iloc[0]["speed_score"]
        
        # Load and process image
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: Failed to load {img_path}")
            axes[idx].axis('off')
            continue
        
        # Project and draw axes
        origin, x_axis, y_axis, z_axis = project_points(q_xyzw, r_xyz, K)
        
        cv2.line(img_bgr, tuple(np.round(origin).astype(int)), 
                 tuple(np.round(x_axis).astype(int)), (255, 64, 255), 4, cv2.LINE_AA)
        cv2.line(img_bgr, tuple(np.round(origin).astype(int)), 
                 tuple(np.round(y_axis).astype(int)), (40, 255, 120), 4, cv2.LINE_AA)
        cv2.line(img_bgr, tuple(np.round(origin).astype(int)), 
                 tuple(np.round(z_axis).astype(int)), (255, 255, 0), 4, cv2.LINE_AA)
        
        # Display
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        
        # Add title with filename and speed score on one line
        title = f"{filename}  |  SPEED: {speed_score:.4f}"
        axes[idx].set_title(title, fontsize=10, pad=10)
    
    # Minimize spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.96, bottom=0.02, left=0.02, right=0.98)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    if show:
        plt.show()


if __name__ == "__main__":
    # # Example: Single image
    # plot_image_with_pose(
    #     filename="img405_RT130.png",
    #     image_path="/path/to/images/RT130",
    #     predictions_dir="./predictions",
    #     camera_json_path="./predictions/camera.json",
    #     output_path="./output.png",
    #     show=True
    # )
    
    # Example: 3x3 grid (3 good, 3 mid, 3 bad)
    # filenames = [
    #     'img124_RT164.png',
    #     'img315_RT078.png',
    #     'img351_RT254.png',
    #     'img292_RT172.png',
    #     'img281_RT285.png',
    #     'img183_RT280.png',
    #     'img531_RT137.png',
    #     'img252_RT147.png',
    #     'img283_RT147.png'
    # ]
    filenames = [
        'img052_RT045.png',
        'img297_RT073.png',
        'img421_RT069.png',
        'img123_RT099.png',
        'img398_RT204.png',
        'img245_RT070.png',
        'img458_RT217.png',
        'img578_RT138.png',
        'img000_RT095.png'
    ]
    
    plot_multiple_images_with_pose(
        filenames=filenames,
        base_path="/Volumes/T7/SPADES_frames/two_d_histogram",
        # predictions_dir="/Users/jost/Data/astrospikes/Astrospike/10_27_2025/share_astro_results/v2/float/mobilenet_kpt_v2x_lnes_20251002_2001/results",
        predictions_dir="/Users/jost/Data/astrospikes/Astrospike/10_27_2025/share_astro_results/v2/float/mobilenet_kpt_v2x_two_d_histogram_20251002_1754/results_5",
        camera_json_path="/Users/jost/Jost/Code/2024-nc-hackathon-spades/30_07_25/predictions/camera.json",
        output_path="/Users/jost/Jost/Code/2024-nc-hackathon-spades/outputs/grids/two_d_histogram.png",
        show=True,
        event_representation="Two D Histogram",
        akida_version="V2"
    )

