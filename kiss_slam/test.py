import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.datasets import get_dataset

def extract_and_visualize_icp(config_path, data_dir):
    pipeline = OdometryPipeline(config_path=config_path)
    # Pass the dataloader type directly into get_dataset (e.g., 'kitti', 'ouster', 'bag', etc.)
    dataset = get_dataset(dataloader_name=dataloader_type, data_dir=data_directory, **kwargs)
    
    # Initialize an Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="KISS-SLAM Staircase Debugger", width=1024, height=768)
    
    # Create an empty geometry object to update dynamically
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True

    print(f"{'Frame':<8} | {'Transl. (m)':<12} | {'Rot. (deg)':<12} | {'ICP Error (m)':<15}")
    print("-" * 60)

    for frame_idx in range(len(dataset)):
        raw_points, timestamps = dataset[frame_idx]
        registered_points = pipeline.step(raw_points, timestamps)
        
        # Calculate Pose Metrics
        delta_pose = pipeline.get_last_delta_pose()
        translation = np.linalg.norm(delta_pose[:3, 3])
        rotation = np.linalg.norm(R.from_matrix(delta_pose[:3, :3]).as_euler('xyz', degrees=True))
        
        # Estimate approximate ICP error against the active local map
        map_points = np.asarray(pipeline.local_map.point_cloud())
        if len(map_points) > 0 and len(registered_points) > 0:
            sample = registered_points[np.random.choice(len(registered_points), min(300, len(registered_points)), replace=False)]
            dists = np.linalg.norm(sample[:, None, :] - map_points[None, :, :], axis=-1)
            icp_error = np.mean(np.min(dists, axis=1))
        else:
            icp_error = 0.0

        print(f"{frame_idx:<8} | {translation:<12.4f} | {rotation:<12.4f} | {icp_error:<15.4f}")

        # Update Visualizer Geometry
        pcd.points = o3d.utility.Vector3dVector(registered_points)
        
        # Color Coding: If translation or error spikes (Staircase slip/jerk), turn points RED
        if translation > 0.3 or icp_error > 0.08:
            print(f"--> CRITICAL SLIP DETECTED AT FRAME {frame_idx}!")
            pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(registered_points), 1))) # Red
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.8, 0.2], (len(registered_points), 1))) # Green

        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    extract_and_visualize_icp("config/kiss_icp.yaml", "/path/to/hku_zym.bag")
