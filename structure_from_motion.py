import os
import sys
import numpy as np
import open3d as o3d
import shutil
import subprocess

image_dir = "./videos/masked_frames_sample"
output_dir = "./sfm"
database_path = os.path.join(output_dir, "database.db")
os.makedirs(output_dir, exist_ok=True)
sparse_dir = os.path.join(output_dir, "sparse")
os.makedirs(sparse_dir, exist_ok=True)
sparse_model_dir = os.path.join(sparse_dir, "0")
os.makedirs(sparse_model_dir, exist_ok=True)


#Feature extraction with parameter tuning
subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--ImageReader.single_camera", "true",
    "--SiftExtraction.use_gpu", "1",
    "--SiftExtraction.peak_threshold", "0.0000001",
    "--SiftExtraction.edge_threshold", "30"
])

subprocess.run([
    "colmap", "exhaustive_matcher",
    "--database_path", database_path
])

subprocess.run([
    "colmap", "mapper",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--output_path", sparse_dir
])


#Point cloud visualization
sys.path.append(".")

from read_write_model import read_points3D_binary

points3D = read_points3D_binary("./sfm_segmented_only/sparse/0/points3D.bin")

xyz = np.array([p.xyz for p in points3D.values()])
rgb = np.array([p.rgb for p in points3D.values()]) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

print(f"[INFO] Loaded {len(xyz)} points.")
o3d.visualization.draw_geometries([pcd])

