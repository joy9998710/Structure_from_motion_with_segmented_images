import os
import sys
import numpy as np
import open3d as o3d
import shutil
import subprocess


image_dir = "./videos/masked_frames"
sampled_dir = "./videos/masked_frames_sample"
output_dir = "./sfm_segmented_only"
database_path = os.path.join(output_dir, "database.db")
sparse_dir = os.path.join(output_dir, "sparse")
sparse_model_dir = os.path.join(sparse_dir, "0")
ply_path = os.path.join(sparse_model_dir, "points3D.ply")

# 1. 10의 배수 index 이미지만 복사
os.makedirs(sampled_dir, exist_ok=True)

all_images = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# 복사 안 되어 있을 경우에만 복사 (이미 있다면 스킵)
if not os.listdir(sampled_dir):
    for idx, fname in enumerate(all_images):
        if idx % 5 == 0:
            src = os.path.join(image_dir, fname)
            dst = os.path.join(sampled_dir, fname)
            shutil.copy(src, dst)
    print(f"[INFO] Copied {len(os.listdir(sampled_dir))} sampled images to '{sampled_dir}'")

# 2. SfM 수행
os.makedirs(sparse_dir, exist_ok=True)

subprocess.run([
    "colmap", "feature_extractor",
    "--database_path", database_path,
    "--image_path", sampled_dir,
    "--ImageReader.single_camera", "true"
])

subprocess.run([
    "colmap", "exhaustive_matcher",
    "--database_path", database_path
])

subprocess.run([
    "colmap", "mapper",
    "--database_path", database_path,
    "--image_path", sampled_dir,
    "--output_path", sparse_dir
])



# # 현재 디렉토리에 read_write_model.py가 있을 경우
sys.path.append(".")

from read_write_model import read_points3D_binary

# 포인트 클라우드 불러오기
points3D = read_points3D_binary("./sfm_segmented_only/sparse/0/points3D.bin")

# 포인트 위치 및 색상 추출
xyz = np.array([p.xyz for p in points3D.values()])
rgb = np.array([p.rgb for p in points3D.values()]) / 255.0

# Open3D 포인트 클라우드 시각화
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

print(f"[INFO] Loaded {len(xyz)} points.")
o3d.visualization.draw_geometries([pcd])

