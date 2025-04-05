import os
import sqlite3
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 사용할 이미지 이름 지정
target_image_name = "masked_00000.jpg"  # 예: masked_frames_sample/00000.png
database_path = "./sfm_ori_segmented/database.db"
sampled_dir = "./videos/masked_frames_sample"

# 1. DB 연결
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# 2. image_id 조회
cursor.execute("SELECT image_id FROM images WHERE name=?", (target_image_name,))
result = cursor.fetchone()
if result is None:
    raise ValueError(f"Image '{target_image_name}' not found in database.")
image_id = result[0]

# 3. keypoints 조회
cursor.execute("SELECT data FROM keypoints WHERE image_id=?", (image_id,))
keypoints_blob = cursor.fetchone()[0]
keypoints = np.frombuffer(keypoints_blob, dtype=np.float32).reshape(-1, 2)

conn.close()

# 4. 이미지 로드 및 keypoints 시각화
img_path = os.path.join(sampled_dir, target_image_name)
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for (x, y) in keypoints:
    cv2.circle(img_rgb, (int(x), int(y)), 2, (255, 0, 0), -1)

plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.title(f"Features in '{target_image_name}'")
plt.axis('off')
plt.show()
