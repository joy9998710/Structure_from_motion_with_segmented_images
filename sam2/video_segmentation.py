import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

matplotlib.use("TkAgg")

#Using GPU
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

video_dir = "../videos/school"

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# frame_idx = 0
# plt.figure(figsize=(9,6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# plt.show()

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)


ann_frame_idx = 0
ann_obj_id = 1

#[x_min, y_min, x_max, y_max]
box = np.array([200, 400, 700, 900], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx = ann_frame_idx,
    obj_id = ann_obj_id,
    box=box
)

# plt.figure(figsize=(9,6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_box(box, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()

video_segments = {}
for out_frame_idx, obj_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

vis_frame_stride = 300
plt.close("all")

output_dir = "../videos/masked_frames"

for out_frame_idx in range(0, len(frame_names)):
    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
    frame = np.array(Image.open(frame_path).convert("RGB"))
    total_mask = np.zeros(frame.shape[:2], dtype=bool)

    for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
        out_mask_np = out_mask.squeeze() if out_mask.ndim == 3 else out_mask
        total_mask |= out_mask_np
    
    masked_frame = np.where(total_mask[..., None], frame, 255).astype(np.uint8)

    output_path = os.path.join(output_dir, f"masked_{frame_names[out_frame_idx]}")
    Image.fromarray(masked_frame).save(output_path)

    # plt.figure(figsize=(6,4))
    # plt.title(f"frame {out_frame_idx}")
    # plt.imshow(masked_frame)
    # plt.axis("off")
    # plt.show()