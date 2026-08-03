#!/usr/bin/env python3
"""预处理流程演示脚本。
每一步生成一个独立文件，记录完整数据管线。

图片: data/self-made/iamges/20260721_155055_11741_R_Raw.png
Mask:  data/self-made/mask/20260721_155055_11741_R_Raw.png
"""

from pathlib import Path
import numpy as np
from PIL import Image

# ============================================================
# 配置
# ============================================================
IMAGE_PATH = "data/woodscape/rgbImages/0074_MVL.png"
MASK_PATH  = "data/woodscape/rgbLabels/0074_MVL.png"
MASK_TYPE  = "rgb_4color"   # woodscape: 绿→干净, 蓝+红→异常, 黑→忽略
IMG_SIZE   = 224
GRID_SIZE  = 8
OUT_DIR    = Path("outputs/preprocess_demo/woodscape")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("预处理流程: self-made 数据集")
print(f"图片: {IMAGE_PATH}")
print(f"Mask: {MASK_PATH}")
print("=" * 60)

# ================================================================
# Step 1: 加载原始图像和原始 Mask
# ================================================================
print("\n[Step 1] 加载原始图像和原始 Mask...")
image_raw = np.array(Image.open(IMAGE_PATH).convert("RGB"))
mask_raw  = np.array(Image.open(MASK_PATH).convert("RGB"))

print(f"  图像尺寸: {image_raw.shape} (H×W×C)")
print(f"  Mask尺寸: {mask_raw.shape}")
colors = np.unique(mask_raw.reshape(-1, 3), axis=0)
print(f"  Mask唯一颜色: {colors.tolist()}")

# 保存 Step 1
Image.fromarray(image_raw).save(str(OUT_DIR / "01_image_raw.png"))
Image.fromarray(mask_raw).save(str(OUT_DIR / "01_mask_raw.png"))

# ================================================================
# Step 2: 二值化 — 原始 Mask → binary_mask + ignore_mask
# ================================================================
print(f"\n[Step 2] 二值化: mask_type={MASK_TYPE}...")

if MASK_TYPE == "red_black":
    # self-made: 红(255,0,0)→异常, 黑(0,0,0)→干净, 全图参与
    is_red   = (mask_raw[:,:,0] > 200) & (mask_raw[:,:,1] < 50) & (mask_raw[:,:,2] < 50)
    binary_mask = is_red.astype(np.float32)
    ignore_mask = np.zeros(mask_raw.shape[:2], dtype=bool)
    n_dirty = int(is_red.sum())

elif MASK_TYPE == "rgb_4color":
    # woodscape rgbLabels: 绿→忽略, 黑→干净, 蓝+红→异常
    is_green = (mask_raw[:,:,1] > 200) & (mask_raw[:,:,0] < 50) & (mask_raw[:,:,2] < 50)
    is_red   = (mask_raw[:,:,0] > 200) & (mask_raw[:,:,1] < 50) & (mask_raw[:,:,2] < 50)
    is_blue  = (mask_raw[:,:,2] > 200) & (mask_raw[:,:,0] < 50) & (mask_raw[:,:,1] < 50)
    is_black = (mask_raw[:,:,0] == 0) & (mask_raw[:,:,1] == 0) & (mask_raw[:,:,2] == 0)

    binary_mask = (is_red | is_blue).astype(np.float32)  # 红+蓝→异常=1
    ignore_mask = is_green                                # 绿→忽略
    n_dirty = int((is_red | is_blue).sum())
    n_clean = int(is_black.sum())
    n_ignore = int(is_green.sum())
    print(f"  绿色(忽略): {n_ignore} ({n_ignore/(mask_raw.shape[0]*mask_raw.shape[1])*100:.2f}%)")
    print(f"  蓝色(透明脏污): {int(is_blue.sum())}")
    print(f"  红色(不透明脏污): {int(is_red.sum())}")
    print(f"  黑色(干净): {n_clean}")

total = mask_raw.shape[0] * mask_raw.shape[1]
if MASK_TYPE == "red_black":
    n_clean = total - n_dirty
print(f"  异常像素: {n_dirty} ({n_dirty/total*100:.2f}%)")
print(f"  干净像素: {n_clean} ({n_clean/total*100:.2f}%)")
print(f"  忽略像素: {int(ignore_mask.sum())}")
print(f"  binary_mask dtype={binary_mask.dtype}, 值域={{{binary_mask.min(), binary_mask.max()}}}")

# 保存 Step 2 — binary_mask 转 RGB 可视化 (红=异常, 黑=干净)
mask_viz = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
mask_viz[binary_mask > 0.9] = [255, 0, 0]  # 异常=红
# 干净=0.0 保持黑色(0,0,0)
Image.fromarray(mask_viz).save(str(OUT_DIR / "02_binary_mask.png"))

# 同时保存 binary_mask 原始数值 (单通道 PNG, 0/1)
Image.fromarray((binary_mask * 255).astype(np.uint8)).save(
    str(OUT_DIR / "02_binary_mask_raw.png"))

# ================================================================
# Step 3: Resize 224×224
# ================================================================
print("\n[Step 3] Resize 224×224...")
from torchvision.transforms import functional as F

image_pil = Image.fromarray(image_raw)
# binary_mask 偏移: {0,1} → {1,2} (兼容 PIL uint8)
mask_pil = Image.fromarray((binary_mask + 1).astype(np.uint8))

image_resized = F.resize(image_pil, [IMG_SIZE, IMG_SIZE],
                         interpolation=F.InterpolationMode.BILINEAR)
mask_resized = F.resize(mask_pil, [IMG_SIZE, IMG_SIZE],
                        interpolation=F.InterpolationMode.NEAREST)

# mask 恢复偏移: {1,2} → {0,1}
mask_resized_np = np.array(mask_resized).astype(np.float32) - 1.0

# 忽略mask 同步 resize
ignore_pil = Image.fromarray(ignore_mask)
ignore_resized = np.array(
    ignore_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))

print(f"  image: {np.array(image_resized).shape}")
print(f"  mask:  {mask_resized_np.shape}")
print(f"  ignore: {ignore_resized.shape}, True数={ignore_resized.sum()}")

# 保存 Step 3
Image.fromarray(np.array(image_resized)).save(str(OUT_DIR / "03_image_resized.png"))
mask_viz_03 = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
mask_viz_03[mask_resized_np > 0.9] = [255, 0, 0]
Image.fromarray(mask_viz_03).save(str(OUT_DIR / "03_mask_resized.png"))

# ================================================================
# Step 4: 数据增强
# ================================================================
print("\n[Step 4] 数据增强...")
import random, torch

img_aug = image_resized
msk_aug = mask_resized

# --- 4a. Horizontal Flip ---
do_flip = True  # 固定执行, 可复现
if do_flip:
    img_aug = F.hflip(img_aug)
    msk_aug = F.hflip(msk_aug)
    print("  4a Horizontal Flip: True")

Image.fromarray(np.array(img_aug)).save(str(OUT_DIR / "04a_flip_image.png"))
msk_flip_np = np.array(msk_aug).astype(np.float32) - 1.0
flip_viz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
flip_viz[msk_flip_np > 0.9] = [255, 0, 0]
Image.fromarray(flip_viz).save(str(OUT_DIR / "04a_flip_mask.png"))

# --- 4b. Brightness ---
brightness_factor = 1.0 + random.uniform(-0.3, 0.3)
print(f"  4b Brightness: ×{brightness_factor:.3f}")
img_aug = F.adjust_brightness(img_aug, brightness_factor=brightness_factor)
Image.fromarray(np.array(img_aug)).save(str(OUT_DIR / "04b_brightness_image.png"))

# --- 4c. Contrast ---
contrast_factor = 1.0 + random.uniform(-0.3, 0.3)
print(f"  4c Contrast: ×{contrast_factor:.3f}")
img_aug = F.adjust_contrast(img_aug, contrast_factor=contrast_factor)
Image.fromarray(np.array(img_aug)).save(str(OUT_DIR / "04c_contrast_image.png"))

# --- 4d. Edge Occlusion ---
do_occlusion = True  # 固定执行
if do_occlusion:
    img_np = np.array(img_aug)
    h, w = img_np.shape[:2]
    margin = random.randint(h // 6, h // 3)
    edge = random.choice(['top', 'bottom', 'left', 'right'])
    if edge == 'top':
        img_np[:margin, :, :] = 0
    elif edge == 'bottom':
        img_np[-margin:, :, :] = 0
    elif edge == 'left':
        img_np[:, :margin, :] = 0
    else:
        img_np[:, -margin:, :] = 0
    img_aug = Image.fromarray(img_np)
    print(f"  4d Edge Occlusion: {edge}, {margin}px")

Image.fromarray(np.array(img_aug)).save(str(OUT_DIR / "04d_occlusion_image.png"))
# mask 不受遮挡影响, 与 flip 后相同
msk_aug_np = np.array(msk_aug).astype(np.float32) - 1.0
aug_viz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
aug_viz[msk_aug_np > 0.9] = [255, 0, 0]
Image.fromarray(aug_viz).save(str(OUT_DIR / "04d_final_mask.png"))

# ================================================================
# Step 5: 生成 8×8 Tile Label (mask 侧 — 在转 Tensor 之前)
# ================================================================
print("\n[Step 5] 生成 8×8 Tile Label...")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.tile_label import mask_to_tile_label

# msk_aug_np 是增强后的 mask (numpy float32, 值域 {0,1})
tile_label = mask_to_tile_label(
    msk_aug_np,
    grid_size=GRID_SIZE,
    threshold=0.1,
    binary=False,
    use_gaussian_weight=True,
    gamma=0.7,
    smooth_sigma=0,
)
tile_ignore = mask_to_tile_label(
    ignore_resized.astype(np.float32),
    grid_size=GRID_SIZE,
    threshold=0.5,
    binary=False,
    use_gaussian_weight=False,
    gamma=1.0,
    smooth_sigma=0.0,
) > 0.5

coverage = msk_aug_np.mean()

print(f"  tile_label: {tile_label.shape}, 值域=[{tile_label.min():.4f}, {tile_label.max():.4f}]")
print(f"  tile_ignore: True数={tile_ignore.sum()}")
print(f"  coverage: {coverage:.4f} ({coverage*100:.1f}%)")
print(f"  Tile > 0.5: {(tile_label > 0.5).sum()}/{GRID_SIZE*GRID_SIZE}")

# 保存 Step 5 — Tile 网格 (原图叠加网格+数值)
from metrics.visualize import save_grid_overlay
save_grid_overlay(np.array(img_aug), tile_label, str(OUT_DIR / "05_tile_grid.png"), img_size=IMG_SIZE)

# 保存 Step 5 — Tile 热力图
from metrics.visualize import save_heatmap
save_heatmap(tile_label, str(OUT_DIR / "05_tile_heatmap.png"), img_size=IMG_SIZE)

# 保存 Step 5 — CSV
import csv
csv_path = OUT_DIR / "05_tile_label.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["coverage", f"{coverage:.6f}"])
    writer.writerow(["tile_ignored", f"{int(tile_ignore.sum())}"])
    writer.writerow([])
    writer.writerow(["row", "col", "tile_label", "tile_ignore"])
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            writer.writerow([i, j, f"{tile_label[i,j]:.6f}", f"{bool(tile_ignore[i,j])}"])

print("  → 已保存: 05_tile_grid.png, 05_tile_heatmap.png, 05_tile_label.csv")

# ================================================================
# Step 6: 转 Tensor (图像侧 — 最后一步，准备模型输入)
# ================================================================
print("\n[Step 6] 转 Tensor...")
img_tensor = F.to_tensor(img_aug)                               # [3, 224, 224]
msk_tensor = torch.from_numpy(np.array(msk_aug)).long() - 1      # 恢复: {1,2}→{0,1}
msk_final = msk_tensor.numpy().astype(np.float32)                 # [224, 224]

print(f"  image tensor: {list(img_tensor.shape)}, dtype={img_tensor.dtype}")
print(f"  mask tensor:  {list(msk_tensor.shape)}, dtype={msk_tensor.dtype}")
print(f"  image 值域: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
print(f"  mask 唯一值: {np.unique(msk_final)}")

# 保存 Step 6 — image tensor 可视化
img_viz = (img_tensor.numpy().transpose(1,2,0) * 255).astype(np.uint8)
Image.fromarray(img_viz).save(str(OUT_DIR / "06_image_tensor.png"))

# 保存 Step 6 — mask tensor 可视化
msk_viz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
msk_viz[msk_final > 0.9] = [255, 0, 0]
Image.fromarray(msk_viz).save(str(OUT_DIR / "06_mask_tensor.png"))

print("  → 已保存: 06_image_tensor.png, 06_mask_tensor.png")

# ================================================================
# 总结
# ================================================================
print(f"\n{'='*60}")
print("输出文件:")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        kb = f.stat().st_size / 1024
        print(f"  {f.name} ({kb:.1f} KB)")
print(f"{'='*60}")
print("最终模型输入:")
print(f"  image:       torch.float32 {list(img_tensor.shape)}")
print(f"  tile_label:  torch.float32 {list(tile_label.shape)}")
print(f"  tile_ignore: torch.bool    {list(tile_ignore.shape)}")
print(f"  coverage:    torch.float32 [1] = {coverage:.4f}")
print(f"{'='*60}")
