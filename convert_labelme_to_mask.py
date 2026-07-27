#!/usr/bin/env python3
"""
Labelme JSON → 二值 mask PNG 批量转换脚本
===========================================
功能：
  1. 读取 Labelme 6.3.1 JSON 文件中的 polygon 标注
  2. 将 "anomaly" 标签的区域填充为 1，背景为 0
  3. 输出单通道 uint8 PNG mask
  4. 同时生成 RGB 可视化图（anomaly=红色，背景=黑色）

输入：
  - JSON 文件夹：包含 Labelme 标注的 .json 文件
  - 每个 JSON 中 shapes[].label == "anomaly" 的多边形会被转为 mask

输出：
  - mask/ 目录：单通道 PNG，像素值 ∈ {0, 1}
  - vis/  目录：RGB 可视化图，红色=anomaly，黑色=background
"""

import json
import os
import sys
import numpy as np
import cv2
from pathlib import Path


# ============================================================
# 路径配置 — 根据实际情况修改
# ============================================================
JSON_DIR = "data/self/json"          # Labelme JSON 文件夹
MASK_OUTPUT_DIR = "data/self/mask"   # 二值 mask 输出文件夹
VIS_OUTPUT_DIR = "data/self/vis"     # 可视化图输出文件夹


def json_to_mask(json_path: str, mask_output_dir: str, vis_output_dir: str) -> bool:
    """
    将单个 Labelme JSON 转换为二值 mask PNG + 可视化图。

    参数:
        json_path:       JSON 文件完整路径
        mask_output_dir: mask PNG 输出目录
        vis_output_dir:  可视化图输出目录

    返回:
        True  转换成功
        False 转换失败（无 anomaly 标注或文件异常）
    """
    # 1. 读取 JSON -------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_height = data["imageHeight"]
    image_width = data["imageWidth"]
    shapes = data.get("shapes", [])

    # 2. 创建全零 mask (背景=0) ----------------------------------------
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 3. 遍历 shapes，只处理 label == "anomaly" 的 polygon ------------
    anomaly_count = 0
    for shape in shapes:
        if shape.get("label") != "anomaly":
            continue  # 跳过非 anomaly 标签（当前数据集不存在，但保留健壮性）
        if shape.get("shape_type") != "polygon":
            continue  # 跳过非 polygon 类型

        points = shape["points"]
        # 将浮点坐标转为整数 numpy 数组，cv2.fillPoly 要求 shape: (N, 1, 2)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 填充 polygon 区域为 1 (anomaly)
        cv2.fillPoly(mask, [pts], color=1)
        anomaly_count += 1

    # 如果没有 anomaly 标注，仍然生成全黑 mask（符合实际情况） ----------
    if anomaly_count == 0:
        print(f"  ⚠ 未找到 anomaly 标注: {os.path.basename(json_path)}")

    # 4. 保存 mask PNG -----------------------------------------------
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    mask_path = os.path.join(mask_output_dir, f"{base_name}.png")
    cv2.imwrite(mask_path, mask)  # 单通道 uint8，值为 0 或 1

    # 5. 生成可视化图 (RGB) -------------------------------------------
    # 背景黑色(0,0,0)，anomaly 红色(255,0,0)
    vis = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    vis[mask == 1] = [0, 0, 255]  # BGR 格式: 红色 = (0, 0, 255)

    vis_path = os.path.join(vis_output_dir, f"{base_name}.png")
    cv2.imwrite(vis_path, vis)

    return True


def main():
    """
    主函数：批量处理 JSON 文件夹中的所有 .json 文件。
    """
    # 获取脚本所在目录作为项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    json_dir = os.path.join(script_dir, JSON_DIR)
    mask_dir = os.path.join(script_dir, MASK_OUTPUT_DIR)
    vis_dir = os.path.join(script_dir, VIS_OUTPUT_DIR)

    # 检查 JSON 目录是否存在
    if not os.path.isdir(json_dir):
        print(f"❌ JSON 目录不存在: {json_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 收集所有 JSON 文件
    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        print(f"❌ JSON 目录中没有 .json 文件: {json_dir}")
        sys.exit(1)

    print(f"找到 {len(json_files)} 个 JSON 文件")
    print(f"Mask 输出目录: {mask_dir}")
    print(f"可视化输出目录: {vis_dir}")
    print("-" * 50)

    # 批量转换
    success_count = 0
    fail_count = 0

    for json_path in json_files:
        try:
            ok = json_to_mask(str(json_path), mask_dir, vis_dir)
            if ok:
                print(f"  ✅ {json_path.name}")
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ❌ {json_path.name} — 异常: {e}")
            fail_count += 1

    # 汇总 ------------------------------------------------------------
    print("-" * 50)
    print(f"转换完成: 成功 {success_count}, 失败 {fail_count}")

    # 验证：随机抽查一个 mask 的像素值 ----------------------------------
    mask_files = sorted(Path(mask_dir).glob("*.png"))
    if mask_files:
        sample = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(sample)
        print(f"\n验证样本: {mask_files[0].name}")
        print(f"  尺寸: {sample.shape}")
        print(f"  唯一像素值: {unique_vals}")
        print(f"  dtype: {sample.dtype}")
        if np.array_equal(unique_vals, [0, 1]) or np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [1]):
            print(f"  ✅ np.unique(mask) 符合预期 ([0, 1] 或其子集)")
        else:
            print(f"  ⚠ 警告: 像素值不符合预期！")


if __name__ == "__main__":
    main()
