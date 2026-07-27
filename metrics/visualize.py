import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_heatmap(tile_pred, path, img_size: int = 224, auto_norm: bool = False):
    """将 tile 预测结果保存为热力图。

    颜色映射基于绝对值（0=蓝, 0.5=绿, 1=红），不做 min-max 归一化。
    这样 0.3 始终显示为绿/黄，0.9 始终显示为红，
    不会因为同张图中没有高值就把 0.3 拉成红色。

    Args:
        tile_pred: (grid_size, grid_size) 的 tile 概率值 [0,1]
        path: 保存路径
        img_size: 输出图像尺寸
        auto_norm: 已废弃 — 不应使用，会导致低值区域被拉红
    """
    if tile_pred.ndim == 3:
        tile_pred = tile_pred[0]

    # 直接使用绝对值映射颜色（不归一化）
    heatmap = np.array(Image.fromarray(
        (tile_pred * 255).astype(np.uint8)
    ).resize((img_size, img_size), Image.NEAREST))

    heatmap_rgb = _apply_jet_colormap(heatmap)
    Image.fromarray(heatmap_rgb).save(path)


def save_overlay(image, tile_pred, path, img_size: int = 224, alpha: float = 0.5,
                 auto_norm: bool = False):
    """将 tile 热力图叠加到原图上保存。颜色基于绝对值，不归一化。

    Args:
        image: (3, H, W) tensor 或 (H, W, 3) ndarray
        tile_pred: (grid_size, grid_size) tile 概率值 [0,1]
        path: 保存路径
        img_size: 输出图像尺寸
        alpha: 热力图叠加透明度
    """
    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # 直接绝对值映射（不归一化）
    heatmap = np.array(Image.fromarray(
        (tile_pred * 255).astype(np.uint8)
    ).resize((img_size, img_size), Image.NEAREST))

    heatmap_rgb = _apply_jet_colormap(heatmap)

    # 如果原图尺寸和 img_size 不一致，先缩放原图
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

    overlay = (image * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def _apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """简易 JET colormap 实现。

    映射规则 (v 压缩后):
        0.0  → 蓝色 (冷区，无脏污)
        0.2  → 绿色
        0.5  → 黄/橙 (⚠ 阈值临界点，≥0.5=脏污)
        1.0  → 红色 (热区，高脏污概率)

    关键: 0.5 不在绿色而在黄/橙色 → 超过阈值就进入暖色区。
    """
    v = gray.astype(np.float32) / 255.0

    # 线性映射: 绝对值直接映射颜色
    # 不使用 gamma 校正——会放大干净图上的微小噪声
    # 0.0→蓝, 0.25→青, 0.5→黄(阈值), 1.0→红
    # <0.1 的值全部显示为蓝色系，不会出现绿色
    # R: 0.25→0, 0.5→1.0, 1.0→1.0
    r = np.clip((v - 0.25) * 4.0, 0, 1)
    # G: 0.0→1.0, 0.5→1.0, 0.5→0
    g = np.clip(v * 4.0, 0, 1) - np.clip((v - 0.5) * 4.0, 0, 1)
    # B: 1.0→0
    b = np.clip(1.0 - v * 4.0, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_grid_overlay(image, tile_pred, path, img_size: int = 224):
    """将 8×8 网格和置信度数值叠加到原图上保存。

    Args:
        image: (3, H, W) tensor 或 (H, W, 3) ndarray
        tile_pred: (grid_size, grid_size) tile 概率值 [0,1]
        path: 保存路径
        img_size: 输出图像尺寸
    """
    # 处理 image
    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

    grid_size = tile_pred.shape[0]
    tile_sz = img_size // grid_size

    # 在图片上绘制
    pil_img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 尝试加载字体，失败则用默认
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  size=max(10, tile_sz // 3))
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i in range(grid_size):
        for j in range(grid_size):
            y0, x0 = i * tile_sz, j * tile_sz
            y1, x1 = y0 + tile_sz, x0 + tile_sz
            val = tile_pred[i, j]

            # 网格线
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 180), width=1)

            # 置信度数值
            text = f"{val:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x0 + (tile_sz - tw) // 2
            ty = y0 + (tile_sz - th) // 2

            # 半透明背景框（根据置信度决定深浅）
            alpha_bg = int(140 + val * 80)
            if val > 0.5:
                bg_color = (200, 40, 40, alpha_bg)   # 红色背景 = 脏污
            else:
                bg_color = (40, 40, 200, alpha_bg)   # 蓝色背景 = 干净
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=bg_color)

            # 白色文字
            draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

    # 合并
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    result.save(path)


def save_comparison(image, gt_tile, pred_tile, path, img_size: int = 224):
    """生成 GT vs Pred 对比图：左=GT热力图，右=模型预测热力图。

    Args:
        image: (3, H, W) tensor 或 (H, W, 3) ndarray
        gt_tile: (grid_size, grid_size) GT tile 标签 [0,1]
        pred_tile: (grid_size, grid_size) 模型预测 tile 概率 [0,1]
        path: 保存路径
        img_size: 输出图像尺寸
    """
    # 处理 image
    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

    grid_size = gt_tile.shape[0]
    tile_sz = img_size // grid_size

    def make_overlay(tile_data, alpha=0.55):
        # 上采样 tile → img_size
        hm = np.array(Image.fromarray(
            (tile_data * 255).astype(np.uint8)
        ).resize((img_size, img_size), Image.NEAREST))
        hm_rgb = _apply_jet_colormap(hm)
        return (image * (1 - alpha) + hm_rgb * alpha).astype(np.uint8)

    gt_overlay = make_overlay(gt_tile)
    pred_overlay = make_overlay(pred_tile)

    # 左右拼接
    # 添加标题条
    label_h = 20
    label_bar = np.full((label_h, img_size * 2, 3), 40, dtype=np.uint8)
    # 用白色分隔线
    combined = np.concatenate([gt_overlay, pred_overlay], axis=1)
    combined = np.concatenate([label_bar, combined], axis=0)

    # 用 PIL 写文字标签
    pil_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((img_size // 2 - 30, 2), "GT (Ground Truth)", fill=(255, 255, 255), font=font)
    draw.text((img_size + img_size // 2 - 50, 2), "Pred (Model)", fill=(255, 255, 255), font=font)

    # 中间分隔线
    draw.line([(img_size, label_h), (img_size, img_size + label_h)],
              fill=(255, 255, 255), width=2)

    pil_img.save(path)


def save_heatmap_compare(image, gt_tile, pred_tile, path, img_size: int = 224):
    """GT 热力图 vs Pred 热力图 并排对比。

    Args:
        image: (3, H, W) tensor 或 ndarray（仅用于统一尺寸）
        gt_tile: (G, G) GT tile 标签
        pred_tile: (G, G) 预测 tile 概率
        path: 保存路径
        img_size: 输出尺寸
    """
    # 生成两张 pure heatmap
    def _make_heatmap(tile_data):
        hm = np.array(Image.fromarray(
            (tile_data * 255).astype(np.uint8)
        ).resize((img_size, img_size), Image.NEAREST))
        return _apply_jet_colormap(hm)
    _side_by_side(_make_heatmap(gt_tile), _make_heatmap(pred_tile),
                  path, img_size, "GT Heatmap", "Pred Heatmap")


def save_overlay_compare(image, gt_tile, pred_tile, path, img_size: int = 224):
    """GT overlay vs Pred overlay 并排对比（热力图叠加在原图上）。

    Args:
        image: (3, H, W) tensor 或 ndarray
        gt_tile: (G, G) GT tile 标签
        pred_tile: (G, G) 预测 tile 概率
        path: 保存路径
        img_size: 输出尺寸
    """
    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

    def _make_overlay(tile_data, alpha=0.55):
        hm = np.array(Image.fromarray(
            (tile_data * 255).astype(np.uint8)
        ).resize((img_size, img_size), Image.NEAREST))
        hm_rgb = _apply_jet_colormap(hm)
        return (image * (1 - alpha) + hm_rgb * alpha).astype(np.uint8)

    _side_by_side(_make_overlay(gt_tile), _make_overlay(pred_tile),
                  path, img_size, "GT Overlay", "Pred Overlay")


def save_grid_compare(image, gt_tile, pred_tile, path, img_size: int = 224):
    """GT 网格数值 vs Pred 网格数值 并排对比。

    Args:
        image: (3, H, W) tensor 或 ndarray
        gt_tile: (G, G) GT tile 标签
        pred_tile: (G, G) 预测 tile 概率
        path: 保存路径
        img_size: 输出尺寸
    """
    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = np.array(Image.fromarray(image).resize((img_size, img_size)))

    grid_size = gt_tile.shape[0]
    tile_sz = img_size // grid_size

    def _draw_grid(tile_data):
        pil = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=max(10, tile_sz // 3))
        except (IOError, OSError):
            font = ImageFont.load_default()
        for i in range(grid_size):
            for j in range(grid_size):
                y0, x0 = i * tile_sz, j * tile_sz
                y1, x1 = y0 + tile_sz, x0 + tile_sz
                val = tile_data[i, j]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=1)
                text = f"{val:.2f}"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx, ty = x0 + (tile_sz - tw) // 2, y0 + (tile_sz - th) // 2
                alpha_bg = int(140 + val * 80)
                bg = (200, 40, 40, alpha_bg) if val > 0.5 else (40, 40, 200, alpha_bg)
                draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=bg)
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
        return np.array(pil)

    _side_by_side(_draw_grid(gt_tile), _draw_grid(pred_tile),
                  path, img_size, "GT Grid", "Pred Grid")


def _side_by_side(left_img, right_img, path, img_size, left_label, right_label):
    """通用左右并排拼接，上方带标题条。"""
    label_h = 20
    label_bar = np.full((label_h, img_size * 2, 3), 40, dtype=np.uint8)
    combined = np.concatenate([left_img, right_img], axis=1)
    combined = np.concatenate([label_bar, combined], axis=0)

    pil_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((img_size // 2 - len(left_label) * 4, 2), left_label,
              fill=(255, 255, 255), font=font)
    draw.text((img_size + img_size // 2 - len(right_label) * 4, 2), right_label,
              fill=(255, 255, 255), font=font)
    draw.line([(img_size, label_h), (img_size, img_size + label_h)],
              fill=(255, 255, 255), width=2)
    pil_img.save(path)
