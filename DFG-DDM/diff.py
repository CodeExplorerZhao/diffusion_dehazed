import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path

def generate_pixel_diff_map(
    pred_img_path: str,  # 预测图像路径
    gen_img_path: str,   # 生成图像路径
    save_dir: str = "./diff_maps",  # 差异图保存目录
    normalize: bool = True,  # 是否将图像归一化到0-1（输入为255范围时开启）
    cmap: str = "jet"  # 热力图配色（jet/viridis/inferno等）
):
    """
    生成预测图像与生成图像的像素差异图
    :param pred_img_path: 预测图像路径（支持jpg/png/bmp）
    :param gen_img_path: 生成图像路径
    :param save_dir: 差异图保存目录
    :param normalize: 若图像是0-255范围，设为True；0-1范围设为False
    :param cmap: 热力图配色方案
    """
    # 1. 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    
    # 2. 读取图像并统一为RGB格式（避免灰度图/Alpha通道干扰）
    def read_image(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32)
            if normalize:
                img_np = img_np / 255.0  # 归一化到0-1
            return img_np
        except Exception as e:
            raise ValueError(f"读取图像{img_path}失败: {e}")
    
    pred_img = read_image(pred_img_path)
    gen_img = read_image(gen_img_path)
    
    # 3. 检查图像尺寸是否一致（核心校验）
    if pred_img.shape != gen_img.shape:
        # 自动对齐尺寸（按生成图像尺寸缩放预测图像）
        gen_h, gen_w = gen_img.shape[:2]
        pred_img = cv2.resize(pred_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)
        print(f"警告：图像尺寸不一致，已将预测图像缩放到{gen_w}x{gen_h}")
    
    # 4. 计算像素差异
    # 4.1 绝对误差（逐通道）
    abs_diff = np.abs(pred_img - gen_img)
    # 4.2 平均绝对误差（单通道，便于可视化）
    abs_diff_mean = np.mean(abs_diff, axis=-1)
    # 4.3 相对误差（避免除0，加小epsilon）
    rel_diff = np.abs(pred_img - gen_img) / (np.maximum(pred_img, gen_img) + 1e-6)
    rel_diff_mean = np.mean(rel_diff, axis=-1)
    
    # 5. 归一化差异值到0-1（便于可视化）
    abs_diff_mean_norm = (abs_diff_mean - abs_diff_mean.min()) / (abs_diff_mean.max() - abs_diff_mean.min() + 1e-6)
    rel_diff_mean_norm = (rel_diff_mean - rel_diff_mean.min()) / (rel_diff_mean.max() - rel_diff_mean.min() + 1e-6)
    
    # 6. 可视化并保存差异图
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # 第一行：原始图像 + 绝对误差热力图 + 相对误差热力图
    axes[0,0].imshow(pred_img)
    axes[0,0].set_title("预测图像", fontsize=14)
    axes[0,0].axis("off")
    
    axes[0,1].imshow(abs_diff_mean_norm, cmap=cmap)
    axes[0,1].set_title("像素绝对误差热力图", fontsize=14)
    axes[0,1].axis("off")
    plt.colorbar(axes[0,1].imshow(abs_diff_mean_norm, cmap=cmap), ax=axes[0,1], shrink=0.8)
    
    axes[0,2].imshow(rel_diff_mean_norm, cmap=cmap)
    axes[0,2].set_title("像素相对误差热力图", fontsize=14)
    axes[0,2].axis("off")
    plt.colorbar(axes[0,2].imshow(rel_diff_mean_norm, cmap=cmap), ax=axes[0,2], shrink=0.8)
    
    # 第二行：生成图像 + 绝对误差灰度图 + 相对误差灰度图
    axes[1,0].imshow(gen_img)
    axes[1,0].set_title("生成图像", fontsize=14)
    axes[1,0].axis("off")
    
    axes[1,1].imshow(abs_diff_mean_norm, cmap="gray")
    axes[1,1].set_title("像素绝对误差灰度图", fontsize=14)
    axes[1,1].axis("off")
    
    axes[1,2].imshow(rel_diff_mean_norm, cmap="gray")
    axes[1,2].set_title("像素相对误差灰度图", fontsize=14)
    axes[1,2].axis("off")
    
    # 保存整体可视化图
    # fig.savefig(f"{save_dir}/pixel_diff_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 单独保存关键差异图（便于后续分析）
    # 绝对误差热力图（彩色）
    abs_diff_colored = cv2.applyColorMap((abs_diff_mean_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/abs_diff_jet.png", abs_diff_colored)
    # 相对误差热力图（彩色）
    rel_diff_colored = cv2.applyColorMap((rel_diff_mean_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/rel_diff_jet.png", rel_diff_colored)
    # 原始绝对误差矩阵（npy格式，便于数值分析）
    # np.save(f"{save_dir}/abs_diff_matrix.npy", abs_diff)
    
    print(f"差异图已保存至：{save_dir}")
    print(f"绝对误差统计 - 均值：{abs_diff_mean.mean():.4f}, 最大值：{abs_diff_mean.max():.4f}")
    print(f"相对误差统计 - 均值：{rel_diff_mean.mean():.4f}, 最大值：{rel_diff_mean.max():.4f}")

# ------------------------------
# 示例调用（直接运行即可）
# ------------------------------
if __name__ == "__main__":
    # 替换为你的预测图像和生成图像路径
    PRED_IMG_PATH = "../../datasets/Haze1k_ddm/test/GT/moderate_test_2.png"  # 预测图像
    GEN_IMG_PATH = "/home/yueyinlei/paper/DFG-DDM/test_results/images_all_185000_8/Sate/Haze1k/fft-diffusion/processed/moderate_test_2.png"    # 生成图像
    
    # 生成差异图
    generate_pixel_diff_map(
        pred_img_path=PRED_IMG_PATH,
        gen_img_path=GEN_IMG_PATH,
        save_dir="./diff_maps",
        normalize=True  # 若图像是0-1范围，改为False
    )