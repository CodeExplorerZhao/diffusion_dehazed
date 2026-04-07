import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ====================== 1. 全局配置（按需修改） ======================
# 基础路径
FINALLY_RESULT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/finally_result"  # 方法结果根目录
GT_DIR = "../../datasets/Haze1k_ddm2/test/GT"  # 真值目录
HAZY_DIR = "../../datasets/Haze1k_ddm2/test/hazy"  # 标签目录
OUTPUT_ROOT_DIR = "./display"  # 最终保存根目录

# 随机选择配置
PREFIXES = ["moderate", "thick", "thin"]  # 三类前缀
SELECT_NUM_PER_PREFIX = 5  # 每类随机选5张
RANDOM_SEED = 42  # 固定随机种子（保证可复现）

# 差异图配置
NORMALIZE = True  # 图像是否归一化（0-255→True，0-1→False）
CMAP = "jet"  # 热力图配色
SAVE_DIFF_VISUAL = False  # 是否保存可视化差异图（True/False）
GENERATE_GT_SELF_DIFF = True  # 是否生成GT与自身的对比误差图（基准验证）
GENERATE_HAZY_GT_DIFF = True  # 是否生成hazy vs GT的误差图（去雾基准）

# 支持的图片格式（解决jpg/png匹配问题）
SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
SUPPORTED_EXTS = [ext.lower() for ext in SUPPORTED_EXTS]

# ====================== 2. 工具函数 ======================
def get_core_filename(filename):
    """提取文件名核心部分（去掉后缀），解决jpg/png格式不匹配问题"""
    fname = os.path.splitext(filename)[0]
    return fname.lower()  # 转小写，避免大小写问题

def read_image(img_path):
    """读取图像并统一为RGB格式，处理归一化"""
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32)
        if NORMALIZE:
            img_np = img_np / 255.0  # 归一化到0-1
        return img_np
    except Exception as e:
        raise ValueError(f"读取图像{img_path}失败: {e}")

def save_original_image(img_path, save_dir, img_name, prefix=""):
    """
    通用保存原始图函数（支持为不同类型图加前缀区分）
    :param prefix: 前缀（如"pred_"、"gt_"、"hazy_"），用于区分不同类型原始图
    """
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    # 读取原始图（不归一化，保持原始像素值）
    img = Image.open(img_path).convert("RGB")
    # 保存原始图（加前缀区分）
    save_path = os.path.join(save_dir, f"{prefix}{img_name}_original.png")
    img.save(save_path)
    print(f"✅ 原始图已保存：{save_path}")

def generate_gt_self_diff_map(gt_img_path: str, save_dir: str, img_name: str):
    """生成GT与自身对比的误差图（基准验证，理论误差应为0）"""
    if not GENERATE_GT_SELF_DIFF:
        return
    
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    # 读取同一张GT图两次，模拟对比
    gt_img1 = read_image(gt_img_path)
    gt_img2 = read_image(gt_img_path)
    
    # 计算GT自身的差异（理论值全为0）
    abs_diff = np.abs(gt_img1 - gt_img2)
    abs_diff_mean = np.mean(abs_diff, axis=-1)
    rel_diff = np.abs(gt_img1 - gt_img2) / (np.maximum(gt_img1, gt_img2) + 1e-6)
    rel_diff_mean = np.mean(rel_diff, axis=-1)
    
    # 归一化（即使全0也不报错）
    abs_diff_norm = (abs_diff_mean - abs_diff_mean.min()) / (abs_diff_mean.max() - abs_diff_mean.min() + 1e-6)
    rel_diff_norm = (rel_diff_mean - rel_diff_mean.min()) / (rel_diff_mean.max() - rel_diff_mean.min() + 1e-6)
    
    # 保存可视化大图（可选）
    if SAVE_DIFF_VISUAL:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(gt_img1)
        axes[0].set_title("GT图像1", fontsize=14)
        axes[0].axis("off")
        
        axes[1].imshow(gt_img2)
        axes[1].set_title("GT图像2（同一张图）", fontsize=14)
        axes[1].axis("off")
        
        im = axes[2].imshow(abs_diff_norm, cmap=CMAP)
        axes[2].set_title("GT vs GT 绝对误差热力图（基准）", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        fig.savefig(f"{save_dir}/{img_name}_gt_self_diff_visual.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 保存差异图（全黑，因为误差为0）
    abs_jet = cv2.applyColorMap((abs_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_gt_self_abs_diff_jet.png", abs_jet)
    
    rel_jet = cv2.applyColorMap((rel_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_gt_self_rel_diff_jet.png", rel_jet)
    
    # 保存统计信息（验证误差是否为0）
    with open(f"{save_dir}/{img_name}_gt_self_diff_stats.txt", "w") as f:
        f.write("===== GT vs GT 误差统计（基准） =====\n")
        f.write(f"绝对误差 - 均值：{abs_diff_mean.mean():.6f}, 最大值：{abs_diff_mean.max():.6f}\n")
        f.write(f"相对误差 - 均值：{rel_diff_mean.mean():.6f}, 最大值：{rel_diff_mean.max():.6f}\n")
    
    print(f"✅ {img_name} GT自对比基准图已保存（理论误差应为0）")

def generate_hazy_gt_diff_map(hazy_img_path: str, gt_img_path: str, save_dir: str, img_name: str):
    """生成hazy vs GT的误差图（去雾效果基准，代表无去雾时的原始误差）"""
    if not GENERATE_HAZY_GT_DIFF:
        return
    
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    # 读取图像
    hazy_img = read_image(hazy_img_path)
    gt_img = read_image(gt_img_path)
    
    # 尺寸对齐（统一按GT尺寸缩放）
    target_h, target_w = gt_img.shape[:2]
    if hazy_img.shape[:2] != (target_h, target_w):
        hazy_img = cv2.resize(hazy_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"警告：{img_name} - hazy图尺寸不一致，已缩放至{target_w}x{target_h}")
    
    # 计算hazy vs GT的差异
    abs_diff = np.abs(hazy_img - gt_img)
    abs_diff_mean = np.mean(abs_diff, axis=-1)
    rel_diff = np.abs(hazy_img - gt_img) / (np.maximum(hazy_img, gt_img) + 1e-6)
    rel_diff_mean = np.mean(rel_diff, axis=-1)
    
    # 归一化
    abs_diff_norm = (abs_diff_mean - abs_diff_mean.min()) / (abs_diff_mean.max() - abs_diff_mean.min() + 1e-6)
    rel_diff_norm = (rel_diff_mean - rel_diff_mean.min()) / (rel_diff_mean.max() - rel_diff_mean.min() + 1e-6)
    
    # 保存可视化大图（可选）
    if SAVE_DIFF_VISUAL:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(hazy_img)
        axes[0].set_title("hazy有雾图像", fontsize=14)
        axes[0].axis("off")
        
        axes[1].imshow(gt_img)
        axes[1].set_title("GT真值图像", fontsize=14)
        axes[1].axis("off")
        
        im = axes[2].imshow(abs_diff_norm, cmap=CMAP)
        axes[2].set_title("hazy vs GT 绝对误差热力图（去雾基准）", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        fig.savefig(f"{save_dir}/{img_name}_hazy_gt_diff_visual.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 保存差异图
    abs_jet = cv2.applyColorMap((abs_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_hazy_gt_abs_diff_jet.png", abs_jet)
    
    rel_jet = cv2.applyColorMap((rel_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_hazy_gt_rel_diff_jet.png", rel_jet)
    
    # 保存统计信息
    with open(f"{save_dir}/{img_name}_hazy_gt_diff_stats.txt", "w") as f:
        f.write("===== hazy vs GT 误差统计（去雾基准） =====\n")
        f.write(f"绝对误差 - 均值：{abs_diff_mean.mean():.4f}, 最大值：{abs_diff_mean.max():.4f}\n")
        f.write(f"相对误差 - 均值：{rel_diff_mean.mean():.4f}, 最大值：{rel_diff_mean.max():.4f}\n")
    
    print(f"✅ {img_name} hazy vs GT基准图已保存")

def generate_pred_gt_diff_map(
    pred_img_path: str,
    gt_img_path: str,
    save_dir: str,
    img_name: str
):
    """仅生成预测图 vs GT的差异图"""
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    pred_img = read_image(pred_img_path)
    gt_img = read_image(gt_img_path)
    
    # 尺寸对齐
    target_h, target_w = gt_img.shape[:2]
    if pred_img.shape[:2] != (target_h, target_w):
        pred_img = cv2.resize(pred_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"警告：{img_name} - 预测图尺寸不一致，已缩放至{target_w}x{target_h}")
    
    # 计算预测 vs GT的差异
    abs_diff = np.abs(pred_img - gt_img)
    abs_diff_mean = np.mean(abs_diff, axis=-1)
    rel_diff = np.abs(pred_img - gt_img) / (np.maximum(pred_img, gt_img) + 1e-6)
    rel_diff_mean = np.mean(rel_diff, axis=-1)
    
    # 归一化
    abs_diff_norm = (abs_diff_mean - abs_diff_mean.min()) / (abs_diff_mean.max() - abs_diff_mean.min() + 1e-6)
    rel_diff_norm = (rel_diff_mean - rel_diff_mean.min()) / (rel_diff_mean.max() - rel_diff_mean.min() + 1e-6)
    
    # 可选保存可视化大图
    if SAVE_DIFF_VISUAL:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(pred_img)
        axes[0].set_title("方法预测去雾图像", fontsize=14)
        axes[0].axis("off")
        
        axes[1].imshow(gt_img)
        axes[1].set_title("GT真值图像", fontsize=14)
        axes[1].axis("off")
        
        im = axes[2].imshow(abs_diff_norm, cmap=CMAP)
        axes[2].set_title("预测 vs GT 绝对误差热力图", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        fig.savefig(f"{save_dir}/{img_name}_pred_gt_diff_visual.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ {img_name} 预测vsGT可视化差异图已保存")
    
    # 保存关键差异图
    abs_jet = cv2.applyColorMap((abs_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_pred_gt_abs_diff_jet.png", abs_jet)
    
    rel_jet = cv2.applyColorMap((rel_diff_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{save_dir}/{img_name}_pred_gt_rel_diff_jet.png", rel_jet)
    
    # 保存统计信息
    with open(f"{save_dir}/{img_name}_pred_gt_diff_stats.txt", "w") as f:
        f.write("===== 预测 vs GT 误差统计 =====\n")
        f.write(f"绝对误差 - 均值：{abs_diff_mean.mean():.4f}, 最大值：{abs_diff_mean.max():.4f}, 最小值：{abs_diff_mean.min():.4f}\n")
        f.write(f"相对误差 - 均值：{rel_diff_mean.mean():.4f}, 最大值：{rel_diff_mean.max():.4f}, 最小值：{rel_diff_mean.min():.4f}\n")
    
    print(f"✅ {img_name} 预测vsGT差异图+统计信息已保存至：{save_dir}")

# ====================== 3. 核心逻辑 ======================
def get_files_by_prefix(dir_path, prefixes):
    """按前缀获取目录下的文件，返回{核心文件名: 完整路径}的字典"""
    file_dict = {}
    if not os.path.exists(dir_path):
        raise ValueError(f"目录不存在：{dir_path}")
    
    for fname in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, fname)):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        core_name = get_core_filename(fname)
        if any(core_name.startswith(prefix.lower()) for prefix in prefixes):
            file_dict[core_name] = os.path.join(dir_path, fname)
    return file_dict

def main():
    random.seed(RANDOM_SEED)
    
    # 步骤1：获取GT文件并随机选择
    gt_file_dict = get_files_by_prefix(GT_DIR, PREFIXES)
    selected_core_names = {}
    for prefix in PREFIXES:
        prefix_core_names = [name for name in gt_file_dict.keys() if name.startswith(prefix.lower())]
        if len(prefix_core_names) < SELECT_NUM_PER_PREFIX:
            raise ValueError(f"❌ {prefix}前缀的GT文件不足{SELECT_NUM_PER_PREFIX}张（仅{len(prefix_core_names)}张）")
        selected_core_names[prefix] = random.sample(prefix_core_names, SELECT_NUM_PER_PREFIX)
    print(f"✅ 随机选择完成：{SELECT_NUM_PER_PREFIX}张/类 × {len(PREFIXES)}类 = {SELECT_NUM_PER_PREFIX*len(PREFIXES)}张")
    
    # 步骤2：保存GT/hazy原始图 + 生成GT自对比 + hazy vs GT基准图
    print("\n========== 保存GT/hazy原始图 + 生成基准误差图 ==========")
    hazy_file_dict = get_files_by_prefix(HAZY_DIR, PREFIXES)
    for prefix in PREFIXES:
        for core_name in selected_core_names[prefix]:
            # 保存GT原始图 + 生成GT自对比图
            if core_name in gt_file_dict:
                gt_original_save_dir = os.path.join(OUTPUT_ROOT_DIR, "GT", prefix)
                save_original_image(gt_file_dict[core_name], gt_original_save_dir, core_name, prefix="gt_")
                generate_gt_self_diff_map(gt_file_dict[core_name], gt_original_save_dir, core_name)
            else:
                print(f"❌ {core_name} - GT文件缺失，跳过保存")
            
            # 保存hazy原始图 + 生成hazy vs GT基准图
            if core_name in hazy_file_dict and core_name in gt_file_dict:
                hazy_original_save_dir = os.path.join(OUTPUT_ROOT_DIR, "hazy", prefix)
                save_original_image(hazy_file_dict[core_name], hazy_original_save_dir, core_name, prefix="hazy_")
                generate_hazy_gt_diff_map(hazy_file_dict[core_name], gt_file_dict[core_name], hazy_original_save_dir, core_name)
            else:
                print(f"❌ {core_name} - hazy/GT文件缺失，跳过hazy vs GT基准图生成")
    
    # 步骤3：遍历方法文件夹
    method_dirs = [d for d in os.listdir(FINALLY_RESULT_DIR) if os.path.isdir(os.path.join(FINALLY_RESULT_DIR, d))]
    if not method_dirs:
        raise ValueError(f"❌ finally_result目录下无方法文件夹：{FINALLY_RESULT_DIR}")
    print(f"\n✅ 检测到{len(method_dirs)}个方法：{method_dirs}")
    
    for method_name in method_dirs:
        method_path = os.path.join(FINALLY_RESULT_DIR, method_name)
        print(f"\n========== 处理方法：{method_name} ==========")
        method_file_dict = get_files_by_prefix(method_path, PREFIXES)
        
        for prefix in PREFIXES:
            for core_name in selected_core_names[prefix]:
                # 校验文件是否存在
                if core_name not in gt_file_dict:
                    print(f"❌ {core_name} - GT文件缺失，跳过")
                    continue
                if core_name not in method_file_dict:
                    print(f"❌ {core_name} - {method_name}预测文件缺失，跳过")
                    continue
                
                # 构建路径
                gt_img_path = gt_file_dict[core_name]
                pred_img_path = method_file_dict[core_name]
                method_save_dir = os.path.join(OUTPUT_ROOT_DIR, method_name, prefix)
                
                # 保存该方法的去雾预测原始图
                save_original_image(pred_img_path, method_save_dir, core_name, prefix="pred_")
                
                # 生成预测vsGT差异图
                generate_pred_gt_diff_map(pred_img_path, gt_img_path, method_save_dir, core_name)
    
    print(f"\n🎉 所有处理完成！结果保存至：{OUTPUT_ROOT_DIR}")

# ====================== 4. 执行入口 ======================
if __name__ == "__main__":
    main()