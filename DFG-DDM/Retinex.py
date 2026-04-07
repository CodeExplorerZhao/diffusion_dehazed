import cv2
import numpy as np
import os
from pathlib import Path

# -------------------------- 核心Retinex函数（适配遥感，无ximgproc依赖） --------------------------
def single_scale_retinex(img, sigma=50):
    """单尺度Retinex（SSR）- 适配遥感，分通道拉伸"""
    img_float = np.float64(img) + 1e-6
    
    # 估计光照分量L
    L_r = cv2.GaussianBlur(img_float[:, :, 0], (0, 0), sigma)
    L_g = cv2.GaussianBlur(img_float[:, :, 1], (0, 0), sigma)
    L_b = cv2.GaussianBlur(img_float[:, :, 2], (0, 0), sigma)
    L = np.stack([L_r, L_g, L_b], axis=-1)
    
    # 分离反射分量R
    R = np.log(img_float) - np.log(L)
    
    # 分通道拉伸（1%/99%分位数）
    R_stretched = np.zeros_like(R)
    for c in range(3):
        R_min = np.percentile(R[:, :, c], 1)
        R_max = np.percentile(R[:, :, c], 99)
        R_stretched[:, :, c] = (R[:, :, c] - R_min) / (R_max - R_min + 1e-6) * 255
    
    R_ssr = np.uint8(np.clip(R_stretched, 0, 255))
    return R_ssr

def multi_scale_retinex(img, sigmas=[50, 100, 200], weights=[1/3, 1/3, 1/3]):
    """多尺度Retinex（MSR）- 适配遥感，高斯模糊替代引导滤波"""
    img_float = np.float64(img) + 1e-6
    msr = np.zeros_like(img_float)
    
    # 叠加多尺度SSR
    for sigma, weight in zip(sigmas, weights):
        L_r = cv2.GaussianBlur(img_float[:, :, 0], (0, 0), sigma)
        L_g = cv2.GaussianBlur(img_float[:, :, 1], (0, 0), sigma)
        L_b = cv2.GaussianBlur(img_float[:, :, 2], (0, 0), sigma)
        L = np.stack([L_r, L_g, L_b], axis=-1)
        
        msr += weight * (np.log(img_float) - np.log(L))
    
    # 分通道拉伸
    msr_stretched = np.zeros_like(msr)
    for c in range(3):
        msr_min = np.percentile(msr[:, :, c], 1)
        msr_max = np.percentile(msr[:, :, c], 99)
        msr_stretched[:, :, c] = (msr[:, :, c] - msr_min) / (msr_max - msr_min + 1e-6) * 255
    
    # 高斯模糊平滑（替代引导滤波）
    msr_final = np.uint8(np.clip(msr_stretched, 0, 255))
    msr_final = cv2.GaussianBlur(msr_final, (5, 5), 0)
    
    return msr_final

# -------------------------- 批量处理函数 --------------------------
def batch_retinex_dehaze(
    input_dir: str,          # 输入图像目录
    ssr_output_dir: str,     # SSR去雾结果保存目录
    msr_output_dir: str,     # MSR去雾结果保存目录
    supported_formats: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')  # 支持的图像格式
):
    """
    批量处理目录下所有图像的Retinex去雾
    :param input_dir: 原始雾图目录
    :param ssr_output_dir: SSR结果保存目录（对应retinex）
    :param msr_output_dir: MSR结果保存目录（对应mlt_retinex）
    :param supported_formats: 支持的图像格式
    """
    # 1. 创建输出目录（不存在则自动创建）
    Path(ssr_output_dir).mkdir(parents=True, exist_ok=True)
    Path(msr_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. 遍历输入目录下所有文件
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    if not file_list:
        raise ValueError(f"输入目录{input_dir}下未找到支持的图像文件！")
    
    # 3. 批量处理每个图像
    total_files = len(file_list)
    for idx, filename in enumerate(file_list):
        # 拼接完整路径
        input_path = os.path.join(input_dir, filename)
        ssr_output_path = os.path.join(ssr_output_dir, filename)
        msr_output_path = os.path.join(msr_output_dir, filename)
        
        # 跳过已处理的文件（可选，避免重复处理）
        if os.path.exists(ssr_output_path) and os.path.exists(msr_output_path):
            print(f"[{idx+1}/{total_files}] 已处理：{filename}，跳过")
            continue
        
        try:
            # 读取图像
            img = cv2.imread(input_path)
            if img is None:
                print(f"[{idx+1}/{total_files}] 读取失败：{filename}，跳过")
                continue
            
            # 处理16位遥感图像
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            
            # 转换为RGB（Retinex处理）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 执行Retinex去雾
            ssr_img = single_scale_retinex(img_rgb)
            msr_img = multi_scale_retinex(img_rgb)
            
            # 转换回BGR（OpenCV保存）
            ssr_img_bgr = cv2.cvtColor(ssr_img, cv2.COLOR_RGB2BGR)
            msr_img_bgr = cv2.cvtColor(msr_img, cv2.COLOR_RGB2BGR)
            
            # 保存结果
            # cv2.imwrite(ssr_output_path, ssr_img_bgr)
            cv2.imwrite(msr_output_path, msr_img_bgr)
            
            print(f"[{idx+1}/{total_files}] 处理完成：{filename}")
        
        except Exception as e:
            print(f"[{idx+1}/{total_files}] 处理出错：{filename}，错误信息：{e}")
    
    print(f"\n批量处理完成！总计处理{total_files}个文件")
    print(f"SSR结果保存至：{ssr_output_dir}")
    print(f"MSR结果保存至：{msr_output_dir}")

# -------------------------- 主函数调用 --------------------------
if __name__ == "__main__":
    # 配置路径（按你的需求修改）
    INPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_185000_8/Sate/Haze1k/fft-diffusion/original"
    SSR_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/retinex"  # 单尺度结果
    MSR_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/mlt_retinex"  # 多尺度结果
    
    # 执行批量处理
    batch_retinex_dehaze(
        input_dir=INPUT_DIR,
        ssr_output_dir=SSR_OUTPUT_DIR,
        msr_output_dir=MSR_OUTPUT_DIR
    )
