import cv2
import numpy as np
import os
from pathlib import Path

def post_process(img, enhance_strength="light"):
    """
    自适应轻量级后处理：温和增强边缘+抑制噪声，提升SSIM且保证视觉效果
    Args:
        img: (H,W,3) 的 np.array，像素范围 [0,1]
        enhance_strength: 增强强度（light/medium/strong），默认轻量
    """
    # 1. 安全归一化+格式转换（防止溢出）
    img = np.clip(img, 0.0, 1.0)
    img_8bit = (img * 255).round().astype(np.uint8)
    
    # 2. 自适应增强策略（先去噪，再轻量锐化）
    # 第一步：轻度高斯去噪（抑制扩散模型的细微噪声）
    img_denoise = cv2.GaussianBlur(img_8bit, (3, 3), 0.5)  # 小核去噪，保留细节
    
    # 第二步：根据强度选择温和的锐化核
    if enhance_strength == "light":
        # 最轻量（推荐）：仅增强边缘，几乎无伪影
        kernel = np.array([[0, -0.2, 0], [-0.2, 1.4, -0.2], [0, -0.2, 0]])
    elif enhance_strength == "medium":
        # 中等强度：平衡增强与效果
        kernel = np.array([[0, -0.5, 0], [-0.5, 2.0, -0.5], [0, -0.5, 0]])
    elif enhance_strength == "strong":
        # 较强强度：仅适合细节极少的图像
        kernel = np.array([[0, -0.8, 0], [-0.8, 3.2, -0.8], [0, -0.8, 0]])
    else:
        raise ValueError("enhance_strength 可选值：light/medium/strong")
    
    # 应用锐化（使用cv2.filter2D的浮点计算，避免截断失真）
    img_8bit = img_denoise.astype(np.float32)
    img_sharpen = cv2.filter2D(img_8bit, -1, kernel)
    # 裁剪到合法范围，转回8bit
    img_sharpen = np.clip(img_sharpen, 0, 255).astype(np.uint8)
    
    # 3. 还原到[0,1]浮点型
    img_processed = img_sharpen.astype(np.float32) / 255.0
    return img_processed

def batch_process_images(input_dir, output_dir, enhance_strength="light"):
    """
    批量处理目录下的所有图像文件（优化版）
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        enhance_strength: 增强强度（light/medium/strong）
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]
    total_files = len(file_list)
    
    if total_files == 0:
        print(f"警告：输入目录 {input_dir} 下未找到支持的图像文件！")
        return
    
    for idx, filename in enumerate(file_list):
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 读取图像（BGR转RGB，保持色彩正确）
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"跳过：无法读取 {filename}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # 应用优化后的后处理
            processed_img_rgb = post_process(img_rgb, enhance_strength)
            # 转回BGR保存
            processed_img_bgr = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2BGR)
            processed_img_bgr = (processed_img_bgr * 255).round().astype(np.uint8)
            
            cv2.imwrite(output_path, processed_img_bgr)
            print(f"进度：{idx+1}/{total_files} | 已处理：{filename}")
        
        except Exception as e:
            print(f"错误：处理 {filename} 失败 - {str(e)}")
            continue
    
    print(f"\n处理完成！输出目录：{output_dir}")

# --------------------------
# 主函数（修改强度只需改这里）
# --------------------------
if __name__ == "__main__":
    INPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_340000/Sate/Haze1k/fft-diffusion/processed"
    OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_340000/Sate/Haze1k/fft-diffusion/processed_post"
    
    # 选择增强强度：light（推荐）/medium/strong
    batch_process_images(INPUT_DIR, OUTPUT_DIR, enhance_strength="light")