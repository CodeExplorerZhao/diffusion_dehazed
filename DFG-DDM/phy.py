import cv2
import numpy as np
import os
from pathlib import Path
import time

# -------------------------- 1. DCP暗通道先验去雾（增加容错） --------------------------
def dark_channel_prior_dehaze(img, omega=0.95, t0=0.1, patch_size=15):
    """
    DCP暗通道先验去雾（经典传统去雾算法）- 鲁棒版
    """
    # 1. 归一化到0-1
    img_float = np.float64(img) / 255.0
    img_float = np.maximum(img_float, 1e-8)
    
    # 2. 计算暗通道（增加空值判断）
    def dark_channel(img, size):
        if img.size == 0:
            return np.zeros_like(img[:, :, 0])
        r, g, b = cv2.split(img)
        min_img = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_img = cv2.erode(min_img, kernel)
        return dark_img
    
    dark_img = dark_channel(img_float, patch_size)
    
    # 3. 估计全局大气光A（核心修复：容错空数组）
    h, w = img_float.shape[:2]
    num_pixels = max(int(h * w * 0.001), 1)  # 至少取1个像素，避免0个
    flat_img = img_float.reshape(-1, 3)
    flat_dark = dark_img.ravel()
    
    # 按暗通道从亮到暗排序，处理空索引
    if len(flat_dark) == 0:
        A = np.array([1.0, 1.0, 1.0])  # 默认大气光
    else:
        idx = flat_dark.argsort()[::-1][:num_pixels]
        # 若索引为空，取图像全局最大值
        if len(idx) == 0 or len(flat_img[idx]) == 0:
            A = np.max(flat_img, axis=0)
        else:
            A = np.max(flat_img[idx], axis=0)
    
    # 4. 估计透射率t（增加非空判断）
    if np.max(A) < 1e-8:
        A = np.array([1.0, 1.0, 1.0])
    t = 1 - omega * dark_channel(img_float / A, patch_size)
    t = np.maximum(t, t0)  # 限制最小透射率
    
    # 5. 恢复无雾图像 J = (I - A) / t + A
    dehaze_img = np.zeros_like(img_float)
    for c in range(3):
        dehaze_img[:, :, c] = (img_float[:, :, c] - A[c]) / t + A[c]
    
    # 6. 归一化+后处理
    dehaze_img = np.clip(dehaze_img * 255, 0, 255).astype(np.uint8)
    # 自适应对比度增强（优化遥感图像效果）
    dehaze_img = cv2.detailEnhance(dehaze_img, sigma_s=10, sigma_r=0.15)
    
    return dehaze_img

# -------------------------- 2. NLP非局部先验去雾（优化版+容错） --------------------------
# def non_local_prior_dehaze(img, sigma=15, h=0.1, window_size=21):
#     """
#     NLP非局部先验去雾（基于非局部均值滤波优化透射率估计）- 鲁棒版
#     """
#     img_float = np.float64(img) / 255.0
#     img_float = np.maximum(img_float, 1e-8)
    
#     # 1. 先通过暗通道估计初始透射率（增加容错）
#     def dark_channel(img, size=15):
#         if img.size == 0:
#             return np.zeros_like(img[:, :, 0])
#         r, g, b = cv2.split(img)
#         min_img = cv2.min(cv2.min(r, g), b)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#         return cv2.erode(min_img, kernel)
    
#     dark_img = dark_channel(img_float)
    
#     # 核心修复：大气光A估计容错
#     bright_pixels = img_float[dark_img > 0.9]
#     if len(bright_pixels) == 0:
#         A = np.max(img_float)  # 无满足条件的像素，取全局最大值
#     else:
#         A = np.max(bright_pixels)
    
#     t_init = 1 - 0.95 * dark_channel(img_float / A)
#     t_init = np.maximum(t_init, 0.1)
    
#     # 2. 非局部均值滤波优化透射率（核心：NLP先验）
#     t_refined = cv2.fastNlMeansDenoising(
#         src=np.uint8(t_init * 255),
#         h=h * 255,
#         templateWindowSize=7,
#         searchWindowSize=window_size
#     )
#     t_refined = np.float64(t_refined) / 255.0
    
#     # 3. 高斯平滑进一步优化
#     t_refined = cv2.GaussianBlur(t_refined, (0, 0), sigma)
    
#     # 4. 恢复无雾图像（增加除数非空判断）
#     dehaze_img = np.zeros_like(img_float)
#     t_refined = np.maximum(t_refined, 0.01)  # 避免除0
#     for c in range(3):
#         dehaze_img[:, :, c] = (img_float[:, :, c] - A) / t_refined + A
    
#     # 5. 后处理：色彩恢复+细节增强
#     dehaze_img = np.clip(dehaze_img * 255, 0, 255).astype(np.uint8)
#     # 修复：避免灰度化后色彩丢失（遥感图像关键）
#     gray_dehaze = cv2.equalizeHist(cv2.cvtColor(dehaze_img, cv2.COLOR_RGB2GRAY))
#     color_dehaze = cv2.cvtColor(gray_dehaze, cv2.COLOR_GRAY2RGB)
#     dehaze_img = cv2.addWeighted(color_dehaze, 0.8, img, 0.2, 0)  # 融合原始色彩
    
#     return dehaze_img


def non_local_prior_dehaze(img, sigma=15, h=0.1, window_size=21):
    """
    NLP非局部先验去雾（修复变灰问题，保留遥感彩色）
    """
    img_float = np.float64(img) / 255.0
    img_float = np.maximum(img_float, 1e-8)
    
    # 1. 先通过暗通道估计初始透射率（增加容错）
    def dark_channel(img, size=15):
        if img.size == 0:
            return np.zeros_like(img[:, :, 0])
        r, g, b = cv2.split(img)
        min_img = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return cv2.erode(min_img, kernel)
    
    dark_img = dark_channel(img_float)
    
    # 大气光A估计容错
    bright_pixels = img_float[dark_img > 0.9]
    if len(bright_pixels) == 0:
        A = np.max(img_float)  # 无满足条件的像素，取全局最大值
    else:
        A = np.max(bright_pixels)
    
    t_init = 1 - 0.95 * dark_channel(img_float / A)
    t_init = np.maximum(t_init, 0.1)
    
    # 2. 非局部均值滤波优化透射率（核心：NLP先验）
    t_refined = cv2.fastNlMeansDenoising(
        src=np.uint8(t_init * 255),
        h=h * 255,
        templateWindowSize=7,
        searchWindowSize=window_size
    )
    t_refined = np.float64(t_refined) / 255.0
    
    # 3. 高斯平滑进一步优化
    t_refined = cv2.GaussianBlur(t_refined, (0, 0), sigma)
    
    # 4. 恢复无雾图像（增加除数非空判断，分通道处理）
    dehaze_img = np.zeros_like(img_float)
    t_refined = np.maximum(t_refined, 0.01)  # 避免除0
    for c in range(3):  # 对R/G/B通道独立恢复，保留色彩
        dehaze_img[:, :, c] = (img_float[:, :, c] - A) / t_refined + A
    
    # 5. 修复后处理：移除灰度化，保留彩色+细节增强（核心修改）
    dehaze_img = np.clip(dehaze_img * 255, 0, 255).astype(np.uint8)
    # 彩色图像细节增强（替代灰度均衡化）
    dehaze_img = cv2.detailEnhance(dehaze_img, sigma_s=15, sigma_r=0.2)
    # 自适应直方图均衡化（CLAHE）- 彩色版，保留色彩
    lab = cv2.cvtColor(dehaze_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    dehaze_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return dehaze_img
# -------------------------- 3. 批量处理函数（整合DCP/NLP） --------------------------
def batch_dehaze_all_methods(
    input_dir: str,
    dcp_output_dir: str,
    nlp_output_dir: str,
    supported_formats: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
):
    """
    批量处理：生成DCP、NLP去雾结果（鲁棒版）
    """
    # 创建所有输出目录
    for dir_path in [dcp_output_dir, nlp_output_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 遍历输入文件
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    if not file_list:
        raise ValueError(f"输入目录{input_dir}无支持的图像文件！")
    
    total = len(file_list)
    for idx, filename in enumerate(file_list):
        input_path = os.path.join(input_dir, filename)
        dcp_path = os.path.join(dcp_output_dir, filename)
        nlp_path = os.path.join(nlp_output_dir, filename)
        
        # 跳过已处理文件
        if all(os.path.exists(p) for p in [dcp_path, nlp_path]):
            print(f"[{idx+1}/{total}] 已处理：{filename}，跳过")
            continue
        
        try:
            # 读取图像（增加有效性判断）
            img = cv2.imread(input_path)
            if img is None or img.size == 0:
                print(f"[{idx+1}/{total}] 读取失败/图像为空：{filename}，跳过")
                continue
            
            # 处理16位遥感图像
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            
            # 转换为RGB（统一处理）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # -------------------------- 执行各方法去雾 --------------------------
            # 1. DCP暗通道先验
            dcp_img = dark_channel_prior_dehaze(img_rgb)
            # 2. NLP非局部先验
            nlp_img = non_local_prior_dehaze(img_rgb)
            
            # -------------------------- 保存结果（转回BGR） --------------------------
            cv2.imwrite(dcp_path, cv2.cvtColor(dcp_img, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(nlp_path, cv2.cvtColor(nlp_img, cv2.COLOR_RGB2BGR))
            
            print(f"[{idx+1}/{total}] 处理完成：{filename}")
        
        except Exception as e:
            print(f"[{idx+1}/{total}] 处理出错：{filename}，错误：{str(e)[:80]}...")
            # 出错时仍尝试保存空文件，避免重复报错
            with open(dcp_path + ".error", "w") as f:
                f.write(str(e))
            with open(nlp_path + ".error", "w") as f:
                f.write(str(e))
    
    print(f"\n批量处理完成！")
    print(f"DCP结果：{dcp_output_dir}")
    print(f"NLP结果：{nlp_output_dir}")

# -------------------------- 主函数调用 --------------------------
if __name__ == "__main__":
    # 配置路径（按需修改）
    INPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_185000_8/Sate/Haze1k/fft-diffusion/original"
    DCP_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/dcp_dehaze_old"
    NLP_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/nlp_dehaze_old"
    
    # 执行批量处理（DCP+NLP）
    batch_dehaze_all_methods(
        input_dir=INPUT_DIR,
        dcp_output_dir=DCP_OUTPUT_DIR,
        nlp_output_dir=NLP_OUTPUT_DIR
    )

# import cv2
# import numpy as np
# import os
# from pathlib import Path

# # -------------------------- 1. 修复版DCP（适配遥感全局雾） --------------------------
# def dark_channel_prior_dehaze(img, omega=0.6, t0=0.6, patch_size=25):
#     img_float = np.float64(img) / 255.0
#     img_float = np.maximum(img_float, 1e-8)

#     def dark_channel(img, size):
#         r, g, b = cv2.split(img)
#         min_img = cv2.min(cv2.min(r, g), b)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#         return cv2.erode(min_img, kernel)

#     dark_img = dark_channel(img_float, patch_size)

#     # 非常保守的 A 估计
#     h, w = img_float.shape[:2]
#     num_pixels = max(int(h * w * 0.0001), 1)
#     flat_img = img_float.reshape(-1, 3)
#     flat_dark = dark_img.ravel()
#     idx = flat_dark.argsort()[::-1][:num_pixels]
#     A = np.mean(flat_img[idx], axis=0)
#     A = np.clip(A, 0.2, 0.8)

#     # 非常弱的透射率
#     t = 1 - omega * dark_channel(img_float / A, patch_size)
#     t = np.maximum(t, t0)
#     t = cv2.GaussianBlur(t, (3,3), 0.5)

#     # 恢复
#     dehaze = np.zeros_like(img_float)
#     for c in range(3):
#         dehaze[:,:,c] = (img_float[:,:,c] - A[c]) / t + A[c]

#     dehaze = np.clip(dehaze * 255, 0, 255).astype(np.uint8)
#     return dehaze
# # -------------------------- 2. 修复版NLP（适配遥感全局雾） --------------------------
# def non_local_prior_dehaze(img, sigma=10, h=0.02, window_size=15):
#     img_float = np.float64(img) / 255.0
#     img_float = np.maximum(img_float, 1e-8)

#     def dark_channel(img, size=25):
#         r, g, b = cv2.split(img)
#         min_img = cv2.min(cv2.min(r, g), b)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#         return cv2.erode(min_img, kernel)

#     dark_img = dark_channel(img_float)
#     bright_pixels = img_float[dark_img > 0.7]
#     if len(bright_pixels) == 0:
#         A = np.mean(img_float)
#     else:
#         A = np.mean(bright_pixels)
#     A = np.clip(A, 0.2, 0.8)

#     t_init = 1 - 0.7 * dark_channel(img_float / A)
#     t_init = np.maximum(t_init, 0.5)

#     t_refined = cv2.fastNlMeansDenoising(
#         np.uint8(t_init*255),
#         h=h*255,
#         templateWindowSize=5,
#         searchWindowSize=15
#     )
#     t_refined = np.float64(t_refined)/255.0
#     t_refined = cv2.GaussianBlur(t_refined, (3,3), 0.5)
#     t_refined = np.maximum(t_refined, 0.1)

#     dehaze = np.zeros_like(img_float)
#     for c in range(3):
#         dehaze[:,:,c] = (img_float[:,:,c] - A) / t_refined + A

#     dehaze = np.clip(dehaze*255, 0, 255).astype(np.uint8)
#     return dehaze
# # -------------------------- 3. 批量处理函数 --------------------------
# def batch_dehaze_all_methods(
#     input_dir: str,
#     dcp_output_dir: str,
#     nlp_output_dir: str,
#     supported_formats: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
# ):
#     # 创建输出目录
#     for dir_path in [dcp_output_dir, nlp_output_dir]:
#         Path(dir_path).mkdir(parents=True, exist_ok=True)
    
#     # 遍历文件
#     file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
#     if not file_list:
#         raise ValueError(f"输入目录{input_dir}无支持的图像文件！")
    
#     total = len(file_list)
#     for idx, filename in enumerate(file_list):
#         input_path = os.path.join(input_dir, filename)
#         dcp_path = os.path.join(dcp_output_dir, filename)
#         nlp_path = os.path.join(nlp_output_dir, filename)
        
#         # 跳过已处理
#         if all(os.path.exists(p) for p in [dcp_path, nlp_path]):
#             print(f"[{idx+1}/{total}] 已处理：{filename}，跳过")
#             continue
        
#         try:
#             # 读取图像
#             img = cv2.imread(input_path)
#             if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
#                 print(f"[{idx+1}/{total}] 无效图像：{filename}，跳过")
#                 continue
            
#             # 16位转8位
#             if img.dtype == np.uint16:
#                 img = (img / 256).astype(np.uint8)
            
#             # 转RGB
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#             # 去雾处理
#             dcp_img = dark_channel_prior_dehaze(img_rgb)
#             nlp_img = non_local_prior_dehaze(img_rgb)
            
#             # 保存（转回BGR）
#             cv2.imwrite(dcp_path, cv2.cvtColor(dcp_img, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(nlp_path, cv2.cvtColor(nlp_img, cv2.COLOR_RGB2BGR))
            
#             print(f"[{idx+1}/{total}] 处理完成：{filename}")
        
#         except Exception as e:
#             print(f"[{idx+1}/{total}] 处理出错：{filename}，错误：{str(e)[:80]}...")
#             with open(dcp_path + ".error", "w") as f:
#                 f.write(str(e))
#             with open(nlp_path + ".error", "w") as f:
#                 f.write(str(e))
    
#     print(f"\n批量处理完成！")
#     print(f"DCP结果：{dcp_output_dir}")
#     print(f"NLP结果：{nlp_output_dir}")

# # -------------------------- 主函数调用 --------------------------
# if __name__ == "__main__":
#     INPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_185000_8/Sate/Haze1k/fft-diffusion/original"
#     DCP_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/dcp_dehaze"
#     NLP_OUTPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/nlp_dehaze"
    
#     batch_dehaze_all_methods(
#         input_dir=INPUT_DIR,
#         dcp_output_dir=DCP_OUTPUT_DIR,
#         nlp_output_dir=NLP_OUTPUT_DIR
#     )

# import cv2
# import numpy as np
# import os
# from pathlib import Path

# # -------------------------- 1. 中等强度DCP（兼顾去雾+色彩） --------------------------
# def dark_channel_prior_dehaze_moderate(img, omega=0.75, t0=0.4, patch_size=20):
#     """
#     中等强度DCP：
#     - omega=0.75（激进0.95/温和0.6）：去雾力度适中
#     - t0=0.4（激进0.1/温和0.6）：避免天空过暗，又能有效去雾
#     - patch_size=20：适配遥感尺度，暗通道估计更稳定
#     """
#     img_float = np.float64(img) / 255.0
#     img_float = np.maximum(img_float, 1e-8)
    
#     # 暗通道计算（闭运算，保留细节）
#     def dark_channel(img, size):
#         r, g, b = cv2.split(img)
#         min_img = cv2.min(cv2.min(r, g), b)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#         return cv2.morphologyEx(min_img, cv2.MORPH_CLOSE, kernel)
    
#     dark_img = dark_channel(img_float, patch_size)
    
#     # 中等保守的大气光A估计（取0.0003%像素+均值，限制范围）
#     h, w = img_float.shape[:2]
#     num_pixels = max(int(h * w * 0.0003), 1)  # 激进0.001/温和0.0001
#     flat_img = img_float.reshape(-1, 3)
#     flat_dark = dark_img.ravel()
#     idx = flat_dark.argsort()[::-1][:num_pixels]
#     A = np.mean(flat_img[idx], axis=0)
#     A = np.clip(A, 0.25, 0.85)  # 比温和版稍宽松，兼顾去雾
    
#     # 透射率估计（中等强度，轻度平滑）
#     t = 1 - omega * dark_channel(img_float / A, patch_size)
#     t = np.maximum(t, t0)
#     t = cv2.GaussianBlur(t, (5, 5), 1.0)  # 中等平滑，避免突变
    
#     # 分通道恢复（保留色彩）
#     dehaze_img = np.zeros_like(img_float)
#     for c in range(3):
#         dehaze_img[:, :, c] = (img_float[:, :, c] - A[c]) / t + A[c]
    
#     # 轻度后处理（仅基础增强，不破坏色彩）
#     dehaze_img = np.clip(dehaze_img * 255, 0, 255).astype(np.uint8)
#     dehaze_img = cv2.detailEnhance(dehaze_img, sigma_s=8, sigma_r=0.08)  # 中等增强
    
#     return dehaze_img

# # -------------------------- 2. 中等强度NLP（兼顾去雾+色彩） --------------------------
# def non_local_prior_dehaze_moderate(img, sigma=15, h=0.04, window_size=18):
#     """
#     中等强度NLP：
#     - h=0.04（激进0.1/温和0.02）：非局部滤波强度适中
#     - t0=0.4：透射率不太低，避免色彩畸变
#     - CLAHE轻度增强：兼顾对比度+不偏色
#     """
#     img_float = np.float64(img) / 255.0
#     img_float = np.maximum(img_float, 1e-8)
    
#     # 暗通道计算（中等patch_size）
#     def dark_channel(img, size=20):
#         r, g, b = cv2.split(img)
#         min_img = cv2.min(cv2.min(r, g), b)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
#         return cv2.morphologyEx(min_img, cv2.MORPH_CLOSE, kernel)
    
#     dark_img = dark_channel(img_float)
    
#     # 中等保守的大气光A估计
#     bright_pixels = img_float[dark_img > 0.8]  # 激进0.9/温和0.7
#     if len(bright_pixels) == 0:
#         A = np.mean(img_float)
#     else:
#         A = np.mean(bright_pixels)
#     A = np.clip(A, 0.25, 0.85)
    
#     # 初始透射率（中等去雾强度）
#     t_init = 1 - 0.75 * dark_channel(img_float / A)
#     t_init = np.maximum(t_init, 0.4)
    
#     # 非局部滤波（中等强度）
#     t_refined = cv2.fastNlMeansDenoising(
#         src=np.uint8(t_init * 255),
#         h=h * 255,
#         templateWindowSize=5,
#         searchWindowSize=window_size  # 激进21/温和15
#     )
#     t_refined = np.float64(t_refined) / 255.0
#     t_refined = cv2.GaussianBlur(t_refined, (5, 5), 1.0)
#     t_refined = np.maximum(t_refined, 0.05)
    
#     # 分通道恢复
#     dehaze_img = np.zeros_like(img_float)
#     for c in range(3):
#         dehaze_img[:, :, c] = (img_float[:, :, c] - A) / t_refined + A
    
#     # 中等强度后处理（轻度CLAHE，不偏色）
#     dehaze_img = np.clip(dehaze_img * 255, 0, 255).astype(np.uint8)
#     dehaze_img = cv2.detailEnhance(dehaze_img, sigma_s=12, sigma_r=0.1)
#     # 轻度CLAHE（clipLimit=1.5，兼顾对比度和自然）
#     lab = cv2.cvtColor(dehaze_img, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12,12))  # 激进2.0/温和1.0
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     dehaze_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
#     return dehaze_img

# # -------------------------- 3. 批量处理函数（中等强度版） --------------------------
# def batch_dehaze_moderate(
#     input_dir: str,
#     dcp_moderate_dir: str,
#     nlp_moderate_dir: str,
#     supported_formats: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
# ):
#     # 创建输出目录
#     for dir_path in [dcp_moderate_dir, nlp_moderate_dir]:
#         Path(dir_path).mkdir(parents=True, exist_ok=True)
    
#     # 遍历文件
#     file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
#     if not file_list:
#         raise ValueError(f"输入目录{input_dir}无支持的图像文件！")
    
#     total = len(file_list)
#     for idx, filename in enumerate(file_list):
#         input_path = os.path.join(input_dir, filename)
#         dcp_path = os.path.join(dcp_moderate_dir, filename)
#         nlp_path = os.path.join(nlp_moderate_dir, filename)
        
#         # 跳过已处理
#         if all(os.path.exists(p) for p in [dcp_path, nlp_path]):
#             print(f"[{idx+1}/{total}] 已处理：{filename}，跳过")
#             continue
        
#         try:
#             # 读取图像
#             img = cv2.imread(input_path)
#             if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
#                 print(f"[{idx+1}/{total}] 无效图像：{filename}，跳过")
#                 continue
            
#             # 16位转8位
#             if img.dtype == np.uint16:
#                 img = (img / 256).astype(np.uint8)
            
#             # 转RGB
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#             # 中等强度去雾处理
#             dcp_img = dark_channel_prior_dehaze_moderate(img_rgb)
#             nlp_img = non_local_prior_dehaze_moderate(img_rgb)
            
#             # 保存（转回BGR）
#             cv2.imwrite(dcp_path, cv2.cvtColor(dcp_img, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(nlp_path, cv2.cvtColor(nlp_img, cv2.COLOR_RGB2BGR))
            
#             print(f"[{idx+1}/{total}] 处理完成：{filename}")
        
#         except Exception as e:
#             print(f"[{idx+1}/{total}] 处理出错：{filename}，错误：{str(e)[:80]}...")
#             with open(dcp_path + ".error", "w") as f:
#                 f.write(str(e))
#             with open(nlp_path + ".error", "w") as f:
#                 f.write(str(e))
    
#     print(f"\n批量处理完成！")
#     print(f"中等强度DCP结果：{dcp_moderate_dir}")
#     print(f"中等强度NLP结果：{nlp_moderate_dir}")

# # -------------------------- 主函数调用 --------------------------
# if __name__ == "__main__":
#     # 配置你的路径
#     INPUT_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_185000_8/Sate/Haze1k/fft-diffusion/original"
#     DCP_MODERATE_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/dcp_dehaze_moderate"
#     NLP_MODERATE_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/nlp_dehaze_moderate"
    
#     # 执行批量处理
#     batch_dehaze_moderate(
#         input_dir=INPUT_DIR,
#         dcp_moderate_dir=DCP_MODERATE_DIR,
#         nlp_moderate_dir=NLP_MODERATE_DIR
#     )