import cv2
import numpy as np
from tqdm import tqdm
import os
import lpips
import torch
import torch_fidelity
# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR


def calculate_psnr(img1, img2, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def read_image(image_path):
    """读取本地图像，转为RGB格式（H, W, 3），像素范围[0,255]"""
    # 用cv2读取（默认BGR），转为RGB
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_lpips(img1, img2, loss_fn, device='cuda'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img1 (ndarray): Images with range [0, 255], RGB order, shape HWC.
        img2 (ndarray): Images with range [0, 255], RGB order, shape HWC.
        loss_fn (lpips.LPIPS): Pre-initialized LPIPS model.
        device (str): Compute device ('cuda' or 'cpu').

    Returns:
        float: lpips result (lower is better).
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    assert img1.shape[2] == 3, 'LPIPS requires 3-channel RGB images.'

    # 1. 转换为 PyTorch Tensor，并变更为 float32
    img1_t = torch.from_numpy(img1).float()
    img2_t = torch.from_numpy(img2).float()

    # 2. 归一化: [0, 255] -> [-1.0, 1.0] (LPIPS 的强制要求)
    img1_t = (img1_t / 255.0) * 2.0 - 1.0
    img2_t = (img2_t / 255.0) * 2.0 - 1.0

    # 3. 调整维度: (H, W, C) -> (C, H, W) -> 增加 Batch 维度 (1, C, H, W)
    img1_t = img1_t.permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = img2_t.permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. 前向传播计算感知损失 (不需要计算梯度)
    with torch.no_grad():
        lpips_val = loss_fn(img1_t, img2_t)

    return lpips_val.item()
# ========== 核心：批量计算的主函数 ==========
# def batch_calculate_metrics(gt_dir, processed_dir, test_y_channel=True):
#     """
#     批量计算两个目录下对应图像的SSIM和PSNR
#     Args:
#         gt_dir: 真值图像目录（如 /root/DFG-DDM/.../gt/）
#         processed_dir: 处理后图像目录（如 /root/DFG-DDM/.../processed/）
#         test_y_channel: 是否在YCbCr的Y通道计算（推荐True）
#     Returns:
#         平均SSIM、平均PSNR
#     """
#     # 1. 获取真值目录下的所有图像文件（过滤非图片格式）
#     img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
#     gt_files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in img_extensions]
#     if not gt_files:
#         raise ValueError(f"真值目录 {gt_dir} 下未找到图片文件！")
    
#     # 2. 初始化指标列表
#     ssim_list = []
#     psnr_list = []
#     failed_files = []

#     # 3. 循环处理每张图像（带进度条）
#     for gt_file in tqdm(gt_files, desc="批量计算SSIM/PSNR", unit="img"):
#         try:
#             # 提取文件名（去掉后缀），匹配processed目录下的文件
#             file_name = os.path.splitext(gt_file)[0]
#             # 遍历processed目录，找同名（不同后缀也匹配）的文件
#             processed_file = None
#             for f in os.listdir(processed_dir):
#                 if os.path.splitext(f)[0] == file_name:
#                     processed_file = f
#                     break
#             if not processed_file:
#                 tqdm.write(f"警告：{processed_dir} 中未找到 {file_name} 对应的处理后文件，跳过")
#                 failed_files.append(gt_file)
#                 continue

#             # 拼接完整路径
#             gt_path = os.path.join(gt_dir, gt_file)
#             processed_path = os.path.join(processed_dir, processed_file)

#             # 读取图像
#             img_gt = read_image(gt_path)
#             img_processed = read_image(processed_path)

#             # 统一图像尺寸（防止尺寸不一致报错）
#             if img_gt.shape != img_processed.shape:
#                 img_processed = cv2.resize(img_processed, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
#                 tqdm.write(f"警告：{gt_file} 和 {processed_file} 尺寸不一致，已自动对齐")

#             # 计算SSIM和PSNR
#             ssim_val = calculate_ssim(img_gt, img_processed, test_y_channel=test_y_channel)
#             psnr_val = calculate_psnr(img_gt, img_processed, test_y_channel=test_y_channel)

#             # 保存结果
#             ssim_list.append(ssim_val)
#             psnr_list.append(psnr_val)

#             # 打印单张结果（可选）
#             tqdm.write(f"文件 {file_name}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}")

#         except Exception as e:
#             tqdm.write(f"处理 {gt_file} 时出错：{str(e)}")
#             failed_files.append(gt_file)
#             continue

#     # 4. 计算平均值
#     avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
#     avg_psnr = np.mean(psnr_list) if psnr_list else 0.0

#     # 5. 输出汇总结果
#     print("\n" + "="*50)
#     print(f"批量计算完成！")
#     print(f"成功处理文件数：{len(ssim_list)}")
#     print(f"失败/跳过文件数：{len(failed_files)}")
#     if failed_files:
#         print(f"失败文件列表：{failed_files}")
#     print(f"平均SSIM (Y通道)：{avg_ssim:.4f}")
#     print(f"平均PSNR (Y通道)：{avg_psnr:.2f}")
#     print("="*50)

#     return avg_ssim, avg_psnr


# ========== 核心：批量计算的主函数 ==========
def batch_calculate_metrics(gt_dir, processed_dir, test_y_channel=True, use_gpu=True):
    """
    批量计算两个目录下对应图像的 SSIM, PSNR 和 LPIPS
    Args:
        gt_dir: 真值图像目录
        processed_dir: 处理后图像目录
        test_y_channel: 是否在YCbCr的Y通道计算PSNR/SSIM（LPIPS固定使用RGB）
        use_gpu: 是否使用 GPU 计算 LPIPS 以加速
    Returns:
        平均SSIM、平均PSNR、平均LPIPS
    """
    # 0. 初始化 LPIPS 模型 (放在循环外，防止显存爆炸！)
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"正在加载 LPIPS 模型至 {device}...")
    # 可选 net='alex' 或 net='vgg'，底层视觉评价通常默认用 alex
    lpips_loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
    lpips_loss_fn.eval() # 开启推理模式

    # 1. 获取真值目录下的所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    gt_files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in img_extensions]
    if not gt_files:
        raise ValueError(f"真值目录 {gt_dir} 下未找到图片文件！")
    
    # 2. 初始化指标列表
    ssim_list = []
    ssim_list_thin = []
    ssim_list_moderate = []
    ssim_list_thick = []
    psnr_list = []
    psnr_list_thin = []
    psnr_list_moderate = []
    psnr_list_thick = []

    lpips_list = []  
    lpips_list_thin = []
    lpips_list_moderate = []
    lpips_list_thick = []
    failed_files = []

    # 3. 循环处理每张图像
    for gt_file in tqdm(gt_files, desc="计算 PSNR/SSIM/LPIPS", unit="img"):
        try:
            file_name = os.path.splitext(gt_file)[0]
            processed_file = None
            for f in os.listdir(processed_dir):
                if os.path.splitext(f)[0] == file_name:
                    processed_file = f
                    break
            if not processed_file:
                tqdm.write(f"警告：{processed_dir} 中未找到 {file_name} 对应的文件，跳过")
                failed_files.append(gt_file)
                continue

            gt_path = os.path.join(gt_dir, gt_file)
            processed_path = os.path.join(processed_dir, processed_file)

            img_gt = read_image(gt_path)
            img_processed = read_image(processed_path)

            if img_gt.shape != img_processed.shape:
                img_processed = cv2.resize(img_processed, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # --- 核心指标计算 ---
            # 1 & 2. PSNR 和 SSIM (依据你的设置，可能只算Y通道)
            ssim_val = calculate_ssim(img_gt, img_processed, test_y_channel=test_y_channel)
            psnr_val = calculate_psnr(img_gt, img_processed, test_y_channel=test_y_channel)
            
            # 3. LPIPS 计算 (强制要求 RGB 格式计算，不转 Y 通道)
            lpips_val = calculate_lpips(img_processed, img_gt, lpips_loss_fn, device=device)

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)
            lpips_list.append(lpips_val)
            if 'thin' in file_name.lower():
                ssim_list_thin.append(ssim_val)
                psnr_list_thin.append(psnr_val)
                lpips_list_thin.append(lpips_val)
            elif 'moderate' in file_name.lower():
                ssim_list_moderate.append(ssim_val)
                psnr_list_moderate.append(psnr_val)
                lpips_list_moderate.append(lpips_val)
            elif 'thick' in file_name.lower():
                ssim_list_thick.append(ssim_val)
                psnr_list_thick.append(psnr_val)
                lpips_list_thick.append(lpips_val)

            # 可选：打印单张信息（如果觉得刷屏可以注释掉）
            tqdm.write(f"文件 {file_name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

        except Exception as e:
            tqdm.write(f"处理 {gt_file} 时出错：{str(e)}")
            failed_files.append(gt_file)
            continue

    # 4. 计算平均值
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim_thin = np.mean(ssim_list_thin) if ssim_list_thin else 0.0
    avg_ssim_moderate = np.mean(ssim_list_moderate) if ssim_list_moderate else 0.0
    avg_ssim_thick = np.mean(ssim_list_thick) if ssim_list_thick else 0.0
    avg_psnr_thin = np.mean(psnr_list_thin) if psnr_list_thin else 0.0
    avg_psnr_moderate = np.mean(psnr_list_moderate) if psnr_list_moderate else 0.0
    avg_psnr_thick = np.mean(psnr_list_thick) if psnr_list_thick else 0.0
    avg_lpips = np.mean(lpips_list) if lpips_list else 0.0
    avg_lpips_thin = np.mean(lpips_list_thin) if lpips_list_thin else 0.0
    avg_lpips_moderate = np.mean(lpips_list_moderate) if lpips_list_moderate else 0.0
    avg_lpips_thick = np.mean(lpips_list_thick) if lpips_list_thick else 0.0

    # metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=gt_dir,
    #     input2=processed_dir,
    #     cuda=True, 
    #     isc=True, 
    #     fid=True, 
    #     kid=True, 
    #     prc=True, 
    #     verbose=False,
    #     kid_subset_size=62
    # )

    # print("计算完成，结果如下：")
    # for metric_name, metric_value in metrics_dict.items():
    #     print(f"{metric_name}: {metric_value:.4f}")

    # 5. 输出汇总结果
    print("\n" + "="*50)
    print(f"批量计算完成！")
    print(f"成功处理文件数：{len(ssim_list)}")
    print(f"失败/跳过文件数：{len(failed_files)}")
    if failed_files:
        print(f"失败文件列表：{failed_files}")
    print(f"平均SSIM (Y通道)：{avg_ssim:.4f}")
    print(f"平均PSNR (Y通道)：{avg_psnr:.2f}")
    print(f"平均LPIPS (RGB通道)：{avg_lpips:.4f}")
    print(f"平均SSIM (Y通道) - 细雾：{avg_ssim_thin:.4f}")
    print(f"平均SSIM (Y通道) - 中等雾：{avg_ssim_moderate:.4f}")
    print(f"平均SSIM (Y通道) - 厚雾：{avg_ssim_thick:.4f}")
    print(f"平均PSNR (Y通道) - 细雾：{avg_psnr_thin:.2f}")
    print(f"平均PSNR (Y通道) - 中等雾：{avg_psnr_moderate:.2f}")
    print(f"平均PSNR (Y通道) - 厚雾：{avg_psnr_thick:.2f}")
    print(f"平均LPIPS (RGB通道) - 细雾：{avg_lpips_thin:.4f}")
    print(f"平均LPIPS (RGB通道) - 中等雾：{avg_lpips_moderate:.4f}")
    print(f"平均LPIPS (RGB通道) - 厚雾：{avg_lpips_thick:.4f}")

    print("="*50)


    print("="*50)

    return avg_psnr, avg_ssim, avg_lpips
# ========== 运行主函数 ==========
if __name__ == "__main__":
    # 请修改为你的实际目录路径
    GT_DIR = "/root/autodl-tmp/datasets/Haze1k_ddm/test_with_val/GT"
    PROCESSED_DIR = "/root/DFG-DDM/test_results/images_phy/Sate/Haze1k/fft-diffusion/processed"

    GT_DIR = "/root/autodl-tmp/datasets/Haze1k_ddm/test/GT"
    PROCESSED_DIR = "/root/DFG-DDM/test_results/images_phy/Sate/Haze1k/fft-diffusion/processed_onlytest"

    GT_DIR = "/root/autodl-tmp/datasets/Haze1k_ddm/test/GT"
    PROCESSED_DIR = "/root/DFG-DDM/test_results/images_phy_new/Sate/Haze1k/fft-diffusion/processed"
    GT_DIR = "../datasets/Haze1k_ddm_thick/test/GT"
    PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/images_baseline_thick_new/Sate/Haze1k/fft-diffusion/processed"

    GT_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/dehamer_ckpts_epoch132/gt"
    PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/dehamer_ckpts_epoch132/output"
    GT_DIR = "../datasets/Haze1k_ddm/test/GT"
    PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/images_baseline/Sate/Haze1k/fft-diffusion/processed"
    # GT_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/dformer_ckpts_epoch132/gt"
    # PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/dformer_ckpts_epoch132/output"

    # GT_DIR = "../datasets/Haze1k_ddm_thick/test/GT"
    # PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/images_baseline_thick_new_85000/Sate/Haze1k/fft-diffusion/processed"

    # GT_DIR = "../datasets/Haze1k_ddm/test/GT"
    # PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/images_baseline/Sate/Haze1k/fft-diffusion/processed"
    # GT_DIR = "../../datasets/Haze1k_ddm_thick/test/GT"
    # PROCESSED_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_fft_55000/Sate/Haze1k/fft-diffusion"

    GT_DIR = "../../datasets/Haze1k_ddm2/test/GT"
    PROCESSED_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/images_all_1640000/Sate/Haze1k/fft-diffusion/processed"
    # PROCESSED_DIR = "/home/yueyinlei/paper/DFG-DDM/test_results/zhao/gridDehazy/dehazed"
    # PROCESSED_DIR = "/home/yueyinlei/paper/new_path/DFG-DDM/test_results/GridDehazeNet/"
    # 执行批量计算
    avg_ssim, avg_psnr, avg_lpips = batch_calculate_metrics(GT_DIR, PROCESSED_DIR, test_y_channel=True, use_gpu=False)

# img1 = read_image('/root/DFG-DDM/test_results/images_phy/Sate/Haze1k/fft-diffusion/gt/3.jpg')
# img2 = read_image('/root/DFG-DDM/test_results/images_phy/Sate/Haze1k/fft-diffusion/processed/3.png')
# ssim_value = calculate_ssim(img1, img2, test_y_channel=True)
# print(f'SSIM: {ssim_value:.4f}')

# val and test :平均SSIM (Y通道)：0.8470.  平均PSNR (Y通道)：24.16
# only test: 平均SSIM (Y通道)：0.8616   平均PSNR (Y通道)：24.50

#  python test.py --config 'haze.yml' --test_set 'Haze1k'
#  python train.py --resume /root/DFG-DDM/ckpts/phy/Sate_ddpm_data_fft_95000.pth.tar --config 'haze.yml' 2>&1 | tee -a logs/train_phy_log.txt