import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# -------------------------- 核心参数配置 --------------------------
# 图像路径（替换成你自己的清晰图像路径）
IMG_PATH = "/home/yueyinlei/paper/new_path/DFG-DDM/image_clear.png"  # 建议用RGB图像，尺寸建议256x256
# 保存加噪图像的文件夹（自动创建）
SAVE_DIR = "diffusion_noise_process"
# 总时间步（T越大，加噪过程越细腻，建议T=10/20/50，方便绘图展示）
T = 20
# 加噪策略："linear"（线性）或 "cosine"（余弦，更贴合主流扩散模型）
NOISE_SCHEDULE = "cosine"
# 设备（自动选择GPU/CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 扩散模型加噪核心函数 --------------------------
def get_beta_schedule(timesteps, schedule="linear", start=0.0001, end=0.02):
    """
    生成噪声调度（beta_t），控制不同时间步的加噪强度
    :param timesteps: 总时间步T
    :param schedule: 加噪策略：linear（线性）/cosine（余弦）
    :param start: 初始beta值（t=0）
    :param end: 最终beta值（t=T-1）
    :return: beta_t: [T]
    """
    if schedule == "linear":
        # 线性调度：beta从start线性增加到end（简单直观，适合展示）
        beta_t = torch.linspace(start, end, timesteps, device=DEVICE)
    elif schedule == "cosine":
        # 余弦调度：beta先慢后快（主流扩散模型如DDPM用此策略）
        steps = torch.linspace(0, timesteps, timesteps + 1, device=DEVICE)
        alpha_t_bar = torch.cos(((steps / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_t_bar = alpha_t_bar / alpha_t_bar[0]
        beta_t = 1 - (alpha_t_bar[1:] / alpha_t_bar[:-1])
        beta_t = torch.clamp(beta_t, 0.0001, 0.9999)
    else:
        raise ValueError(f"不支持的加噪策略：{schedule}")
    return beta_t

def diffusion_add_noise(x_0, beta_t, t):
    """
    对清晰图像x_0，按时间步t添加噪声（前向加噪过程）
    公式：x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * ε （ε~N(0,1)）
    :param x_0: 清晰图像，shape=[C, H, W]，值∈[0,1]
    :param beta_t: 噪声调度，shape=[T]
    :param t: 当前时间步（0<=t<T）
    :return: x_t: 加噪后的图像, noise: 采样的高斯噪声
    """
    # 计算累积乘积alpha_t_bar = product_{s=0}^t (1 - beta_s)
    alpha_t = 1 - beta_t
    alpha_t_bar = torch.cumprod(alpha_t, dim=0)  # [T]
    
    # 取当前时间步t对应的alpha_t_bar
    alpha_t_bar_t = alpha_t_bar[t].view(-1, 1, 1)  # 广播到[C, H, W]
    
    # 采样高斯噪声（和图像同形状）
    noise = torch.randn_like(x_0, device=DEVICE)
    
    # 前向加噪公式
    x_t = torch.sqrt(alpha_t_bar_t) * x_0 + torch.sqrt(1 - alpha_t_bar_t) * noise
    
    # 限制值范围在[0,1]（避免绘图时出现异常值）
    x_t = torch.clamp(x_t, 0.0, 1.0)
    return x_t, noise

# -------------------------- 图像加载与预处理 --------------------------
def load_and_preprocess_image(img_path):
    """加载图像并预处理为模型输入格式"""
    # 读取图像（BGR→RGB，归一化到[0,1]，转为Tensor）
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
    img_rgb = cv2.resize(img_rgb, (512, 512))  # 统一尺寸（方便展示）
    # 转为Tensor: [H, W, C] → [C, H, W]，归一化到[0,1]
    x_0 = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    x_0 = x_0.to(DEVICE)
    return x_0, img_rgb

# -------------------------- 主流程：生成加噪图像并保存 --------------------------
def main():
    # 1. 创建保存文件夹
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. 加载清晰图像
    x_0, img_original = load_and_preprocess_image(IMG_PATH)
    # 保存原始清晰图像
    cv2.imwrite(os.path.join(SAVE_DIR, "00_original.jpg"), 
                cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))
    print(f"原始图像已保存：{os.path.join(SAVE_DIR, '00_original.jpg')}")
    
    # 3. 生成噪声调度（beta_t）
    beta_t = get_beta_schedule(T, schedule=NOISE_SCHEDULE)
    
    # 4. 逐时间步加噪并保存图像
    for t in range(T):
        # 对清晰图像加噪（t=0: 几乎无噪；t=T-1: 完全噪声）
        x_t, noise = diffusion_add_noise(x_0, beta_t, t)
        
        # 将Tensor转为numpy（方便保存）：[C, H, W] → [H, W, C]
        x_t_np = x_t.permute(1, 2, 0).cpu().numpy()
        # 归一化到[0,255]（图像保存要求）
        x_t_np = (x_t_np * 255).astype(np.uint8)
        
        # 保存加噪图像（命名格式：步长_加噪图像.jpg）
        save_path = os.path.join(SAVE_DIR, f"{t+1:02d}_noise_step_{t}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(x_t_np, cv2.COLOR_RGB2BGR))
        
        # 打印进度（展示加噪强度）
        alpha_t_bar = torch.cumprod(1 - beta_t, dim=0)[t]
        noise_strength = 1 - alpha_t_bar.item()  # 加噪强度（0~1）
        print(f"时间步t={t} | 加噪强度={noise_strength:.4f} | 保存路径：{save_path}")
    
    # 5. 绘制加噪强度曲线（方便绘图展示时用）
    alpha_t_bar = torch.cumprod(1 - beta_t, dim=0).cpu().numpy()
    noise_strength = 1 - alpha_t_bar  # 加噪强度=1 - alpha_t_bar
    plt.figure(figsize=(8, 4))
    plt.plot(range(T), noise_strength, color="#2E86AB", linewidth=2)
    plt.xlabel("时间步 t")
    plt.ylabel("加噪强度（1 - α̅_t）")
    plt.title(f"扩散模型加噪强度变化（{NOISE_SCHEDULE}调度）")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "noise_strength_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n加噪强度曲线已保存：{os.path.join(SAVE_DIR, 'noise_strength_curve.png')}")
    print(f"所有加噪图像已保存到：{SAVE_DIR}")

if __name__ == "__main__":
    main()