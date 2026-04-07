import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm  # 用于显示进度条
import json

def _dark_channel(I, patch=15):
    # I: [B,3,H,W] in [0,1]
    # dark = min_c I, then min over local window
    min_c = I.min(dim=1, keepdim=True)[0]  # [B,1,H,W]
    # minpool = -maxpool(-x)
    pad = patch // 2
    dark = -F.max_pool2d(-min_c, kernel_size=patch, stride=1, padding=pad)
    return dark  # [B,1,H,W]

def _box_filter(x, r):
    # fast box filter via avg_pool
    x_padded = F.pad(x, (r, r, r, r), mode='replicate')
    k = 2 * r + 1
    return F.avg_pool2d(x_padded, kernel_size=k, stride=1, padding=0)

def _guided_filter_gray(I_gray, p, r=40, eps=1e-3):
    """
    Guided filter for single-channel guidance I_gray and single-channel input p.
    I_gray: [B,1,H,W], p: [B,1,H,W]
    """
    mean_I = _box_filter(I_gray, r)
    mean_p = _box_filter(p, r)
    mean_Ip = _box_filter(I_gray * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(I_gray * I_gray, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    q = mean_a * I_gray + mean_b
    return q


def estimate_A_tau_dcp(
    I_01: torch.Tensor,
    patch: int = 21,
    omega: float = 0.95,
    t_min: float = 0.1,
    refine: bool = True,
    gf_r: int = 40,
    gf_eps: float = 3e-1,
    top_percent: float = 0.001,
):
    """
    Dark Channel Prior (DCP) estimation.
    Args:
      I_01: [B,3,H,W] float in [0,1]
      patch: dark channel minpool window
      omega: haze strength
      t_min: lower bound on transmission
      refine: whether to apply guided filter refinement
      gf_r, gf_eps: guided filter params
      top_percent: top brightest pixels in dark channel for estimating A
    Returns:
      A0: [B,3,1,1]
      t0: [B,1,H,W]
    """
    assert I_01.ndim == 4 and I_01.size(1) == 3, "I_01 must be [B,3,H,W]"
    B, _, H, W = I_01.shape
    device = I_01.device

    # 1) Dark channel of hazy image
    dark = _dark_channel(I_01, patch=patch)  # [B,1,H,W]

    # 2) Atmospheric light A0 from top bright pixels in dark channel
    # pick top k pixels by dark value
    k = max(1, int(top_percent * H * W))
    dark_flat = dark.view(B, -1)  # [B,HW]
    vals, idx = torch.topk(dark_flat, k=k, dim=1, largest=True, sorted=False)  # idx: [B,k]

    # gather corresponding RGB pixels from I
    I_flat = I_01.permute(0, 2, 3, 1).contiguous().view(B, -1, 3)  # [B,HW,3]
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)                  # [B,k,3]
    top_pixels = torch.gather(I_flat, dim=1, index=idx_exp)        # [B,k,3]

    # use mean of top pixels as A (stable). You can also use max if you want.
    A = top_pixels.mean(dim=1)                                     # [B,3]
    A0 = A.view(B, 3, 1, 1).clamp(1e-3, 1.0)

    # 3) Transmission estimation: t = 1 - omega * dark_channel(I / A)
    I_div_A = I_01 / A0
    dark_div = _dark_channel(I_div_A, patch=patch)
    t0 = 1.0 - omega * dark_div
    t0 = t0.clamp(t_min, 1.0)  # [B,1,H,W]

    # 4) Optional refinement (recommended)
    if refine:
        # guidance: grayscale of I
        I_gray = (0.299 * I_01[:, 0:1] + 0.587 * I_01[:, 1:2] + 0.114 * I_01[:, 2:3]).clamp(0, 1)
        t0 = _guided_filter_gray(I_gray, t0, r=gf_r, eps=gf_eps).clamp(t_min, 1.0)
    

    return A0, t0

def main():
    input_dir = "/home/yueyinlei/paper/datasets/Haze1k_ddm/train/hazy"
    output_dir = "/home/yueyinlei/paper/datasets/Haze1k_ddm/train/tau"
    
    # 新增：定义保存 A 的 JSON 文件路径
    a_dict_path = "/home/yueyinlei/paper/datasets/Haze1k_ddm/train/A_values.json"
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    # 用于在内存中积攒所有图片的 A 值
    A_dict = {}

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Processing Transmission and A"):
            img = Image.open(img_path).convert("RGB")
            I_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            
            A0, t0 = estimate_A_tau_dcp(I_tensor, patch=15, omega=0.95, refine=True)
            
            # --- 1. 保存 tau 图 ---
            tau_map = t0.squeeze().cpu().numpy()
            tau_img = Image.fromarray((tau_map * 255.0).clip(0, 255).astype('uint8'), mode='L')
            
            filename = os.path.basename(img_path)
            name_only, ext = os.path.splitext(filename)
            save_name = f"{name_only}{ext}"
            tau_img.save(os.path.join(output_dir, save_name))
            
            # --- 2. 记录 A 值 ---
            # 把 A0 [1, 3, 1, 1] 变成普通的 python 列表 [R, G, B]
            A_list = A0.squeeze().cpu().tolist()
            A_dict[filename] = A_list

    # 循环结束后，统一将 A_dict 写入 JSON 文件
    with open(a_dict_path, 'w') as f:
        json.dump(A_dict, f, indent=4)
    print(f"\nAll A values saved to {a_dict_path}")

if __name__ == "__main__":
    main()

    # ls -1 ./train/GT > ./train/train.txt