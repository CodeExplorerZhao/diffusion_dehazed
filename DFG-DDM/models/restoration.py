import torch
import torch.nn as nn
import utils
import torchvision
import os
from tqdm import tqdm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='snow', r=None):
        output_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation, "fft-diffusion", "processed")
        input_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation, "fft-diffusion", "original")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(input_folder, exist_ok=True)
        with torch.no_grad():
            qbar = tqdm(val_loader, desc="Processing Validation Set", unit="img", colour="blue")
            # 2. 用tqdm包裹循环，添加进度条描述和单位
            for i, (batch_data, y) in enumerate(qbar):
                y = y[0] if isinstance(y, list) or isinstance(y, tuple) else y
                batch, t0, A0 = batch_data
                if t0.numel() == 0:  # numel()=0 表示空张量
                    t0 = None
                else: t0 = t0.flatten(0, 1).to(self.diffusion.device)
                if A0.numel() == 0:
                    A0 = None
                else:
                    A0 = A0.flatten(0, 1).to(self.diffusion.device)
                x = batch.to(self.diffusion.device)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r, tau_hint=t0)
                x_output = inverse_data_transform(x_output)
                utils.logging.save_image(x_output, os.path.join(output_folder, f"{y}.png"))
                utils.logging.save_image(x_cond, os.path.join(input_folder, f"{y}.png"))

    def diffusive_restoration(self, x_cond, r=None, tau_hint=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size, tau_hint=tau_hint)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
