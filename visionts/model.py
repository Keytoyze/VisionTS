import torch

import os

from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class VisionTS(nn.Module):

    def __init__(self, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/', load_ckpt=True):
        super(VisionTS, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0]()

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()
    

    def forward(self, x, export_image=False, fp64=False):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        # 1. Normalization
        means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
        stdev /= self.norm_const
        x_enc /= stdev
        # Channel Independent
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s') # [bs x nvars x seq_len]

        # 2. Segmentation
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') # [b n s]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([
            x_resize, 
            masked
        ], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        # 4. Reconstruction
        _, y, mask = self.vision_model(
            image_input, 
            mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
        )
        image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        
        # 5. Forecasting
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey
        y_segmentations = self.output_resize(y_grey) # resize back
        y_flatten = einops.rearrange(
            y_segmentations, 
            '(b n) 1 f p -> b (p f) n', 
            b=x_enc.shape[0], f=self.periodicity
        ) # flatten
        y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

        # 6. Denormalization
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))

        if export_image:
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = self.vision_model.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask)
            image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
            green_bg = -torch.ones_like(image_reconstructed) * 2
            image_input = image_input * (1 - mask) + green_bg * mask
            image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
            image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            return y, image_input, image_reconstructed
        return y

