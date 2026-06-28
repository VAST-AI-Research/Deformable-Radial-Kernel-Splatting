import torch
try:
    from piq import ssim, LPIPS
    lpips = LPIPS()
except ImportError:
    from utils.loss_utils import ssim

    class _L1LPIPSFallback(torch.nn.Module):
        def forward(self, img1, img2):
            return torch.abs(img1 - img2).mean(dim=(1, 2, 3))

    lpips = _L1LPIPSFallback()


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def metric(img, pred):
    # img: (3, H, W)
    psnr_val = psnr(img, pred).mean().double()
    ssim_val = ssim(img[None], pred[None]).mean()
    lpips_val = lpips(img[None], pred[None]).mean()
    return psnr_val, ssim_val, lpips_val
