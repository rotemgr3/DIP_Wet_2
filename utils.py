import matplotlib.pyplot as plt
import numpy as np


def compute_psnr(clean, denoised):
    mse = ((clean - denoised) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def plot_images(noisy, clean, denoised, title="Output Example"):
    psnr = compute_psnr(clean, denoised)
    plt.figure(figsize=(12, 4))
    for i, img in enumerate([noisy, clean, denoised], start=1):
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"{title} - PSNR: {psnr:.2f} dB")
    plt.show()