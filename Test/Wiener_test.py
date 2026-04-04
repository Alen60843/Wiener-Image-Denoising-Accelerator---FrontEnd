import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


# helpers
""" ref = clean image, test = noisy image"""
def snr_booklet(ref, test):
    """SNR = sum(ref^2) / sum((ref - test)^2)"""
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    num = np.sum(ref ** 2)
    den = np.sum((ref - test) ** 2)
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)


def psnr_booklet(ref, test):
    """PSNR = 10 log10( MAXI^2 / sum((ref - test)^2) )"""    
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    I_max = np.max(test) ** 2
    err_sum = np.sum((ref - test) ** 2)
    if err_sum == 0:
        return np.inf
    return 10.0 * np.log10((I_max ** 2) / err_sum)


def local_mean_var(img, win_size=5):
    """Local mean and variance using uniform filter."""
    img = img.astype(np.float64)
    local_mean = uniform_filter(img, size=win_size, mode="reflect")
    local_mean_sq = uniform_filter(img ** 2, size=win_size, mode="reflect")
    local_var = local_mean_sq - local_mean ** 2
    return local_mean, np.maximum(local_var, 0.0)



def estimate_noise_var_immerkaer(img):
    """
    Noise variance estimator (Immerkaer method).
    1) Blur a bit to remove image structure
    2) Take residual (mostly noise)
    3) Apply 3x3 high-pass filter (Immerkaer mask)
    4) Compute mean(abs(.)) using the Immerkaer constant
    """
    img = img.astype(np.float64)

    # Step 1–2: remove low-frequency content, remain with mostly noise
    smooth = uniform_filter(img, size=5, mode="reflect")
    resid = img - smooth  # residual = image - blurred image

    # Step 3: Apply Immerkaer 3x3 high-pass kernel
    k = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ], dtype=np.float64)

    # "valid" mode ensures convolution ignores borders (output = input - 2 pixels)
    conv = convolve2d(resid, k, mode="valid", boundary="symm")
    '''
    # plot the smooth residual and convolved images for debugging
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 2)
    plt.imshow(resid, cmap="gray")
    plt.title("Residual Image")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(conv, cmap="gray")
    plt.title("Convolved with HPF Kernel")
    plt.axis("off")
    plt.subplot(1, 3, 1)
    plt.imshow(smooth, cmap="gray")
    plt.title("Smooth Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    '''
    # h_valid and w_valid are the height and width of the valid convolution result,
    # which equals resid size - 2 (because of 3x3 kernel)
    h_valid, w_valid = conv.shape
    mean_abs = np.sum(np.abs(conv)) / (h_valid * w_valid)

    # Step 4: Immerkaer formula to calculate sigma (noise std)
    sigma = np.sqrt(np.pi / 2.0) * (mean_abs / 6.0)

    # Final estimated noise variance (variance = sigma^2)
    #if sigma ** 2 < 400:
     #   return sigma ** 2 
    #else:
    if sigma <= 20:
        return sigma ** 2
    else:
        return 1.25 * (sigma ** 2) 


def adaptive_wiener(noisy, win_size=5):
    """
    Adaptive Wiener filter using global noise variance estimated
    by the Immerkaer method.
    """
    noisy = noisy.astype(np.float64)

    # Estimate global noise variance
    noise_var = estimate_noise_var_immerkaer(noisy)

    # Compute local mean and variance for each pixel window
    mu, sigma2 = local_mean_var(noisy, win_size=win_size)  # mu = local mean, sigma2 = local variance = sigma^2

    # Avoid division by zero or very small numbers
    sigma2_safe = np.maximum(sigma2, 1e-8)

    # Wiener weights
    # w = max(0, 1 - noise_var / sigma2)
    w = np.clip(1.0 - (noise_var / sigma2_safe), 0.0, 1.0)   # Optional boost factor to enhance denoising

    # Final Wiener estimate:
    # u_hat = mean + w * (v - mean)
    restored = mu + w * (noisy - mu )  # Optional boost factor to enhance denoising

    return restored, noise_var


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Wiener with SNR/PSNR/SSIM and Immerkaer noise estimation (Gaussian noise only)."
    )
    parser.add_argument("--image-path", type=str, default="images/house.jpg",help="Path to input image (grayscale or RGB).")
    parser.add_argument("--snr-db", type=float, default=10.0,help="Target SNR in dB for the added Gaussian noise.")
    parser.add_argument("--win-size", type=int, default=5,help="Window size for local statistics.")
    parser.add_argument("--no-show", dest="no_show", action="store_true",help="If set, do not show images.")
    parser.add_argument("--seed", type=int, default=0,help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    img = io.imread(args.image_path)

    # Convert to grayscale in 0–255 range
    if img.ndim == 2:
        # Already grayscale (H, W)
        img_gray = img.astype(np.uint8)

    elif img.ndim == 3:
        # Could be (H, W, 3) RGB, (H, W, 4) RGBA, or (1, H, W), etc.

        if img.shape[-1] == 3:
            # RGB (H, W, 3)
            img_gray = (color.rgb2gray(img) * 255).astype(np.uint8)

        elif img.shape[-1] == 4:
            # RGBA (H, W, 4) -> drop alpha, then convert
            img_rgb = img[:, :, :3]
            img_gray = (color.rgb2gray(img_rgb) * 255).astype(np.uint8)

        elif img.shape[0] == 1:
            # Single-channel but channel-first: (1, H, W)
            img_gray = img[0, :, :].astype(np.uint8)

        else:
            # Fallback: try to squeeze singleton dims
            img_squeezed = np.squeeze(img)

            if img_squeezed.ndim == 2:
                img_gray = img_squeezed.astype(np.uint8)
            elif img_squeezed.ndim == 3 and img_squeezed.shape[-1] == 3:
                img_gray = (color.rgb2gray(img_squeezed) * 255).astype(np.uint8)
            else:
                raise ValueError("Unsupported image shape for grayscale conversion: " + str(img.shape))

    else:
        raise ValueError("Unsupported image dimensions: " + str(img.shape))


    original = img_gray.astype(np.float64)

    # Signal power = E[original^2]
    signal_power = np.mean(original ** 2)
    if signal_power == 0:
        raise ValueError("Signal power is zero; cannot define SNR for a blank image.")
    
    # Compute noise variance from target SNR (in dB)
    # SNR linear = 10^(SNR_dB/10)
    snr_linear = 10 ** (args.snr_db / 10.0)
    noise_var = signal_power / snr_linear
    noise_sigma = np.sqrt(noise_var)

    # Generate and add Gaussian noise
    noise = np.random.normal(0, noise_sigma, size=original.shape)
    noisy = np.clip(original + noise, 0, 255)

    true_noise_var = noise_var
    true_var_str = f"{true_noise_var:.4f}"


    # Apply Wiener filter (with Immerkaer noise variance estimation)
    restored, est_noise_var = adaptive_wiener(noisy, win_size=args.win_size)

    restored = np.clip(restored, 0, 255)

    # Compute quality metrics
    snr_noisy = snr_booklet(original, noisy)
    snr_rest = snr_booklet(original, restored)
    psnr_noisy = psnr_booklet(original, noisy)
    psnr_rest = psnr_booklet(original, restored)
    ssim_rest = ssim(original, restored, data_range=255)

    # Print results
    print(f"Image: {args.image_path}")
    print(f"Target SNR (dB)            = {args.snr_db}")
    print(f"True noise sigma           = {noise_sigma:.4f}")
    print(f"True noise variance        = {true_var_str}")
    print(f"Estimated noise variance   = {est_noise_var:.4f}")
    print(f"SNR (original vs noisy)    = {snr_noisy:.4f} dB")
    print(f"SNR (original vs restored) = {snr_rest:.4f} dB")
    print(f"PSNR (original vs noisy)   = {psnr_noisy:.4f} dB")
    print(f"PSNR (original vs restored)= {psnr_rest:.4f} dB")
    print(f"SSIM (original vs restored)= {ssim_rest:.4f}")

    # Show images
    if not args.no_show:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(noisy, cmap="gray", vmin=0, vmax=255)
        axs[1].set_title(f"Noisy (Gaussian SNR={args.snr_db} dB)")
        axs[1].axis("off")

        axs[2].imshow(restored, cmap="gray", vmin=0, vmax=255)
        axs[2].set_title("Restored (Wiener Filter)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
