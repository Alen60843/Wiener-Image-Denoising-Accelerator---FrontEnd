import argparse
import numpy as np
import cv2
import os
import math
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def read_pixels_from_file(file_path):
    pixels = []
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            if ext == ".hex":
                val = int(line, 16)
            else:
                val = int(line)

            if val < 0:
                val = 0
            if val > 255:
                val = 255

            pixels.append(val)

    return pixels


def fix_length(pixels, expected_pixels):
    if len(pixels) > expected_pixels:
        pixels = pixels[:expected_pixels]
    elif len(pixels) < expected_pixels:
        missing = expected_pixels - len(pixels)
        pixels = pixels + [0] * missing
    return pixels


def pixels_to_img(pixels, height, width):
    arr = np.array(pixels, dtype=np.uint8).reshape((height, width))
    return arr


def calc_snr_db_formula(original, estimate):
    o = original.astype(np.float64)
    e = estimate.astype(np.float64)

    num = np.sum(o * o)
    den = np.sum((o - e) * (o - e))

    if den == 0:
        return float("inf")
    if num == 0:
        return float("-inf")

    snr_linear = num / den
    return 10.0 * math.log10(snr_linear)


def calc_psnr_db_formula(original, estimate):
    o = original.astype(np.float64)
    e = estimate.astype(np.float64)

    error_sum = np.sum((o - e) * (o - e))
    if error_sum == 0:
        return float("inf")

    # Keep your style, but make MAX_I consistent for 8-bit images:
    max_i = 255.0
    psnr_linear = (max_i * max_i) / error_sum
    return 10.0 * np.log10(psnr_linear)


def calc_ssim_skimage(original, estimate):
    score, _ = ssim(original, estimate, data_range=255, full=True)
    return float(score)


def load_cbcr_from_rgb(rgb_src_path, size):
    """
    Load an RGB image, resize to (size,size), convert to YCbCr,
    and return (Cb, Cr) as uint8 HxW arrays.
    """
    img = Image.open(rgb_src_path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    ycbcr = img.convert("YCbCr")
    ycbcr_np = np.array(ycbcr, dtype=np.uint8)  # HxWx3: Y, Cb, Cr
    cb = ycbcr_np[..., 1]
    cr = ycbcr_np[..., 2]
    return cb, cr


def y_cb_cr_to_rgb_image(y_gray, cb, cr):
    """
    y_gray: HxW uint8
    cb, cr: HxW uint8
    returns: HxWx3 uint8 in RGB order
    """
    ycbcr = np.stack([y_gray, cb, cr], axis=-1).astype(np.uint8)  # HxWx3
    rgb_pil = Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")
    return np.array(rgb_pil, dtype=np.uint8)


def gray_to_rgb_repeat(gray):
    """
    gray: HxW uint8
    returns HxWx3 uint8 where R=G=B=gray
    """
    return np.stack([gray, gray, gray], axis=-1)


def main():
    parser = argparse.ArgumentParser(
        description="Create grayscale image from pixel file and optionally compute SNR/PSNR/SSIM. "
                    "Can also reconstruct an RGB image using original chroma (Cb/Cr)."
    )

    parser.add_argument("--file", type=str, default="output_pixels.hex",
                        help="Input pixel file (.hex or .txt) to reconstruct and save")

    parser.add_argument("--out_img", type=str, default="filtered_output.png",
                        help="Output grayscale image filename")

    parser.add_argument("--size", type=int, default=128,
                        help="Image size (square). Example: 128 or 512")

    parser.add_argument("--show", action="store_true",
                        help="Show the reconstructed grayscale image on screen")

    #  RGB reconstruction options 
    parser.add_argument("--out_rgb", type=str, default=None,
                        help="If set, also save an RGB image. "
                             "If chroma is provided, RGB will preserve color; "
                             "otherwise it will be grayscale repeated into 3 channels.")

    parser.add_argument("--show_rgb", action="store_true",
                        help="Show the reconstructed RGB image too (requires --out_rgb or will just display).")

    parser.add_argument("--rgb_src", type=str, default=None,
                        help="Path to the original color image (RGB). "
                             "We will reuse its Cb/Cr to reconstruct color after denoising Y.")

    parser.add_argument("--cb_npy", type=str, default=None,
                        help="Optional path to saved Cb channel (.npy), shape HxW uint8")

    parser.add_argument("--cr_npy", type=str, default=None,
                        help="Optional path to saved Cr channel (.npy), shape HxW uint8")

    # Metrics inputs
    parser.add_argument("--clean", type=str, default=None,
                        help="Optional clean/original pixels file (example: clean_img.hex)")
    parser.add_argument("--noisy", type=str, default=None,
                        help="Optional noisy pixels file (example: noisy_img.txt)")
    parser.add_argument("--restored", type=str, default=None,
                        help="Optional restored pixels file (example: output_pixels_5x5.txt or output_pixels_3x3.txt)")

    args = parser.parse_args()

    height = args.size
    width = args.size
    expected_pixels = height * width

    #  Reconstruct grayscale image from file
    pixels_main = read_pixels_from_file(args.file)
    pixels_main = fix_length(pixels_main, expected_pixels)
    img_main = pixels_to_img(pixels_main, height, width)

    cv2.imwrite(args.out_img, img_main)
    print("Input file:", args.file)
    print("Output grayscale image:", args.out_img)
    print("Image size:", width, "x", height)
    print("Number of pixels used:", expected_pixels)

    if args.show:
        cv2.imshow("Reconstructed Grayscale", img_main)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #  Optional RGB reconstruction 
    if args.out_rgb is not None or args.show_rgb:
        cb = cr = None

        if args.rgb_src is not None:
            cb, cr = load_cbcr_from_rgb(args.rgb_src, args.size)
        elif args.cb_npy is not None and args.cr_npy is not None:
            cb = np.load(args.cb_npy).astype(np.uint8)
            cr = np.load(args.cr_npy).astype(np.uint8)

            # Basic shape safety
            if cb.shape != (height, width) or cr.shape != (height, width):
                raise ValueError(f"Cb/Cr shapes must be ({height},{width}). "
                                 f"Got cb={cb.shape}, cr={cr.shape}")

        if cb is not None and cr is not None:
            rgb = y_cb_cr_to_rgb_image(img_main, cb, cr)
            rgb_note = "RGB reconstructed using Y (denoised) + original Cb/Cr"
        else:
            rgb = gray_to_rgb_repeat(img_main)
            rgb_note = "RGB created by repeating grayscale into 3 channels (no color info provided)"

        # Save RGB if requested
        if args.out_rgb is not None:
            # cv2.imwrite expects BGR
            cv2.imwrite(args.out_rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print("Output RGB image:", args.out_rgb)
            print("RGB mode:", rgb_note)

        if args.show_rgb:
            cv2.imshow("Reconstructed RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #  Metrics 
    if args.clean is None:
        return

    clean_pixels = read_pixels_from_file(args.clean)
    clean_pixels = fix_length(clean_pixels, expected_pixels)
    img_clean = pixels_to_img(clean_pixels, height, width)

    if args.noisy is not None:
        noisy_pixels = read_pixels_from_file(args.noisy)
        noisy_pixels = fix_length(noisy_pixels, expected_pixels)
        img_noisy = pixels_to_img(noisy_pixels, height, width)

        snr_noisy = calc_snr_db_formula(img_clean, img_noisy)
        psnr_noisy = calc_psnr_db_formula(img_clean, img_noisy)
    else:
        snr_noisy = None
        psnr_noisy = None

    if args.restored is not None:
        restored_pixels = read_pixels_from_file(args.restored)
        restored_pixels = fix_length(restored_pixels, expected_pixels)
        img_restored = pixels_to_img(restored_pixels, height, width)

        snr_rest = calc_snr_db_formula(img_clean, img_restored)
        psnr_rest = calc_psnr_db_formula(img_clean, img_restored)
        ssim_rest = calc_ssim_skimage(img_clean, img_restored)
    else:
        snr_rest = None
        psnr_rest = None
        ssim_rest = None

    if snr_noisy is not None:
        print(f"SNR (original vs noisy)    = {snr_noisy:.3f} dB")
    if snr_rest is not None:
        print(f"SNR (original vs restored) = {snr_rest:.3f} dB")
    if psnr_noisy is not None:
        print(f"PSNR (original vs noisy)   = {psnr_noisy:.4f} dB")
    if psnr_rest is not None:
        print(f"PSNR (original vs restored)= {psnr_rest:.4f} dB")
    if ssim_rest is not None:
        print(f"SSIM (original vs restored)= {ssim_rest:.4f}")


if __name__ == "__main__":
    main()
