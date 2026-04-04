import argparse
import numpy as np
import cv2
import os
import math
from skimage.metrics import structural_similarity as ssim



def read_pixels_from_file(file_path):
    # Read pixel values from a HEX or TXT file into a Python list of integers.
    # .hex: each line is hex like "ff"
    # .txt: each line is decimal like "255"
    pixels = []

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if line == "":
                continue

            # Parse
            if ext == ".hex":
                val = int(line, 16)
            else:
                val = int(line)

            # Clip to 0..255 (8-bit grayscale)
            if val < 0:
                val = 0
            if val > 255:
                val = 255

            pixels.append(val)

    return pixels


def fix_length(pixels, expected_pixels):
    # Make sure we have exactly height*width pixels.
    if len(pixels) > expected_pixels:
        pixels = pixels[:expected_pixels]
    elif len(pixels) < expected_pixels:
        missing = expected_pixels - len(pixels)
        pixels = pixels + [0] * missing
    return pixels


def pixels_to_img(pixels, height, width):
    # Convert 1D pixel list to a 2D uint8 image
    arr = np.array(pixels, dtype=np.uint8).reshape((height, width))
    return arr


def calc_snr_db_formula(original, estimate):
    # Using your screenshot formula:
    # SNR_linear = ( sum sum f[x,y]^2 ) / ( sum sum (f[x,y] - fhat[x,y])^2 )
    # Then we print it in dB using: 10*log10(SNR_linear)
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
    # Using your screenshot formula:
    # PSNR = 10*log10( MAX_I^2 / ( sum sum (f - fhat)^2 ) )
    # MAX_I for 8-bit is 255
    #
    # Note: Many books use MSE = (1/(MN))*sum(sum(error^2)).
    # But you asked to follow the screenshot, so we use the SUM directly.
    o = original.astype(np.float64)
    e = estimate.astype(np.float64)

    error_sum = np.sum((o - e) * (o - e))  # sum of squared error

    if error_sum == 0:
        return float("inf")

    max_i = np.max(e) ** 2
    psnr_linear = max_i ** 2 / error_sum
    return 10.0 * np.log10(psnr_linear)


def calc_ssim_skimage(original, estimate):
    # original and estimate are 2D uint8 images
    # data_range=255 because we use 8-bit grayscale images
    score, _ = ssim(
        original,
        estimate,
        data_range=255,
        full=True
    )
    return float(score)



def main():
    parser = argparse.ArgumentParser(
        description="Create grayscale image from pixel file and optionally compute SNR/PSNR/SSIM"
    )

    # Main image reconstruction input
    parser.add_argument("--file", type=str, default="output_pixels.hex", help="Input pixel file (.hex or .txt) to reconstruct and save")

    # Output image filename for reconstructed image from --file
    parser.add_argument("--out_img", type=str, default="filtered_output.png", help="Output image file name for the reconstructed image")

    # Image size (square)
    parser.add_argument("--size", type=int, default=128, help="Image size (example: 128 or 512). Default = 128")

    # Optional: show the reconstructed image
    parser.add_argument("--show", action="store_true", help="Show the reconstructed image on screen")

    # Optional metric files
    parser.add_argument("--clean", type=str, default=None, help="Optional clean/original pixels file (example: clean_img.hex)")
    parser.add_argument("--noisy", type=str, default=None, help="Optional noisy pixels file (example: noisy_img.txt)")
    parser.add_argument("--restored", type=str, default=None, help="Optional restored pixels file (example: output_pixels_5x5.txt or output_pixels_3x3.txt)")

    args = parser.parse_args()

    height = args.size
    width = args.size
    expected_pixels = height * width

    # Reconstruct image from --file (always)
    pixels_main = read_pixels_from_file(args.file)
    pixels_main = fix_length(pixels_main, expected_pixels)
    img_main = pixels_to_img(pixels_main, height, width)

    # Save and print info
    cv2.imwrite(args.out_img, img_main)
    print("Input file:", args.file)
    print("Output image:", args.out_img)
    print("Image size:", width, "x", height)
    print("Number of pixels used:", expected_pixels)

    # Show if requested
    if args.show:
        cv2.imshow("Reconstructed Image", img_main)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Metrics (only if user provided clean + (noisy/restored))
    # We do not crash if something is missing; we just skip metrics.
    if args.clean is None:
        return

    # Load original
    clean_pixels = read_pixels_from_file(args.clean)
    clean_pixels = fix_length(clean_pixels, expected_pixels)
    img_clean = pixels_to_img(clean_pixels, height, width)

    # Compute original vs noisy
    if args.noisy is not None:
        noisy_pixels = read_pixels_from_file(args.noisy)
        noisy_pixels = fix_length(noisy_pixels, expected_pixels)
        img_noisy = pixels_to_img(noisy_pixels, height, width)

        snr_noisy = calc_snr_db_formula(img_clean, img_noisy)
        psnr_noisy = calc_psnr_db_formula(img_clean, img_noisy)
    else:
        snr_noisy = None
        psnr_noisy = None

    # Compute original vs restored
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

    # Print exactly in the format you asked for (only when values exist)
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
