import argparse
from PIL import Image
import numpy as np

# Convert image to HEX and create a noisy image with Gaussian noise using a target SNR

def load_image(image_path, width, height):
    # Open the image from disk (can be RGB, PNG, JPG, etc.)
    img = Image.open(image_path)

    # Resize image to required size (for example 128x128 or 512x512)
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert image to grayscale (8-bit: values from 0 to 255)
    # If the image is already grayscale, this will not change the actual pixel values.
    img = img.convert("L")

    # Convert image to numpy array (uint8)
    img_array = np.array(img, dtype=np.uint8)

    return img_array


def save_image_to_hex(img_array, filename):
    # Flatten the image row by row (top-left to bottom-right)
    pixels = img_array.flatten()

    # Write each pixel as a 2-digit hex number (00..ff), one pixel per line
    with open(filename, "w") as f:
        for p in pixels:
            f.write(f"{int(p):02x}\n")


def compute_snr_db(clean_image, other_image):

    clean = clean_image.astype(float)
    other = other_image.astype(float)

    # Numerator: signal energy
    signal_energy = np.sum(clean ** 2)

    # Denominator: error (noise) energy
    noise_energy = np.sum((clean - other) ** 2)

    # Protect against divide-by-zero (in case images are identical)
    if noise_energy == 0:
        return float("inf")

    snr_linear = signal_energy / noise_energy
    snr_db = 10 * np.log10(snr_linear)

    return snr_db


def add_gaussian_noise(clean_image, target_snr_db):
    # Convert image to float for processing
    clean_float = clean_image.astype(float)

    # Convert SNR from dB to linear
    snr_linear = 10 ** (target_snr_db / 10)

    # Compute total signal energy: sum(f^2)
    signal_energy = np.sum(clean_float ** 2)

    # Required total noise energy: sum(noise^2)
    required_noise_energy = signal_energy / snr_linear

    # If the image has N pixels, and the noise is i.i.d Gaussian,
    # then expected sum(noise^2) = N * sigma^2
    num_pixels = clean_float.size

    # So sigma^2 = required_noise_energy / N
    noise_variance = required_noise_energy / num_pixels

    # We prefer to PRINT sigma, not variance
    noise_sigma = np.sqrt(noise_variance)

    # Generate Gaussian noise with mean=0 and std=sigma
    noise = np.random.normal(0, noise_sigma, clean_float.shape)

    # Add noise to create noisy image
    noisy_float = clean_float + noise

    # Clip to valid 8-bit range
    noisy_float = np.clip(noisy_float, 0, 255)

    # Convert back to uint8 for saving to hex
    noisy_image = noisy_float.astype(np.uint8)

    return noisy_image, noise_sigma


def main():
    # Create argument parser
    parser = argparse.ArgumentParser( description="Convert image to HEX and create a noisy image with Gaussian noise using a target SNR")
    # Input image path
    parser.add_argument("image", help="Path to input image")
    # Works for 128, 512, or any other value
    parser.add_argument("--size", type=int, default=128, help="Image size (example: 128 or 512). Default = 128")
    # Target SNR in dB
    parser.add_argument( "--snr", type=float, default=20.0, help="Target SNR in dB (example: 10, 20, 30). Default = 20")

    args = parser.parse_args()

    # Load and prepare the clean image (resize + grayscale)
    clean_image = load_image(args.image, args.size, args.size)

    # Save clean image to HEX
    save_image_to_hex(clean_image, "clean_image.hex")
    Image.fromarray(clean_image).save("clean_image.png")

    # Add Gaussian noise with requested target SNR (dB)
    noisy_image, noise_sigma = add_gaussian_noise(clean_image, args.snr)

    # Save noisy image to HEX
    save_image_to_hex(noisy_image, "noisy_image.hex")
    Image.fromarray(noisy_image).save("noisy_image.png")

    # Measure SNR using YOUR formula (energy ratio) and print it
    measured_snr_db = compute_snr_db(clean_image, noisy_image)

    # Print useful information
    print("Image size:", args.size, "x", args.size)
    print("Target SNR (dB):", args.snr)
    print(f"Noise sigma (standard deviation): {noise_sigma:.3f}")
    print(f"Measured SNR (dB) : {measured_snr_db:.3f} dB")
    print("Created files:")
    print("  clean_image.hex")
    print("  noisy_image.hex")


if __name__ == "__main__":
    main()
