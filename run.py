import subprocess
import argparse
from Wiener_filter import estimate_noise_var_immerkaer
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--img", required=True)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--snr", type=float, default=20)
args = parser.parse_args()

# Step 1: Convert image to hex
subprocess.run([
    "python", "img_to_hex.py",
    args.img,
    "--size", str(args.size),
    "--snr", str(args.snr)
], check=True)

# Step 2: Estimate noise variance from noisy image
noisy = np.array(Image.open("noisy_image.png").convert("L"))
noise_var = estimate_noise_var_immerkaer(noisy)

print(f"Estimated noise variance = {noise_var}")

# Step 3: Compile RTL
subprocess.run([
    "iverilog", "-g", "2012",
    "-o", "sim.out",
    "-f", "rtl.f"
], check=True)

# Step 4: Run simulation with plusarg
subprocess.run([
    "vvp", "sim.out",
    f"+NOISE_VAR={noise_var}"
], check=True)

# Step 5: Reconstruct output image
# Step 5: Reconstruct 5x5 output image
subprocess.run([
    "python", "reconstruct_image.py",
    "--file", "output_pixels_5x5.txt",
    "--out_img", "restored_5x5.png",
    "--size", str(args.size),
    "--clean", "clean_image.hex",
    "--noisy", "noisy_image.hex",
    "--restored", "output_pixels_5x5.txt"
], check=True)

# Step 6: Reconstruct 3x3 output image
subprocess.run([
    "python", "reconstruct_image.py",
    "--file", "output_pixels_3x3.txt",
    "--out_img", "restored_3x3.png",
    "--size", str(args.size),
    "--clean", "clean_image.hex",
    "--noisy", "noisy_image.hex",
    "--restored", "output_pixels_3x3.txt"
], check=True)