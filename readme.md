Below is a **clean, professional `README.md` you can copy directly into your repository**.
It explains the **full pipeline, files, dependencies, and how to run the system**. It also reflects the **actual structure of your project and scripts**.

---

# Hardware Wiener Image Denoising Accelerator

This project implements a **streaming hardware accelerator for Wiener image denoising** using **SystemVerilog** with a **Python-based preprocessing and evaluation pipeline**.

The system performs:

1. Image preprocessing and noise injection
2. Noise variance estimation
3. Hardware simulation of a Wiener filter
4. Image reconstruction from hardware output
5. Quality evaluation (SNR, PSNR, SSIM)

The pipeline integrates **Python image processing** with **RTL simulation using Icarus Verilog**.

---

# Project Overview

The full processing flow:


Input Image
     │
     ▼
img_to_hex.py
(Convert image → grayscale → resize → add Gaussian noise)
     │
     ├── clean_image.hex
     ├── noisy_image.hex
     ├── clean_image.png
     └── noisy_image.png
     │
     ▼
Wiener_filter.py
(estimate global noise variance)
     │
     ▼
RTL Simulation (Icarus Verilog)
     │
     ├── output_pixels_5x5.txt
     └── output_pixels_3x3.txt
     │
     ▼
reconstruct_image.py
(Rebuild images + compute metrics)
     │
     ├── restored_5x5.png
     └── restored_3x3.png


# Repository Structure 
```
'frontEnd/'
│
├── run.py                     # Main automation script
├── img_to_hex.py              # Image preprocessing + noise generation
├── Wiener_filter.py           # Noise variance estimation
├── reconstruct_image.py       # Rebuild images from hardware output
├── reconstruct_image_rgb.py   # RGB reconstruction variant
│
├── rtl.f                      # RTL file list for compilation
│
├── tb_wiener_top.sv           # Testbench
├── wiener_top.sv              # Top hardware module
├── wiener_core.sv             # Wiener filter core
├── stats_calc_5x5.sv          # Local statistics module
├── window5x5.sv               # Sliding window generator
├── vert_mux5.sv               # Vertical multiplexer
├── col_reflect_block.sv       # Edge reflection logic
├── delay_line.sv              # Line buffer implementation
│
├── clean_image.hex
├── noisy_image.hex
│
├── output_pixels_5x5.txt
├── output_pixels_3x3.txt
│
├── clean_image.png
├── noisy_image.png
│
└── sim.out
```

---

# Requirements

## Python

Python 3.8+

Install dependencies:

```
pip install numpy pillow opencv-python scikit-image
```

Required libraries:

* numpy
* pillow
* opencv-python
* scikit-image

---

## RTL Simulation

Install **Icarus Verilog**.

Check installation:

```
iverilog -V
```

The project uses:

```
SystemVerilog 2012
```

---

# Quick Start (Recommended)

Run the entire pipeline with a single command:

```
python run.py --img ../../images/test1.jpg --size 512 --snr 14
```

This command performs:

1. Image preprocessing
2. Noise generation
3. Noise variance estimation
4. RTL compilation
5. Hardware simulation
6. Image reconstruction
7. Quality metric evaluation

---

# Step-by-Step Pipeline

## Step 1 — Convert Image to HEX

```
python img_to_hex.py input_image.jpg --size 512 --snr 20
```

Outputs:

```
clean_image.hex
noisy_image.hex
clean_image.png
noisy_image.png
```

The script:

* converts the image to grayscale
* resizes it to the specified size
* injects Gaussian noise based on the desired SNR
* exports pixel values for hardware simulation

---

## Step 2 — Estimate Noise Variance

The noise variance is estimated using:

```
estimate_noise_var_immerkaer()
```

This value is used to configure the Wiener filter hardware.

Example output:

```
Estimated noise variance = 873.73
```

---

## Step 3 — Compile Hardware

RTL modules are compiled using:

```
iverilog -g 2012 -o sim.out -f rtl.f
```

The `rtl.f` file lists all required SystemVerilog modules.

---

## Step 4 — Run Simulation

```
vvp sim.out +NOISE_VAR=<variance>
```

Example:

```
vvp sim.out +NOISE_VAR=873.7
```

The testbench:

* loads noisy image pixels
* streams them into the Wiener filter
* produces filtered outputs

Two hardware modes are simulated:

* **5×5 Wiener filter**
* **3×3 statistics mode**

Outputs:

```
output_pixels_5x5.txt
output_pixels_3x3.txt
```

---

## Step 5 — Reconstruct Output Images

```
python reconstruct_image.py \
--file output_pixels_5x5.txt \
--out_img restored_5x5.png \
--size 512
```

And:

```
python reconstruct_image.py \
--file output_pixels_3x3.txt \
--out_img restored_3x3.png \
--size 512
```

The script converts the pixel stream back into an image.

---

# Image Quality Metrics

The reconstruction script can compute:

### SNR

```
SNR = 10 log10( Σ f(x,y)^2 / Σ (f(x,y) − f̂(x,y))^2 )
```

### PSNR

```
PSNR = 10 log10( MAX_I^2 / Σ (f − f̂)^2 )
```

### SSIM

Computed using `skimage.metrics`.

Example output:

```
SNR (original vs noisy)    = 14.13 dB
SNR (original vs restored) = 24.72 dB
PSNR (original vs restored)= 28.10 dB
SSIM (original vs restored)= 0.91
```

---

# Image Size

The system currently assumes:

```
512 × 512 images
```

The size must match the RTL parameters:

```
IMG_W = 512
IMG_H = 512
```

If you change image size, you must update:

```
tb_wiener_top.sv
```

---

# Fixed-Point Format

Noise variance is provided to hardware using **Q16.8 format**.

Conversion used in the testbench:

```
noise_var_q16_8 = round(noise_variance * 256)
```

---

# Hardware Architecture

The Wiener accelerator pipeline includes:

```
Line Buffers
     │
Reflection Logic
     │
5×5 Window Generator
     │
Local Statistics Calculator
     │
Wiener Core
     │
Output Pixel Stream
```

The design is **fully streaming and pipelined**.

---

# Possible Improvements

Future extensions:

* dynamic image size configuration
* FPGA synthesis
* throughput benchmarking
* multi-frame processing
* color image support
* automated regression testing

---

# Author

Alen Faer
Technion – Israel Institute of Technology
Electrical & Computer Engineering

Project: Hardware Acceleration for Wiener Image Denoising

---

If you'd like, I can also generate a **much cleaner "GitHub-polished" README** (with diagrams of your pipeline and hardware architecture). That version makes the repo look **much stronger to recruiters or research labs**.
