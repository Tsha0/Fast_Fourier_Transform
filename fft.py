
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import cv2
import time


def naive_dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of 1D array x using the naive method.
    """
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def naive_idft(X):
    """
    Compute the inverse DFT of 1D array X using the naive method.
    """
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, X) / N

def fft(x):
    """
    Compute the FFT of 1D array x using the Cooley-Tukey algorithm.
    Assumes the length of x is a power of 2.
    """
    N = x.shape[0]
    if N == 1:
        return x
    if N % 2:
        return naive_dft(x)
    even = fft(x[::2])
    odd  = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + factor[:N//2] * odd,
                           even - factor[:N//2] * odd])

def ifft(X):
    """
    Compute the inverse FFT of 1D array X.
    """
    return np.conjugate(fft(np.conjugate(X))) / X.shape[0]


def fft2d(img):
    """
    Compute the 2D FFT of a 2D array (image) by applying the 1D FFT
    to each row then each column.
    """
    M, N = img.shape
    F = np.zeros((M, N), dtype=complex)
    for m in range(M):
        F[m, :] = fft(img[m, :])
    F2 = np.zeros((M, N), dtype=complex)
    for n in range(N):
        F2[:, n] = fft(F[:, n])
    return F2

def ifft2d(F):
    """
    Compute the inverse 2D FFT of a 2D array by applying the 1D IFFT
    to each column then each row.
    """
    M, N = F.shape
    f = np.zeros((M, N), dtype=complex)
    for n in range(N):
        f[:, n] = ifft(F[:, n])
    f2 = np.zeros((M, N), dtype=complex)
    for m in range(M):
        f2[m, :] = ifft(f[m, :])
    return f2


def pad_to_power_of_two(img):
    """
    Pad a 2D array with zeros so that both dimensions are the next power of 2.
    """
    M, N = img.shape
    new_M = 2 ** int(np.ceil(np.log2(M)))
    new_N = 2 ** int(np.ceil(np.log2(N)))
    padded = np.zeros((new_M, new_N), dtype=img.dtype)
    padded[:M, :N] = img
    return padded

def denoise_image(img, cutoff_ratio=0.1):
    """
    Denoise an image by computing its FFT, zeroing out high-frequency coefficients,
    and then computing the inverse FFT.
    
    The cutoff_ratio determines the size of the block (from the top-left, i.e. low frequencies)
    that is kept.
    
    Returns the denoised image (absolute value), the number of nonzero coefficients, and total coefficient count.
    """
    F = fft2d(img)
    M, N = F.shape
    cutoff_M = int(cutoff_ratio * M)
    cutoff_N = int(cutoff_ratio * N)
    mask = np.zeros_like(F, dtype=bool)
    mask[:cutoff_M, :cutoff_N] = True
    F_denoised = np.where(mask, F, 0)
    denoised = ifft2d(F_denoised)
    return np.abs(denoised), np.count_nonzero(F_denoised), F_denoised.size

def compress_image(img, levels):
    """
    Compress an image by thresholding Fourier coefficients.
    For each compression level (percentage of coefficients set to zero),
    compute the inverse FFT to reconstruct the image.
    
    Returns a list of compressed images and a list of nonzero coefficient counts.
    """
    F = fft2d(img)
    total_coeff = F.size
    # Flatten the magnitudes and get sorted indices (smallest to largest)
    F_flat = np.abs(F).flatten()
    sorted_indices = np.argsort(F_flat)
    compressed_images = []
    nonzero_counts = []
    for level in levels:
        # Determine number of coefficients to zero out
        num_to_zero = int(total_coeff * (level / 100.0))
        F_comp = np.copy(F)
        if num_to_zero > 0:
            F_comp.flat[sorted_indices[:num_to_zero]] = 0
        nonzero_counts.append(np.count_nonzero(F_comp))
        comp_img = ifft2d(F_comp)
        compressed_images.append(np.abs(comp_img))
    return compressed_images, nonzero_counts

# ----------------------------
# Modes: Fast Display, Denoise, Compress, and Runtime Plotting
# ----------------------------

def run_mode1(image_path):
    """
    Mode 1: Fast mode – Convert the image to FFT form and display
    the original image alongside its Fourier transform (log-scaled).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return
    img_padded = pad_to_power_of_two(img)
    F = fft2d(img_padded)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_padded, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    # Using logarithm of the magnitude for better visibility
    plt.imshow(np.log(np.abs(F) + 1), cmap='gray', norm=LogNorm())
    plt.title("FFT (log-scaled)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_mode2(image_path):
    """
    Mode 2: Denoise – Apply FFT, zero high frequencies, and display
    the original image alongside its denoised version.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return
    img_padded = pad_to_power_of_two(img)
    denoised, nonzeros, total = denoise_image(img_padded, cutoff_ratio=0.1)
    print("Denoising: Keeping", nonzeros, "out of", total, "coefficients (",
          f"{nonzeros/total*100:.2f}%)")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_padded, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_mode3(image_path):
    """
    Mode 3: Compress – Compress the image at various levels by zeroing
    Fourier coefficients and display the 6 reconstructed images.
    Also print the number of nonzero coefficients for each level.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image.")
        return
    img_padded = pad_to_power_of_two(img)
    levels = [0, 50, 70, 90, 95, 99.9]
    compressed_images, nonzero_counts = compress_image(img_padded, levels)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(compressed_images[i], cmap='gray')
        ax.set_title(f"{levels[i]}% comp. | Nonzeros: {nonzero_counts[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def run_mode4():
    """
    Mode 4: Plot runtime graphs for the naive DFT and FFT algorithms.
    For various square array sizes (powers of 2), run each method 10 times,
    compute mean runtimes and standard deviations, and plot with error bars.
    Note: Since the naive 2D DFT is very slow for large sizes, we restrict
    sizes to small arrays.
    """
    sizes = [8, 16, 32, 64, 128, 256, 512]
    num_runs = 10
    naive_times = []
    fft_times = []
    size_list = []

    def naive_2d(arr):
        M, N = arr.shape
        F = np.zeros((M, N), dtype=complex)
        for m in range(M):
            F[m, :] = naive_dft(arr[m, :])
        F2 = np.zeros((M, N), dtype=complex)
        for n in range(N):
            F2[:, n] = naive_dft(F[:, n])
        return F2

    for n in sizes:
        arr = np.random.rand(n, n)
        t_naive = []
        t_fft = []
        for _ in range(num_runs):
            start = time.time()
            naive_2d(arr)
            t_naive.append(time.time() - start)
            start = time.time()
            fft2d(arr)
            t_fft.append(time.time() - start)
        mean_naive, std_naive = np.mean(t_naive), np.std(t_naive)
        mean_fft, std_fft = np.mean(t_fft), np.std(t_fft)
        naive_times.append((mean_naive, std_naive))
        fft_times.append((mean_fft, std_fft))
        size_list.append(n)
        print(f"Size {n}x{n}: Naive DFT = {mean_naive:.6f}s ± {std_naive:.6f}s, FFT = {mean_fft:.6f}s ± {std_fft:.6f}s")

    plt.errorbar(size_list, [t[0] for t in naive_times], yerr=[t[1] for t in naive_times],
                 fmt='o-', label='Naive DFT')
    plt.errorbar(size_list, [t[0] for t in fft_times], yerr=[t[1] for t in fft_times],
                 fmt='o-', label='FFT')
    plt.xlabel("Array Size (n x n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: Naive DFT vs. FFT")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="FFT Assignment Implementation")
    parser.add_argument("-m", "--mode", type=int, default=1,
                        help="Mode: 1-Fast display, 2-Denoise, 3-Compress, 4-Runtime plots")
    parser.add_argument("-i", "--image", type=str, default="default_image.png",
                        help="Image file path (default: default_image.png)")
    args = parser.parse_args()

    if args.mode == 1:
        run_mode1(args.image)
    elif args.mode == 2:
        run_mode2(args.image)
    elif args.mode == 3:
        run_mode3(args.image)
    elif args.mode == 4:
        run_mode4()
    else:
        print("Invalid mode selected. Choose a mode from 1 to 4.")

if __name__ == "__main__":
    main()
