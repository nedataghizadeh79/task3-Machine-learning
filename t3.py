#  Principles of Machine Learning - Task 1.3
#  Estimating Fractal Dimension using Box Counting Method
#   implementation with image saving and visualization

import numpy as np
import imageio.v3 as iio
import scipy.ndimage as img
import matplotlib.pyplot as plt
import numpy.linalg as la
import os


# 1. Binarization Function (from page 15)

def binarize(imgF):
    """
    Apply Gaussian filters and threshold to create binary image.
    Automatically inverts if needed so that foreground is white (1).
    """
    imgB = np.abs(img.gaussian_filter(imgF, sigma=0.50) -
                  img.gaussian_filter(imgF, sigma=1.00))
    bin_img = np.where(imgB < 0.1 * imgB.max(), 0, 1)
    bin_img = img.binary_closing(bin_img)

    #  Invert automatically if background is dark (like lightning)
    if np.mean(imgF) < 128:   # meaning the image is dark overall
        bin_img = 1 - bin_img
    return bin_img




# 2. box Counting Function

def box_count(imgB):
    """
    Compute box counts for multiple scales.
    Returns arrays of log(1/s) and log(n) for regression.
    """
    assert imgB.shape[0] == imgB.shape[1], "Image must be square"
    w = imgB.shape[0]
    L = int(np.log2(w))
    s_values = [1 / (2 ** l) for l in range(1, L - 1)]

    n_list = []
    for s in s_values:
        box_size = int(w * s)
        count = 0
        # Slide over the image with non-overlapping boxes
        for i in range(0, w, box_size):
            for j in range(0, w, box_size):
                if np.any(imgB[i:i + box_size, j:j + box_size]):
                    count += 1
        n_list.append(count)

    #  logs for regression
    log_inv_s = np.log(1 / np.array(s_values))
    log_n = np.log(np.array(n_list))
    return log_inv_s, log_n, s_values, n_list



# 3. Linear Fit (Least Squares)

def estimate_fractal_dimension(log_inv_s, log_n):
    """Estimate slope (D) using least squares linear regression."""
    A = np.vstack([log_inv_s, np.ones(len(log_inv_s))]).T
    D, b = la.lstsq(A, log_n, rcond=None)[0]
    return D, b


# 4. Main Process Function

def process_image(image_path, output_prefix):
    """
    Full pipeline: read -> binarize -> save -> box count -> plot
    """
    # Step 1: Read and binarize
    img_gray = iio.imread(image_path, mode='L').astype(float)
    img_bin = binarize(img_gray)

    # Step 2: Save binary image
    output_bin_path = f"{output_prefix}_binary.png"
    iio.imwrite(output_bin_path, (img_bin * 255).astype(np.uint8))
    print(f"[+] Saved binarized image: {output_bin_path}")

    # Step 3: Compute box counts
    log_inv_s, log_n, s_values, n_list = box_count(img_bin)

    # Step 4: Fit line to log-log data
    D, b = estimate_fractal_dimension(log_inv_s, log_n)

    # Step 5: Plot
    plt.figure(figsize=(6, 5))
    plt.plot(log_inv_s, log_n, 'o', label='Data')
    plt.plot(log_inv_s, D * log_inv_s + b, '-', label=f'Fit: D={D:.3f}')
    plt.xlabel('log(1/s)')
    plt.ylabel('log(n)')
    plt.title(f'Fractal Dimension Estimation for {output_prefix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plot.png")
    plt.show()

    # Step 6: Return result
    return D


# 5. Run for both images

if __name__ == "__main__":
    tree_D = process_image("tree.png", "tree")
    lightning_D = process_image("lightning.png", "lightning")

    print("\n================== RESULTS ==================")
    print(f"Fractal Dimension (Tree): {tree_D:.4f}")
    print(f"Fractal Dimension (Lightning): {lightning_D:.4f}")
    if tree_D > lightning_D:
        print("→ Tree has the higher fractal dimension.")
    else:
        print("→ Lightning has the higher fractal dimension.")
    print("=============================================")
