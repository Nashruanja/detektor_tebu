"""
Feature Extraction Module untuk Klasifikasi Penyakit Daun Tebu
Menggunakan pipeline yang sama dengan training notebook
"""

import numpy as np
import cv2


def extract_features(img):
    """
    Ekstraksi fitur dari gambar daun tebu
    
    Args:
        img: numpy array gambar (BGR format dari cv2)
    
    Returns:
        numpy array dengan ~1003 fitur
    """
    fv = []
    
    # Resize ke 256x256 (sama dengan training)
    img = cv2.resize(img, (256, 256))
    
    # ========== Color Spaces ==========
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ========== Color Histograms (384 features) ==========
    for color_img in [img, hsv, lab, ycrcb]:
        for ch in range(3):
            hist = cv2.calcHist([color_img], [ch], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            fv.extend(hist)
    
    # ========== Statistical Moments (72 features) ==========
    for color_img in [img, hsv, lab, ycrcb]:
        for ch in range(3):
            ch_data = color_img[:, :, ch].flatten()
            fv.append(np.mean(ch_data))
            fv.append(np.std(ch_data))
            fv.append(np.median(ch_data))
            fv.append(np.var(ch_data))
            mean = np.mean(ch_data)
            std = np.std(ch_data) + 1e-7
            fv.append(np.mean(((ch_data - mean) / std) ** 3))  # Skewness
            fv.append(np.mean(((ch_data - mean) / std) ** 4))  # Kurtosis
    
    # ========== GLCM Texture (20 features) ==========
    gray_q = (gray // 32).astype(np.uint8)
    for dx, dy in [(1,0), (1,1), (0,1), (-1,1)]:
        glcm = np.zeros((8, 8))
        for i in range(1, gray_q.shape[0]-1):
            for j in range(1, gray_q.shape[1]-1):
                if 0 <= i+dy < gray_q.shape[0] and 0 <= j+dx < gray_q.shape[1]:
                    glcm[gray_q[i,j], gray_q[i+dy,j+dx]] += 1
        
        glcm = glcm / (glcm.sum() + 1e-7)
        contrast = sum([(i-j)**2 * glcm[i,j] for i in range(8) for j in range(8)])
        homogeneity = sum([glcm[i,j]/(1+abs(i-j)) for i in range(8) for j in range(8)])
        energy = sum([glcm[i,j]**2 for i in range(8) for j in range(8)])
        correlation = sum([i*j*glcm[i,j] for i in range(8) for j in range(8)])
        entropy = -sum([glcm[i,j]*np.log(glcm[i,j]+1e-7) for i in range(8) for j in range(8)])
        fv.extend([contrast, homogeneity, energy, correlation, entropy])
    
    # ========== LBP Texture (32 features) ==========
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i,j]
            code = 0
            code |= (gray[i-1,j-1] >= center) << 7
            code |= (gray[i-1,j] >= center) << 6
            code |= (gray[i-1,j+1] >= center) << 5
            code |= (gray[i,j+1] >= center) << 4
            code |= (gray[i+1,j+1] >= center) << 3
            code |= (gray[i+1,j] >= center) << 2
            code |= (gray[i+1,j-1] >= center) << 1
            code |= (gray[i,j-1] >= center)
            lbp[i,j] = code
    
    lbp_hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
    lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-7)
    fv.extend(lbp_hist)
    
    # ========== Gabor Wavelets (48 features) ==========
    for theta in np.arange(0, np.pi, np.pi/8):
        for sigma in [3, 5, 7]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            fv.append(np.mean(filtered))
            fv.append(np.std(filtered))
    
    # ========== HOG Features ==========
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    gray_resized = cv2.resize(gray, winSize)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(gray_resized)
    hog_features = hog_features.flatten()[::4]
    fv.extend(hog_features)
    
    # ========== Edge Features (6 features) ==========
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    fv.extend([np.mean(sobel), np.std(sobel), np.max(sobel), 
               np.percentile(sobel, 75), np.percentile(sobel, 90)])
    canny = cv2.Canny(gray, 50, 150)
    fv.append(np.sum(canny > 0) / canny.size)
    
    return np.array(fv)


def create_readable_feature_names():
    """Membuat nama fitur yang mudah dipahami"""
    feature_names = []
    
    # 1. Color Histograms (384)
    color_types = [
        ('RGB', ['Merah', 'Hijau', 'Biru']),
        ('HSV', ['Hue', 'Saturasi', 'Value']),
        ('LAB', ['Lightness', 'A-Channel', 'B-Channel']),
        ('YCrCb', ['Luma', 'Chroma-Red', 'Chroma-Blue'])
    ]
    
    for space, channels in color_types:
        for channel in channels:
            for i in range(32):
                intensity = ['Gelap', 'Sedang-Gelap', 'Sedang-Terang', 'Terang'][i//8]
                feature_names.append(f"Warna: {channel} {intensity} ({space})")
    
    # 2. Statistical (72)
    stats_labels = ['Mean', 'Std', 'Median', 'Var', 'Skew', 'Kurt']
    color_spaces = ['RGB-R', 'RGB-G', 'RGB-B', 'HSV-H', 'HSV-S', 'HSV-V',
                    'LAB-L', 'LAB-A', 'LAB-B', 'YCrCb-Y', 'YCrCb-Cr', 'YCrCb-Cb']
    for space in color_spaces:
        for stat in stats_labels:
            feature_names.append(f"Statistik: {space} {stat}")
    
    # 3. GLCM (20)
    textures = ['Kontras', 'Homogenitas', 'Energi', 'Korelasi', 'Entropi']
    directions = ['Horizontal', 'Diagonal /', 'Vertikal', 'Diagonal \\']
    for direction in directions:
        for texture in textures:
            feature_names.append(f"Tekstur: {texture} {direction}")
    
    # 4. LBP (32)
    for i in range(32):
        feature_names.append(f"LBP: Bin {i}")
    
    # 5. Gabor (48)
    orientations = ['0°', '22.5°', '45°', '67.5°', '90°', '112.5°', '135°', '157.5°']
    for ori in orientations:
        for sigma in [3, 5, 7]:
            feature_names.append(f"Gabor: {ori} σ={sigma} Mean")
            feature_names.append(f"Gabor: {ori} σ={sigma} Std")
    
    # 6. HOG (variable)
    # Estimasi berdasarkan HOG descriptor
    for i in range(225):  # Sesuaikan dengan jumlah fitur HOG sebenarnya
        feature_names.append(f"HOG: Feature {i+1}")
    
    # 7. Edge (6)
    feature_names.extend([
        'Edge: Sobel Mean',
        'Edge: Sobel Std',
        'Edge: Sobel Max',
        'Edge: Sobel P75',
        'Edge: Sobel P90',
        'Edge: Canny Density'
    ])
    
    return feature_names