import os
import cv2
import numpy as np
from skimage import feature
from skimage.color import rgb2gray
from scipy.stats import skew, kurtosis as scipy_kurtosis

# Fungsi ekstraksi fitur tekstur
def extract_texture_features(roi):
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return {k: 0 for k in [
            'glcm_contrast_mean', 'glcm_dissim_mean', 'glcm_homogen_mean', 'glcm_energy_mean',
            'lbp_bin_0', 'lbp_bin_1', 'lbp_bin_2', 'lbp_bin_3', 'lbp_bin_4',
            'mean_intensity', 'std_intensity', 'variance', 'skewness', 'kurtosis'
        ]}

    # Cek format dimensi gambar
    if len(roi.shape) == 3:
        gray = rgb2gray(roi)
    else:
        gray = roi

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        gray_uint8 = (np.zeros((max(2, gray.shape[0]), max(2, gray.shape[1]))) * 255).astype(np.uint8)
    else:
        gray_uint8 = (gray * 255).astype(np.uint8)

    texture_features = {}

    try:
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm_props_collected = {'contrast': [], 'dissimilarity': [], 'homogeneity': [], 'energy': []}

        if gray_uint8.ndim == 2 and gray_uint8.shape[0] > 1 and gray_uint8.shape[1] > 1:
            glcm = feature.graycomatrix(gray_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            glcm_props_collected['contrast'] = feature.graycoprops(glcm, 'contrast').ravel()
            glcm_props_collected['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity').ravel()
            glcm_props_collected['homogeneity'] = feature.graycoprops(glcm, 'homogeneity').ravel()
            glcm_props_collected['energy'] = feature.graycoprops(glcm, 'energy').ravel()

            texture_features.update({
                'glcm_contrast_mean': np.mean(glcm_props_collected['contrast']),
                'glcm_dissim_mean': np.mean(glcm_props_collected['dissimilarity']),
                'glcm_homogen_mean': np.mean(glcm_props_collected['homogeneity']),
                'glcm_energy_mean': np.mean(glcm_props_collected['energy'])
            })
        else:
            texture_features.update({k: 0 for k in ['glcm_contrast_mean', 'glcm_dissim_mean', 'glcm_homogen_mean', 'glcm_energy_mean']})
    except Exception:
        texture_features.update({k: 0 for k in ['glcm_contrast_mean', 'glcm_dissim_mean', 'glcm_homogen_mean', 'glcm_energy_mean']})

    try:
        P, R = 8, 1
        if gray_uint8.shape[0] > 2 * R and gray_uint8.shape[1] > 2 * R:
            lbp = feature.local_binary_pattern(gray_uint8, P=P, R=R, method='uniform')
            n_bins = int(lbp.max() + 1) if lbp.size > 0 else 10
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-7)

            for i in range(min(5, len(lbp_hist))):
                texture_features[f'lbp_bin_{i}'] = lbp_hist[i]
            for i in range(len(lbp_hist), 5):
                texture_features[f'lbp_bin_{i}'] = 0
        else:
            for i in range(5):
                texture_features[f'lbp_bin_{i}'] = 0
    except Exception:
        for i in range(5):
            texture_features[f'lbp_bin_{i}'] = 0

    try:
        flat_gray = gray_uint8.ravel()
        texture_features.update({
            'mean_intensity': np.mean(flat_gray),
            'std_intensity': np.std(flat_gray),
            'variance': np.var(flat_gray),
            'skewness': float(skew(flat_gray)) if flat_gray.size > 0 else 0,
            'kurtosis': float(scipy_kurtosis(flat_gray, fisher=False)) if flat_gray.size > 0 else 0
        })
    except Exception:
        texture_features.update({k: 0 for k in ['mean_intensity', 'std_intensity', 'variance', 'skewness', 'kurtosis']})

    return texture_features