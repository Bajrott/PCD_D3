import os
import numpy as np
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans

# Fungsi ekstraksi warna
def extract_color_features(roi):
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return {k: 0 for k in ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
                               'mean_h', 'mean_s', 'mean_v',
                               'dominant_r_0', 'dominant_g_0', 'dominant_b_0',
                               'dominant_r_1', 'dominant_g_1', 'dominant_b_1',
                               'dominant_r_2', 'dominant_g_2', 'dominant_b_2',
                               'hist_r_0', 'hist_r_1', 'hist_r_2', 'hist_r_3', 'hist_r_4',
                               'hist_g_0', 'hist_g_1', 'hist_g_2', 'hist_g_3', 'hist_g_4',
                               'hist_b_0', 'hist_b_1', 'hist_b_2', 'hist_b_3', 'hist_b_4']}

    # untuk memastikan format float untuk hsv conversion skimage
    if roi.dtype == np.uint8:
        roi_float = roi.astype(np.float32) / 255.0
    else:
        roi_float = roi

    hsv = rgb2hsv(roi_float)
    
    color_features = {
        'mean_r': float(np.mean(roi[:, :, 0])), 'mean_g': float(np.mean(roi[:, :, 1])), 'mean_b': float(np.mean(roi[:, :, 2])),
        'std_r': float(np.std(roi[:, :, 0])), 'std_g': float(np.std(roi[:, :, 1])), 'std_b': float(np.std(roi[:, :, 2])),
        'mean_h': float(np.mean(hsv[:, :, 0])), 'mean_s': float(np.mean(hsv[:, :, 1])), 'mean_v': float(np.mean(hsv[:, :, 2]))
    }

    pixels = roi.reshape(-1, 3)
    if len(pixels) >= 3:
        try:
            n_clusters_actual = min(3, len(np.unique(pixels, axis=0)))
            if n_clusters_actual > 0:
                kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = kmeans.cluster_centers_
                for i in range(3):
                    if i < len(dominant_colors):
                        color_features[f'dominant_r_{i}'] = float(dominant_colors[i][0])
                        color_features[f'dominant_g_{i}'] = float(dominant_colors[i][1])
                        color_features[f'dominant_b_{i}'] = float(dominant_colors[i][2])
                    else:
                        color_features[f'dominant_r_{i}'] = 0.0
                        color_features[f'dominant_g_{i}'] = 0.0
                        color_features[f'dominant_b_{i}'] = 0.0
        except:
            for i in range(3):
                color_features[f'dominant_r_{i}'] = 0.0
                color_features[f'dominant_g_{i}'] = 0.0
                color_features[f'dominant_b_{i}'] = 0.0
    else:
        for i in range(3):
            color_features[f'dominant_r_{i}'] = 0.0
            color_features[f'dominant_g_{i}'] = 0.0
            color_features[f'dominant_b_{i}'] = 0.0

    hist_r = np.histogram(roi[:, :, 0], bins=16, range=[0, 256])[0]
    hist_g = np.histogram(roi[:, :, 1], bins=16, range=[0, 256])[0]
    hist_b = np.histogram(roi[:, :, 2], bins=16, range=[0, 256])[0]
    hist_r = hist_r / (np.sum(hist_r) + 1e-7)
    hist_g = hist_g / (np.sum(hist_g) + 1e-7)
    hist_b = hist_b / (np.sum(hist_b) + 1e-7)

    for i in range(5):
        color_features[f'hist_r_{i}'] = float(hist_r[i])
        color_features[f'hist_g_{i}'] = float(hist_g[i])
        color_features[f'hist_b_{i}'] = float(hist_b[i])

    return color_features