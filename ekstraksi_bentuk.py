# 1. Import library
import os
import numpy as np
from skimage import measure, filters
from skimage.color import rgb2gray
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

# 2. Fungsi ekstraksi fitur bentuk
def extract_shape_features(roi):
    if len(roi.shape) == 3:
        gray = rgb2gray(roi)
    else:
        gray = roi

    if gray.shape[0] == 0 or gray.shape[1] == 0:
        return {k: 0 for k in ['area', 'perimeter', 'eccentricity', 'solidity',
                               'extent', 'major_axis', 'minor_axis', 'orientation',
                               'compactness', 'aspect_ratio',
                               'hu_moment_0', 'hu_moment_1', 'hu_moment_2',
                               'hu_moment_3', 'hu_moment_4']}

    try:
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        binary = clear_border(binary)
        binary = closing(binary, square(3))
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if not regions:
            binary = np.ones_like(gray, dtype=bool)
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            if not regions:
                raise ValueError("Tidak ada region ditemukan bahkan setelah fallback.")
        largest_region = max(regions, key=lambda x: x.area)
    except Exception:
        binary = np.ones_like(gray, dtype=bool)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        if not regions:
            return {k: 0 for k in ['area', 'perimeter', 'eccentricity', 'solidity',
                                   'extent', 'major_axis', 'minor_axis', 'orientation',
                                   'compactness', 'aspect_ratio',
                                   'hu_moment_0', 'hu_moment_1', 'hu_moment_2',
                                   'hu_moment_3', 'hu_moment_4']}
        largest_region = max(regions, key=lambda x: x.area)

    shape_features = {
        'area': largest_region.area,
        'perimeter': largest_region.perimeter if largest_region.perimeter > 0 else 0,
        'eccentricity': largest_region.eccentricity if largest_region.eccentricity is not None else 0,
        'solidity': largest_region.solidity if largest_region.solidity is not None else 0,
        'extent': largest_region.extent if largest_region.extent is not None else 0,
        'major_axis': largest_region.major_axis_length if largest_region.major_axis_length is not None else 0,
        'minor_axis': largest_region.minor_axis_length if largest_region.minor_axis_length is not None else 0,
        'orientation': largest_region.orientation if largest_region.orientation is not None else 0,
    }

    perimeter_sq = largest_region.perimeter ** 2 if largest_region.perimeter > 0 else 1e-7
    minor_axis_len = largest_region.minor_axis_length if largest_region.minor_axis_length and largest_region.minor_axis_length > 0 else 1e-7
    major_axis_len = largest_region.major_axis_length if largest_region.major_axis_length and largest_region.major_axis_length > 0 else 1e-7

    shape_features['compactness'] = (4 * np.pi * largest_region.area) / (perimeter_sq + 1e-7)
    shape_features['aspect_ratio'] = major_axis_len / minor_axis_len

    try:
        moments = measure.moments_hu(measure.moments(binary.astype(np.uint8)))
        for i, moment in enumerate(moments[:5]):
            shape_features[f'hu_moment_{i}'] = moment if not np.isnan(moment) and not np.isinf(moment) else 0
    except Exception:
        for i in range(5):
            shape_features[f'hu_moment_{i}'] = 0

    return shape_features