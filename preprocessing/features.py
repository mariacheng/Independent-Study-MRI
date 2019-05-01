import numpy as np
from skimage.filters import gabor
from scipy import ndimage as simg
from scipy import stats
from scipy.signal import convolve2d
from mahotas.features import haralick, lbp
from mahotas.features.lbp import lbp_transform

# Feature extraction returns one value
def feat_extraction_1d(array):
    return np.array([np.mean(array), np.std(array), stats.skew(array.flatten()), stats.kurtosis(array.flatten())])

def gabor_features(image, scales=(2, 4, 8), orientations=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    features = []
    for theta in orientations:
        for scale in scales:
            features.append(np.abs(gabor(image, scale, theta)[0]).ravel())

    return np.array(features)

def kirsch_features(image):
    kirsch_filter = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    ind = np.array([0, 1, 2, 5, 8, 7, 6, 3]) # circular indices
    features = []
    for i in range(8):
        features.append(convolve2d(image, kirsch_filter, mode='same', boundary='symm').ravel())
        kirsch_filter.flat[ind] = np.roll(kirsch_filter.flat[ind], 1)

    return np.array(features)

def haralick_GLCM_features(image):
    image = image.astype(int)
    features = haralick(image, return_mean=True)

    return features

def local_binary_pattern_features(image):
    features = lbp(image, 3, 6)

    return features


# Feature extraction on local area
def conv_local_binary_pattern_features(image):
    features = lbp_transform(image, 3, 6, preserve_shape=True).ravel()

    return np.array(features)

def conv_feat_extraction_1d(image):
    func = [np.mean, np.std, stats.skew, stats.kurtosis]
    features = []

    for f in func:
        features.append(simg.generic_filter(image, f, size=(3,3)).ravel())

    return np.array(features)
