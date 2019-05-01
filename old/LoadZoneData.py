#!/usr/bin/env python
# coding: utf-8

import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as simg
from scipy import stats
from skimage.feature import greycomatrix, greycoprops

def save_obj(obj, name ):
    with open('./' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def trim_zeros_2D(array):

    # trim x
    for i in range(array.shape[0]):
        if np.sum(array[i,:]) > 0:
            break
    for j in reversed(range(array.shape[0])):
        if np.sum(array[j,:]) > 0:
            break
    array = array[i:j+1,:]
    
    # trim y
    for iy in range(array.shape[1]):
        if np.sum(array[:,iy]) > 0:
            break
    for jy in reversed(range(array.shape[1])):
        if np.sum(array[:,jy]) > 0:
            break
    array = array[:,iy:jy+1]
    
    indexes = [i,j,iy,jy]
    return array, indexes

def import_data(dir_path):
    data = []
    for file in os.listdir(dir_path):

        mat_file = scipy.io.loadmat(os.path.join(dir_path, file))

        patient_dict = {}

        patient_dict['id'] = mat_file['casesTableArr'][0][0][0][0][0][0][0]

        patient_dict['T2'] = mat_file['T2']

        patient_dict['ADC'] = mat_file['ADC']
        patient_dict['CDI'] = mat_file['CDI']
        patient_dict['HBV'] = mat_file['HBV']

        patient_dict['PIRADS_score'] = mat_file['casesTableArr'][0][0][1][0][0]
        patient_dict['curGleason_score'] = mat_file['casesTableArr'][0][0][2][0][0]
        patient_dict['maxGleason_score'] = mat_file['casesTableArr'][0][0][3][0][0]

        patient_dict['PIRADS_map'] = mat_file['casesTableArr'][0][0][4]
        patient_dict['curGleason_map'] = mat_file['casesTableArr'][0][0][5]
        patient_dict['maxGleason_map'] = mat_file['casesTableArr'][0][0][6]

        patient_dict['mask'] = mat_file['PMask']
        patient_dict['zone_map'] = mat_file['casesTableArr'][0][0][7]

        data.append(patient_dict)
    return data


def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def feat_extraction_1d(image):
    
    # mean    
    m = simg.generic_filter(image,np.mean,size=(3,3))
    #std
    std = simg.generic_filter(image,np.std,size=(3,3))
    #skew
    #skew = simg.generic_filter(image,stats.skew,size=(3,3))
    #kurtosis
    #kurt = simg.generic_filter(image,stats.kurtosis,size=(3,3))
    
    image = np.array(image).flatten()
    m = m.flatten()
    std = std.flatten()
    
    features_1d = [np.array(image), m, std]#, skew, kurt]
    return features_1d

def feat_extraction_2d(image):
    """
    # Haralick and GLCM
    im = remap(image, np.min(image), np.max(image), 0, np.max(image.shape))
    glcm = greycomatrix(im.astype(int), [1,2],[0, np.pi/2, np.pi, 3*np.pi/2], levels=np.max(image.shape)+1, symmetric=True, normed=True)
  
    # Contrast
    con = greycoprops(glcm, 'contrast')
    print(con.shape)
    #dissimilarity
    diss = greycoprops(glcm, 'dissimilarity')
    #homogeneity
    hom = greycoprops(glcm, 'homogeneity')
    #energy
    energy = greycoprops(glcm,'energy')
    #correlation
    corr = greycoprops(glcm,'correlation') 
    """
    
    
    return con,diss,hom,energy,corr