from pathlib import Path
import scipy.io
from skimage.transform import resize
import numpy as np
import pickle

BASE_DIR = Path(__file__).resolve().parent
ZONE_DIR = BASE_DIR.joinpath('data\data_candidates_mat')

def mat2dict(file):
    mat_file = scipy.io.loadmat(file)

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

    return patient_dict

def mat2dictfull(dir_path):
    data = []
    for i, file in enumerate(os.listdir(dir_path)):

        mat_file = scipy.io.loadmat(os.path.join(dir_path, file))

        pid = mat_file['casesTableArr'][0][0][0][0][0][0][0]

        patient_dict = {}

        patient_dict['id'] = pid
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

def crop_resize_region(image, mask, doResize=False, outputsize=(32,32), invert=False, invertMax=1):
    r, c = mask.nonzero()

    image = image * mask
    if invert:
        image = image + np.invert(mask) * invertMax

    region = image[r[0]:r[-1]+1, c.min():c.max()+1]
    if doResize:
        region = resize(region, outputsize, mode='symmetric', preserve_range=True)

    return region
