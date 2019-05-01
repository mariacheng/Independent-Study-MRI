# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:54:20 2019

@author: Maria

Zone-level feature extraction
"""
import numpy as np
from preprocessing.features import *
from data.utils import *

patient_files = BASE_DIR.glob('data/data/*.mat')
modalities = ['HBV'] # Choose between ADC, HBV, CDI (MRI modalities)
max_adc = 3949
wZone = True # append zone number to features
if wZone:
    save_zone = 'wZone'
else:
    save_zone = 'woZone'

#print(len(list(patient_files))) #104

### Load modality data according to predetermined 5folds
fold_dict = {}
with open('5folds.txt', 'r') as f:
    for line in f:
        pid, fold_no = line.split()
        fold_dict[str(pid)] = int(fold_no)

zone_data = {}
zone_data['fold'] = []
zone_data['zone'] = []
zone_data['masked_zone'] = []
zone_data['gleason'] = []
zone_data['features'] = []

for file in patient_files:
    patient_dict = mat2dict(file)
    patient_fold = fold_dict.get(patient_dict['id'], 0)
    features = np.empty(0)
    modality = modalities[0]

    #extract data from dict, if the fold exists
    if patient_fold:
        if patient_dict[modality].shape[-1] != patient_dict['zone_map'].shape[-1]:
            continue

        for slice_idx in range(patient_dict[modality].shape[-1]):
            data = patient_dict[modality][:,:,slice_idx]
            zone_map = patient_dict['zone_map'][:,:,slice_idx]
            max_gleason = patient_dict['maxGleason_map'][slice_idx,:]

            if np.any(zone_map): # if the slice has any data
                for i in range(10): # there are 10 zones
                    zone_idx = i+1
                    zone_mask = zone_map == zone_idx
                    if np.any(zone_mask):
                        zone_data['fold'].append(patient_fold)
                        zone_data['zone'].append(zone_idx)
                        zone_data['gleason'].append(max_gleason[i])

                        if modality == 'ADC':
                            mask_zone = crop_resize_region(data, zone_mask, True, (32,32), True, max_adc)
                        else:
                            mask_zone = crop_resize_region(data, zone_mask, True, (32,32))

                        zone_data['masked_zone'].append(mask_zone)

np.save(f'./data/zone_data_{modalities[0]}.npy', zone_data)


### Feature extraction
#PZ = {1,5,7,9,10,4}
zone_data = np.load(f'./data/zone_data_{modalities[0]}.npy')
zone_data = zone_data[()]

for f in range(5):
    fold_idx = np.array([zone_data['fold']]) == f+1
    zones = np.array([zone_data['zone']])[fold_idx]
    fold_data = np.array([zone_data['masked_zone']])[fold_idx]
    labels = (np.array([zone_data['gleason']])[fold_idx] > 0).astype(np.uint8)
    feature_data = []
    zone_labels = []

    for z, zone in enumerate(fold_data):

        features = np.empty(0)
        if wZone:
            features = np.concatenate((features, np.array([zones[z]])))
        feat_1d = feat_extraction_1d(zone)
#        haralick_feat = haralick_GLCM_features(zone)
        lbp_feat = local_binary_pattern_features(zone)

        features = np.concatenate((features, np.array(zone.ravel()), feat_1d, lbp_feat)) # add the zone as a feature
        feature_data.append(features)
        zone_labels.append(labels[z])
    print(f'Saving... fold_{f+1}_{modalities[0]}_{save_zone}')
    np.save(f'./data/fold_{f+1}_{modalities[0]}_{save_zone}_zone_features.npy', feature_data)
    np.save(f'./data/fold_{f+1}_{modalities[0]}_{save_zone}_zone_labels.npy', zone_labels)
