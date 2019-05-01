# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:57:24 2019

@author: Maria

Slice-level feature extraction
"""
import numpy as np
from preprocessing.features import *
from data.utils import *
from skimage import img_as_bool
from skimage.transform import resize

patient_files = BASE_DIR.glob('data/data/*.mat')
modalities = ['HBV'] # Choose between ADC, HBV, CDI (MRI modalities)
max_adc = 3949
wZone = True # append zone number to features
if wZone:
    save_zone = 'wZone'
else:
    save_zone = 'woZone'

### Load modality data into predetermined 5folds
fold_dict = {}
with open('5folds.txt', 'r') as f:
    for line in f:
        pid, fold_no = line.split()
        fold_dict[str(pid)] = int(fold_no)

zone_data = {}
zone_data['fold'] = []
zone_data['zone'] = []
zone_data['gleason'] = []
zone_data['features'] = []
zone_data['masked_zone'] = []

for f, file in enumerate(patient_files):
    patient_dict = mat2dict(file)
    patient_fold = fold_dict.get(patient_dict['id'], 0)
    modality = modalities[0]

    #extract data from dict, if the fold exists
    if patient_fold:
        if patient_dict[modality].shape[-1] != patient_dict['zone_map'].shape[-1]:
            continue

        for slice_idx in range(patient_dict[modality].shape[-1]):
            data = patient_dict[modality][:,:,slice_idx]
            zone_map = patient_dict['zone_map'][:,:,slice_idx]
            slice_mask = patient_dict['mask'][:,:,slice_idx] == 255
            max_gleason = patient_dict['maxGleason_map'][slice_idx,:]

            if np.any(zone_map): # if the slice has any data
                if modality == 'ADC':
                    slice_region = crop_resize_region(data, slice_mask, True, (32,32), True, max_adc)
                else:
                    slice_region = crop_resize_region(data, slice_mask, True, (32,32))

                # Feature extraction
                features = conv_feat_extraction_1d(slice_region)
                gabor_feat = gabor_features(slice_region)
                kirsch_feat = kirsch_features(slice_region)
                lbp_feat = conv_local_binary_pattern_features(slice_region)
                features = np.concatenate((slice_region.ravel().reshape(-1,1).T, features, gabor_feat, kirsch_feat, lbp_feat.reshape(-1,1).T))

                zone_map = crop_resize_region(zone_map, slice_mask, False)
                for i in range(10):
                    zone_idx = i+1
                    zone_mask = img_as_bool(resize(zone_map==zone_idx, (32,32)))

                    if np.any(zone_mask):
                        zone_features = features.T[zone_mask.ravel()]
                        mask_zone = crop_resize_region(slice_region, zone_mask, True, (32, 32))

                        zone_data['fold'].append(patient_fold)
                        zone_data['zone'].append(zone_idx)
                        zone_data['gleason'].append(max_gleason[i])
                        zone_data['features'].append(zone_features)
                        zone_data['masked_zone'].append(mask_zone)

    #print(f+1)

np.save(f'./data/slicezone_data_{modalities[0]}.npy', zone_data)

### Process features 
zone_data = np.load(f'./data/slicezone_data_{modalities[0]}.npy')
zone_data = zone_data[()]

for f in range(5):
    fold_idx = np.array([zone_data['fold']]) == f+1
    zones = np.array([zone_data['zone']])[fold_idx]
    label_data = (np.array([zone_data['gleason']])[fold_idx] > 0).astype(np.uint8)
    features = np.array([zone_data['features']])[fold_idx]
    feature_data = []
    labels = []

    for z, feature_set in enumerate(features):
        feature_set = np.array(feature_set)
        if wZone:
            feature_set = np.concatenate((feature_set, np.tile(zones[z], feature_set.shape[0]).reshape(-1,1)), axis=1)

        feature_data.extend(feature_set)
        labels.extend(np.tile(label_data[z], feature_set.shape[0]))
    print(f'Saving... fold_{f+1}_{modalities[0]}_{save_zone}')
    np.save(f'./data/fold_{f+1}_{modalities[0]}_{save_zone}_slice_features.npy', feature_data)
    np.save(f'./data/fold_{f+1}_{modalities[0]}_{save_zone}_slice_labels.npy', labels)
