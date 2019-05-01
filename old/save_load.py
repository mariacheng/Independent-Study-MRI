#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

def load_data_and_save(dir_path, filenames):
    data = import_data(dir_path)
    return data
    #pdata = organize_pdata(data)

    #del data
    #zdata = organize_zdata(pdata)

    #save_obj(pdata, filenames[0])
    #save_obj(zdata, filenames[1])

def save_obj(obj, name ):
    with open('./' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def gen_X_matrix(ddict, zone):
    mods = ['ADC','CDI','HBV']
    features = ['orig','mean','std','skew','kurt']
    X = {}
    Y = []

    for mod in mods:
        for feat in features:
            X[mod+'_'+feat] = []

    for p in ddict:
        gs = pdata[p][zone]['GS'][0::3] # Take every third element (there are duplicates because of the modes)
        for slice in range(0, len(pdata[p][zone]['ADC']['orig'])):
            if pdata[p][zone]['GS'][slice] > 6:
                pCa = '1'
            else:
                pCa = '0'
            Y.extend(pCa*len(pdata[p][zone][mod][feat][slice]))
            for mod in mods:
                for feat in features:
                    X[mod+'_'+feat].extend(pdata[p][zone][mod][feat][slice])
        """
        pdata[p][zone]['GS']
        for i in range(0,len(pdata[p][zone]['ADC'])):
            if pdata[p][zone]['GS'][i] > 6:
                pCa = '1'
            else:
                pCa = '0'
        
            
            # Append if num samples match for each mod
            if len(pdata[p][zone]['ADC'][i]) == len(pdata[p][zone]['CDI'][i]) == len(pdata[p][zone]['HBV'][i]):
                Y.extend(pCa*len(pdata[p][zone]['ADC'][i]))
                for mod in mods:
                    X[mod].extend(pdata[p][zone][mod][i])
 #               for j in range(0, len(pdata[p][zone]['ADC'][i])):
 #                   if len(pdata[p][zone]['ADC'][i][j]) == len(pdata[p][zone]['CDI'][i][j]) == len(pdata[p][zone]['HBV'][i][j]):
 #                       dat = [d for d in pdata[p]['PZ']['ADC'][i][j] if d != 0]
 #                       Y.extend(pCa*len(dat))
 #                       for mod in mods:
 #                           data = [d for d in pdata[p]['PZ'][mod][i][j] if d != 0]
 #                           X[mod].extend(data)
         """
    return X, Y


def organize_pdata(data):
    pdata = {}
    categories = ['ADC','CDI','HBV','GS']
    features = ['orig','mean','std','skew','kurt']
    
    PZ_zones = [1, 4, 5, 7, 9, 10]
    TZ_zones = [2, 3]
    CZ_zones = [6, 8]

    for patient in data:
        pid = patient['id']
        pdata[pid] = {}
        pdata[pid]['PZ'] = {}
        pdata[pid]['TZ'] = {}
        pdata[pid]['CZ'] = {}
        for cat in categories[:-1]:
            for zone in pdata[pid]:
                pdata[pid][zone][cat] = {}
                pdata[pid][zone]['GS'] = []
                for feat in features:
                    pdata[pid][zone][cat][feat] = []

        # check if the segmentation map has the same number of slices as the mri volume
        if patient['zone_map'].shape[-1] != patient['ADC'].shape[-1]:# and patient['zone_map'].shape != patient['CDI'].shape and patient['zone_map'].shape != patient['CDI'].shape:
            continue

        for slice_index in range(patient['ADC'].shape[-1]):
            
            # To trim the slices
            
            # Check if the slice contains the prostate
            if np.max(patient['mask'][:,:,slice_index]) > 0:
                #binary_mask = patient['mask'][:,:,slice_index] == 255  # create a binary mask of the slice
                zone_map = patient['zone_map'][:,:,slice_index] > 0
                trimmed_zone_map, ix, jx, iy, jy = trim_zeros_2D(zone_map)
                
                # For each modality compute feature extraction algorithms
                # apply the mask to the slice and trim zeros
                for mod in categories[:-1]:
                    mod_slice = (patient[mod][:,:,slice_index] * zone_map)[ix:jx+1,iy:jy+1]
                    mean,std,skew,kurt = feat_extraction_1d(mod_slice)

                    # To trim the zones
                    for zone_index in range(10):
            
                    # check the zone map to see if the slice contains the zone
                        if zone_index + 1 in patient['zone_map'][:,:,slice_index]:
                
                            # extract zone GS and maps into a dictionary                    
                            bin_mask = trimmed_zone_map == zone_index + 1  # create a binary mask
                            
                            mod_crop = mod_slice[bin_mask == 1]
                            mean_crop = mean[bin_mask == 1]
                            std_crop = std[bin_mask == 1]
                            skew_crop = skew[bin_mask == 1]
                            kurt_crop = kurt[bin_mask == 1] 

                            gs = patient['maxGleason_map'][slice_index][zone_index]

                            # add data to correct zone
                            zindex = zone_index + 1
                            if zindex in PZ_zones:
                                pr_zone = 'PZ'
                            elif zindex in TZ_zones:
                                pr_zone = 'TZ'
                            else:
                                pr_zone = 'CZ'

                            pdata[pid][pr_zone][mod]['orig'].append(np.array(mod_crop))
                            pdata[pid][pr_zone][mod]['mean'].append(np.array(mean_crop))
                            pdata[pid][pr_zone][mod]['std'].append(np.array(std_crop))
                            pdata[pid][pr_zone][mod]['skew'].append(np.array(skew_crop))
                            pdata[pid][pr_zone][mod]['kurt'].append(np.array(kurt_crop))
                            pdata[pid][pr_zone]['GS'].append(gs)
            
                    """    
                    # ADC crops
                    adc_crop = patient['ADC'][:,:,slice_index][bin_mask == 1]
                    #adc_mask = patient['ADC'][:,:,slice_index] * binary_mask  # apply the mask to the slice
                    #adc_crop = trim_zeros_2D(adc_mask)  # trim the slice to the dimensions of the prostate zone

                    # CDI crops
                    cdi_crop = patient['CDI'][:,:,slice_index][bin_mask == 1]
                    #cdi_mask = patient['CDI'][:,:,slice_index] * binary_mask  # apply the mask to the slice
                    #cdi_crop = trim_zeros_2D(cdi_mask)  # trim the slice to the dimensions of the prostate zone

                    # HBV crops
                    hbv_crop = patient['HBV'][:,:,slice_index][bin_mask == 1]
                    #hbv_mask = patient['HBV'][:,:,slice_index] * binary_mask  # apply the mask to the slice
                    #hbv_crop = trim_zeros_2D(hbv_mask)  # trim the slice to the dimensions of the prostate zone
                    
                    # GS scores
                    gs = patient['maxGleason_map'][slice_index][zone_index]

                    pdata[pid][pr_zone]['ADC'].append(np.array(adc_crop))
                    pdata[pid][pr_zone]['CDI'].append(np.array(cdi_crop))
                    pdata[pid][pr_zone]['HBV'].append(np.array(hbv_crop))
                    pdata[pid][pr_zone]['GS'].append(gs)
                    """
    return pdata

def organize_zdata(pdata):
    # zone format of pdata
    zdata = {}
    zones = ['PZ','CZ','TZ']
    for zone in zones:
        zdata[zone] = {}
    for patient in pdata:
        for zone in pdata[patient]:
            zdata[zone][patient] = pdata[patient][zone]
    
    return zdata