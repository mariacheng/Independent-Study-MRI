# MRI-ISP
MRI Independent Study Project -
Development of Radiomic-Driven Classifiers for Comparison Against Sequences Produced Using Deep Neural Network Architectures for Image-Drive Cancer Diagnosis

This repo holds the Python files used for preliminary analysis of MRI data (provided by Sunnybrook hospital) for comparison between classifiers produced by:
* imaging modalities (ADC, HBV, CDI)
* feature extraction methods (zone or slice level)
* deep learning radiomic sequencer results (using CNNs)

## Data
* utils.py - convert MATLAB matfile data into useable Python dict

## Pre-processing
* features.py - Feature extraction functions - either returns a single feature value or performs local feature extraction

## Data preprocessing
* feature_extraction.py - Zone-level feature extraction
* feature_extraction_slice.py - Slice-level feature extraction

## Training, Validation, and Testing
* train_model.py - Training and testing various models for zone or slice level feature extraction. Validated using 5-folds (predetermined on a patient-level)
