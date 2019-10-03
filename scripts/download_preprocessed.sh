#!/usr/bin/env bash

# Processed image features for VisDial v1.0, using VGG-19
wget https://s3.amazonaws.com/visual-dialog/data/v1.0/data_img_vgg16_relu7_train.h5 -O data/visdial/data_img.h5
# Processed dialog data for VisDial v1.0
wget  https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_data_train.h5 -O data/visdial/chat_processed_data.h5
wget  https://s3.amazonaws.com/visual-dialog/data/v1.0/visdial_params_train.json -O data/visdial/chat_processed_params.json

# Dense annotations for NDCG calculations
wget https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0 -O data/visdial/visdial_1.0_val_dense_annotations.json