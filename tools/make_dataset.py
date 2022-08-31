import os
import shutil
import pdb

source_dir = './DATASET/AMOS22'
target_dir_ct = './DATASET/AMOS22_raw/AMOS22_raw_data/Task01_AMOS_CT'
target_dir_mr = './DATASET/AMOS22_raw/AMOS22_raw_data/Task02_AMOS_MR'
target_dir_ct_mr = './DATASET/AMOS22_raw/AMOS22_raw_data/Task03_AMOS_CT+MR'
target_dir_crop = './DATASET/AMOS22_raw/AMOS22_cropped_data/'
target_dir_preprocessed = './DATASET/AMOS22_preprocessed/'

if not os.path.exists(target_dir_ct):
    os.makedirs(target_dir_ct+'/imagesTr')
    os.makedirs(target_dir_ct+'/labelsTr')
    os.makedirs(target_dir_ct+'/imagesTs')
if not os.path.exists(target_dir_mr):
    os.makedirs(target_dir_mr+'/imagesTr')
    os.makedirs(target_dir_mr+'/labelsTr')
    os.makedirs(target_dir_mr+'/imagesTs')
if not os.path.exists(target_dir_ct_mr):
    os.makedirs(target_dir_ct_mr+'/imagesTr')
    os.makedirs(target_dir_ct_mr+'/labelsTr')
    os.makedirs(target_dir_ct_mr+'/imagesTs')

# load files from raw dataset
mix_train_images = os.listdir(os.path.join(source_dir, 'imagesTr'))
mix_val_images = os.listdir(os.path.join(source_dir, 'imagesTs'))

ct_train_images = []
mr_train_images = []
ct_val_images = []
mr_val_images = []
for img in mix_train_images:
    if int(img[5:9]) < 500:
        ct_train_images.append(img)
    else:
        mr_train_images.append(img)
for img in mix_val_images:
    if int(img[5:9]) < 500:
        ct_val_images.append(img)
    else:
        mr_val_images.append(img)

# put image and label files to corresponding file
for img in ct_train_images:
    shutil.copy(source_dir+'/imagesTr/'+img, target_dir_ct+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTr/'+img, target_dir_ct+'/labelsTr/')
    shutil.copy(source_dir+'/imagesTr/'+img, target_dir_ct_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTr/'+img, target_dir_ct_mr+'/labelsTr/')
for img in ct_val_images:
    shutil.copy(source_dir+'/imagesTs/'+img, target_dir_ct+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTs/'+img, target_dir_ct+'/labelsTr/')
    shutil.copy(source_dir+'/imagesTs/'+img, target_dir_ct_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTs/'+img, target_dir_ct_mr+'/labelsTr/')
for img in mr_train_images:
    shutil.copy(source_dir+'/imagesTr/'+img, target_dir_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTr/'+img, target_dir_mr+'/labelsTr/')
    shutil.copy(source_dir+'/imagesTr/'+img, target_dir_ct_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTr/'+img, target_dir_ct_mr+'/labelsTr/')
for img in mr_val_images:
    shutil.copy(source_dir+'/imagesTs/'+img, target_dir_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTs/'+img, target_dir_mr+'/labelsTr/')
    shutil.copy(source_dir+'/imagesTs/'+img, target_dir_ct_mr+'/imagesTr/')
    shutil.copy(source_dir+'/labelsTs/'+img, target_dir_ct_mr+'/labelsTr/')

# json files
shutil.copy('./splits/dataset_ct.json', target_dir_ct+'/dataset.json')
shutil.copy('./splits/dataset_mr.json', target_dir_mr+'/dataset.json')
shutil.copy('./splits/dataset_ct+mr.json', target_dir_ct_mr+'/dataset.json')

