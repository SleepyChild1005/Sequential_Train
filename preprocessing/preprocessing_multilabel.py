import h5py
import numpy as np
import cv2
import os
import glob
import pandas as pd
from medpy.io import save

# preprocessing

level = 48
window = 400

# Data directory, edit if your using private data set
data_path = '/home/ubuntu/data/Orbit/Normal/*.hdf'
data_path2 = '/home/ubuntu/data/Orbit/Normal2/*.hdf'

# Define save dirs
base_path = './Preprocessing_Result/'
capture_save_path = base_path + 'Volume_capture/'
vol_save_path = base_path + 'Volume/'

# Create dirs to save
if not os.path.exists(capture_save_path):
    os.makedirs(capture_save_path,exist_ok=True)
    os.makedirs(capture_save_path + 'Scan/head_crop/',exist_ok=True)
    os.makedirs(capture_save_path + 'Scan/OD/', exist_ok=True)
    os.makedirs(capture_save_path + 'Scan/OS/', exist_ok=True)
    os.makedirs(capture_save_path + 'Mask/', exist_ok=True)
    os.makedirs(capture_save_path + 'Mask/OD/', exist_ok=True)
    os.makedirs(capture_save_path + 'Mask/OS/', exist_ok=True)

if not os.path.exists(vol_save_path):
    os.makedirs(vol_save_path+ 'Scan/', exist_ok=True)
    os.makedirs(vol_save_path + 'Mask/', exist_ok=True)

# Checking basic file infos
CT_path_list = glob.glob(data_path)
CT_path_list += glob.glob(data_path2)
print('#########################################')
print("total number of CT : ", len(CT_path_list))
print('#########################################')

# open txt file for log
logtxt = open(base_path+'HDF_file_logs.txt','w+')

# open csv file
csv = pd.read_csv('./csv/Orbit_CT_info.csv')

# exception if error occurs
for CT_idx, CT_path in enumerate(CT_path_list):
    print(CT_idx,'th :: ',CT_path,'preprocessing start')
    # preprocessing main function
    try:
        f = h5py.File(CT_path, 'r')
        logtxt.write(str(CT_idx) + ' th CT :: ' + CT_path + ' :: success ::  \n')
        num_image = f['ExportData']['number_of_image'][()][0]
        CT_name = os.path.splitext(os.path.basename(CT_path))[0]
        
    except:
        print (CT_idx, ' th CT : ', CT_path, '-->   unable to read ')
        logtxt.write(str(CT_idx) + ' th CT :: ' + CT_path + ' :: unable to read :: arr overflow \n')
        continue

    print(CT_idx, ' th CT : ', CT_name,', num_images : ', num_image)
    print('-----> ',CT_path )

    idx_name = 'Image_' + str(CT_idx + 1)
    data = f['ExportData']['Image_1']['image'][()].astype(np.float)
    label = f['ExportData']['Image_1']['label'][()]
    img_name = str(f['ExportData']['Image_1']['name'][()][0])
    print ('img_name = ', img_name)
    img_name = img_name.replace("b'", '')
    img_name = img_name.replace("'", '')
    label_len = np.unique(label)
    print ('label length = ',label_len)

    #####
    print ('CT_idx = ', CT_idx)
    print ('CT_name = ', CT_name)
    print ('idx_name = ', idx_name)
    #####
    CT_sub_name = img_name
    print('**** working ****')

    try:
        depth = f['ExportData']['Image_1']['depth'][()][0]
        if depth == 1:
            continue
    except KeyError:
        depth = 1

    height = f['ExportData']['Image_1']['height'][()][0]
    width = f['ExportData']['Image_1']['width'][()][0]
    spacing = f['ExportData']['Image_1']['spacing'][()]

    print ('h x w x s = ', height, ', ', width,', ',spacing)

    data = np.reshape(data, (depth, height, width, int(data.shape[0]/(depth*height*width))))

    data = np.flip(data, 0)
    data = np.flip(data, 1)
    data = np.flip(data, 2)

    label = np.reshape(label, (depth, height, width))
    label = np.flip(label, 0)
    label = np.flip(label, 1)
    label = np.flip(label, 2)

    label_bg = np.where(label>0, 1, 0)

    label_pos = np.where(label_bg==1)
    label_slice_idx = np.unique(label_pos[0])
    depth_s = np.min(label_slice_idx)
    depth_e = np.max(label_slice_idx)
    print (depth_s," ~ ",depth_e)

    head_ymin = 0
    head_ymax = height-1
    head_zmin = 0
    head_zmax = width -1
    head_data = data[:, head_ymin:head_ymax, head_zmin:head_zmax, :]
    head_label = label[:, head_ymin:head_ymax, head_zmin:head_zmax]

    slice_idx = len(label_slice_idx)//2
    head_data_slice = head_data[label_slice_idx[slice_idx],:,:]

    cv2.imwrite(capture_save_path + 'Scan/head_crop/' + idx_name + '.png', 255 * (head_data_slice - np.min(head_data_slice)) / np.ptp(head_data_slice))

    eye_range_y = (height-1)*2//3
    eye_range_z = (width-1)//2

    # variables for loop
    _lp = 0
    is_OD = True
    _count = 5

    # for computer scientists, [ OD == right eye, OS == left_eye ]
    dict_eye_data = {}
    dict_eye_label = {}
    dict_eye_label_slice = {}
    tumor_cnt = {}

    # preprocessing loop
    while _lp < 2:
        if is_OD:
            eye_data = data[:, head_ymin:head_ymin + eye_range_y, head_zmax - eye_range_z:head_zmax, :]
            eye_label = label[:, head_ymin:head_ymin + eye_range_y, head_zmax - eye_range_z:head_zmax]
            current_eye = 'OD'
        else:
            eye_data = data[:, head_ymin:head_ymin+eye_range_y, head_zmin:head_zmin+eye_range_z, :]
            eye_label = label[:, head_ymin:head_ymin+eye_range_y, head_zmin:head_zmin+eye_range_z]
            current_eye = 'OS'

        dict_eye_data[current_eye] = eye_data
        dict_eye_label[current_eye] = {}
        dict_eye_label_slice[current_eye] = {}

        # adding dimension for labels, meaning (3,) is for 3 labels ((n,) for n labels)
        data_shape = eye_label.shape + (3,)
        eye_multilabel = np.zeros(data_shape)
        eye_multilabel[:, :, :, 0] = np.where(eye_label == 1, 1, 0)
        eye_multilabel[:, :, :, 1] = np.where(eye_label == 2, 1, 0)
        eye_multilabel[:, :, :, 2] = np.where(eye_label == 3, 1, 0)

        #
        dict_eye_label[current_eye][0] = eye_multilabel
        eye_data_slice = eye_data[label_slice_idx[slice_idx],:,:]
        dict_eye_label_slice[current_eye][0] = dict_eye_label[current_eye][0][label_slice_idx[slice_idx], :, :, :]

        cv2.imwrite(capture_save_path + 'Scan/' + current_eye + '/' + idx_name + '_scan_axial.png',
                    255 * (eye_data_slice - np.min(eye_data_slice)) / np.ptp(eye_data_slice))
        cv2.imwrite(capture_save_path + 'Mask/'+current_eye+'/' + idx_name + '_mask_axial.png',
                    255 * dict_eye_label_slice[current_eye][0])

        tumor_cnt[current_eye] = len(np.where(eye_label>0)[0])

        is_OD = False
        _lp += 1

    print('OD_tumor_cnt : ', tumor_cnt['OD'],',   OS_tumor_cnt : ', tumor_cnt['OS'])

    # saving normal data to nii file Scan for backkground, Mask for label
    if (tumor_cnt['OD'] * tumor_cnt['OS'] > 0):
        CT_vol_name = idx_name + '__OD.nii'
        save(dict_eye_data['OD'], vol_save_path + 'Scan/' + CT_vol_name)
        save(dict_eye_label['OD'][0], vol_save_path + 'Mask/' + CT_vol_name)

        CT_vol_name = idx_name + '__OS.nii'
        save(dict_eye_data['OS'], vol_save_path + 'Scan/' + CT_vol_name)
        save(dict_eye_label['OS'][0], vol_save_path + 'Mask/' + CT_vol_name)

# save log
logtxt.close()