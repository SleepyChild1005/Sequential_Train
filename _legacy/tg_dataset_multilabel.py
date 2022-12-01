import torch.utils.data as data
import numpy as np
import os
from medpy.io import load
# used in mainYL

class TargetDataset(data.Dataset):
    def __init__(self, tg_data_dir, mode, fold_number=0, total_fold=4):
        self.tg_data_dir = tg_data_dir
        self.tg_seq_vol_dir = self.tg_data_dir + 'vol/'
        self.tg_seq_mask_dir = self.tg_data_dir + 'mask/'
        self.shuffled = False

        # axial files only, refer preprocessing result/ sequence builder record, log
        tg_patient_ids = np.array(
            ['Image_12', 'Image_28', 'Image_31', 'Image_36', 'Image_37', 'Image_41',
             'Image_45', 'Image_47', 'Image_49', 'Image_51', 'Image_52', 'Image_53',
             'Image_55', 'Image_57', 'Image_59', 'Image_60', 'Image_62', 'Image_64',
             'Image_66', 'Image_67', 'Image_70', 'Image_72'])
        # excluded 13, 30, 65, 68 // 59 --> OD deleted
        # 32, 39, 69 --> light CTs
        # tg_patient_ids = sorted(tg_patient_ids)

        if not self.shuffled:
            np.random.shuffle(tg_patient_ids)
            self.shuffled=True
        else:
            pass
        # set train/test mode
        f = open(self.tg_data_dir + f'../dataset_log{fold_number}.txt', 'w')
        f.write('tg_patient_ids %.5f \n')
        f.write(f'{tg_patient_ids} %.5f \n')
        tg_test_patient_ids = tg_patient_ids[fold_number::total_fold]
        f.write('\n')
        f.write('test ids\n')
        f.write(f'{tg_test_patient_ids} %.5f\n')
        f.close()
        if mode == "train":
            self.tg_mode_patient_ids = [i for i in tg_patient_ids if i not in tg_test_patient_ids]
        elif mode == "test":
            self.tg_mode_patient_ids = tg_test_patient_ids
        ###
        
        self.tg_seq_filename = self.load_filenames(self.tg_seq_vol_dir, self.tg_mode_patient_ids)

    # return list of filenames
    def load_filenames(self, data_dir, mode_patient_id):
        target_filenames = []
        filenames = os.listdir(data_dir)
        for filename in filenames:
            patient_id = filename.split("__")[0]
            if patient_id in mode_patient_id:
                target_filenames.append(filename)
        return target_filenames
    # return volume data
    def get_volume(self, data_dir):
        volume, _ = load(data_dir)
        volume = np.expand_dims(volume, axis=0)
        volume = np.transpose(volume, (3, 0, 1, 2))        
        return volume
    # return mask data
    def get_mask(self, data_dir):
        mask, _ = load(data_dir)
        mask = mask[:,:,1,:]
        mask = np.transpose(mask,(2,0,1))
        return mask

    def get_num_patient(self):
        return len(self.tg_mode_patient_ids)

    def __len__(self):
        return len(self.tg_seq_filename)
    # return volume, mask data of each index
    def __getitem__(self, index):
        vol_path = self.tg_seq_vol_dir + self.tg_seq_filename[index]
        mask_path = self.tg_seq_mask_dir + self.tg_seq_filename[index]

        vol = self.get_volume(vol_path)
        mask = self.get_mask(mask_path)

        return vol, mask


        
   