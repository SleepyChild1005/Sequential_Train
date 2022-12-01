import torch.utils.data as data
import numpy as np
import os
from medpy.io import load
from .patient_id_list import PatientList


class TargetDataset(data.Dataset):
    def __init__(self, tg_data_dir, mode, patientlist: PatientList, fold_number=0, total_fold=4):
        self.tg_data_dir = tg_data_dir
        self.tg_seq_vol_dir = self.tg_data_dir + 'vol/'
        self.tg_seq_mask_dir = self.tg_data_dir + 'mask/'

        tg_patient_ids = patientlist.get_patient_ids()
        tg_test_patient_ids = tg_patient_ids[fold_number::total_fold]

        # set train/test mode
        if mode == "train":
            self.tg_mode_patient_ids = [i for i in tg_patient_ids if i not in tg_test_patient_ids]
        elif mode == "test":
            self.tg_mode_patient_ids = tg_test_patient_ids
        ###
        # print(mode, self.tg_mode_patient_ids)

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
        mask = mask[:, :, 1, :]
        mask = np.transpose(mask, (2, 0, 1))
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



