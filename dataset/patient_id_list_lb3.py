import numpy as np
from pathlib import Path

class PatientList:
    def __init__(self, shuffle:bool=False, total_fold:int=4):
        self.shuffle = shuffle
        self.patient_ids:np.ndarray = self.init_patient_ids()
        self.total_fold = total_fold

    def init_patient_ids(self):
        # axial files only, refer preprocessing result/ sequence builder record, log
        patient_ids = np.array([
            'Image_1', 'Image_5', 'Image_8', 'Image_9', 'Image_12', 'Image_16',
            'Image_22', 'Image_23', 'Image_27', 'Image_28', 'Image_31', 'Image_32',
            'Image_33', 'Image_35',  'Image_37', 'Image_39', 'Image_41', 'Image_45',
            'Image_46', 'Image_47', 'Image_49', 'Image_51', 'Image_52', 'Image_53',
            'Image_55', 'Image_57', 'Image_59', 'Image_60','Image_62', 'Image_66',
            'Image_67', 'Image_69', 'Image_70', 'Image_72'])
        # excluded 13, 30, 65, 68 // 59 --> OD deleted
        # 32, 39, 69 --> light CTs
        # tg_patient_ids = sorted(tg_patient_ids)

        if self.shuffle:
            np.random.shuffle(patient_ids)
        else:
            patient_ids = sorted(patient_ids)

        return patient_ids

    def get_patient_ids(self):
        return self.patient_ids

    def shuffle_patient_ids(self):
        np.random.shuffle(self.patient_ids)

    def write_log_text(self, dir, fold: int=0):
        file_name = 'patient_id_log'
        txt_path = f'{dir}/Fold_{fold}'
        Path(txt_path).mkdir(exist_ok=True)
        f = open(f'{txt_path}/{file_name}{fold}.txt', 'a')
        f.write('\n##################################')
        f.write(f'\nPatient Ids Fold {fold}\n')
        f.write('\n----------------------\n')
        f.write('\nTotal Ids')
        f.write(f'\n{self.get_patient_ids()}')
        f.write('\n----------------------\n')
        f.write('\nTest Ids')
        f.write(f'\n{self.get_test_ids(fold)}')
        f.close()

    def get_test_ids(self, fold: int=0):
        test_ids = self.patient_ids[fold::self.total_fold]
        return test_ids
