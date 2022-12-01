import numpy as np
from medpy.io import load, save

import os
import argparse
import cv2

level = 48
window = 400

### mask filenames == vol filnames
class Extractor:
    # init data/save directory, number of sequences
    def __init__(self, data_dir, save_dir, num_sequence = 3):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.num_sequence = num_sequence

        self.filenames = os.listdir(data_dir + 'Scan/')
    

    def extract(self):
        # Create directry for saving
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = self.save_dir
        rad = self.num_sequence//2

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'vol/', exist_ok=True)
        os.makedirs(save_path+'mask/', exist_ok=True)

        # txt files for sequence building record/log
        recordtxt = open(save_path + 'sequence_record.txt', 'w')
        logtxt = open(save_path + 'sequence_log.txt', 'w')
        success_cnt = 0
        fail_cnt = 0

        for filename in self.filenames:
            print(filename)
            # load scan/mask preprocessed files
            scan_vol,_ = load(self.data_dir + 'Scan/' + filename)
            mask_vol,_ = load(self.data_dir + 'Mask/' + filename)

            scan_vol = np.clip(scan_vol, level-window/2, level + window/2)

            masked_axial_idx_list = []
            for axial_idx in range(mask_vol.shape[0]):
                if axial_idx==0 or axial_idx==mask_vol.shape[0]-1 or np.sum(mask_vol[axial_idx,:,:,:])==0:
                    continue
                masked_axial_idx_list.append(axial_idx)

            recordtxt.write(filename + ' : ' + str(len(masked_axial_idx_list)) + '\n')

            # cut value len change, 5
            if (len(masked_axial_idx_list) < 5):
                fail_cnt += 1
                continue

            if len(masked_axial_idx_list) > 0 and masked_axial_idx_list[-1] < mask_vol.shape[0]:
                masked_axial_idx_list.insert(0, masked_axial_idx_list[0]-1)
                masked_axial_idx_list.append(masked_axial_idx_list[-1]+1)
            else :
                print(filename)
                print(np.sum(mask_vol[0,:,:,:]))
                print(np.sum(mask_vol[mask_vol.shape[0]-1,:,:,:]))
                print(len(masked_axial_idx_list))
            # write success if done
            success_cnt += 1
            logtxt.write(filename + '  :: success :: axial idx cnt ' + str(len(masked_axial_idx_list)) + '\n')
            for axial_idx in masked_axial_idx_list:
                scan_sequence = scan_vol[axial_idx - rad:axial_idx + rad + 1, :, :]
                mask_sequence = mask_vol[axial_idx - rad:axial_idx + rad + 1, :, :, :]

                # zero np array for resized sequences
                resized_scan_sequence = np.zeros((self.num_sequence, 64, 64))
                resized_mask_sequence = np.zeros((self.num_sequence, 64, 64, 3))

                for i in range(self.num_sequence):
                    ### resize
                    resized_scan_sequence[i, :, :] = cv2.resize(scan_sequence[i, :, :], dsize=(64, 64),
                                                                interpolation=cv2.INTER_LINEAR)
                    resized_mask_sequence[i, :, :, 0] = cv2.resize(mask_sequence[i, :, :, 0].astype('float32'),
                                                            dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
                    resized_mask_sequence[i, :, :, 1] = cv2.resize(mask_sequence[i, :, :, 1].astype('float32'),
                                                                   dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
                    resized_mask_sequence[i, :, :, 2] = cv2.resize(mask_sequence[i, :, :, 2].astype('float32'),
                                                                   dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

                resized_mask_sequence = np.where(resized_mask_sequence >= 0.5, 1, 0)
                resized_scan_sequence = (resized_scan_sequence - np.min(resized_scan_sequence)) / np.ptp(
                    resized_scan_sequence)

                # transpose for better code readability in trainer
                resized_scan_sequence = np.transpose(resized_scan_sequence, (1, 2, 0))
                resized_mask_sequence = np.transpose(resized_mask_sequence, (1, 2, 0, 3))

                save(resized_scan_sequence, save_path + 'vol/' + filename + '_z' + str(axial_idx) + '.nii')
                save(resized_mask_sequence, save_path + 'mask/' + filename + '_z' + str(axial_idx) + '.nii')
        # write log, record txt
        logtxt.write('\n'+':: Success Count : '+str(success_cnt) + ' ::: Failure Count : '+str(fail_cnt) + ' :: ')
        logtxt.close()
        recordtxt.close()

                    
if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./Preprocessing_Result/Volume/', help='vol_data_dir')
    parser.add_argument("--save_dir", type=str, default='./Sequence_Built/Sequential_Slices_Clipping/', help="save_dir")
    opt = parser.parse_args()

    builder = Extractor(data_dir=opt.data_dir, save_dir=opt.save_dir)
    builder.extract()