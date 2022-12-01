from utils import get_parser, get_result_dirs_ready, make_folder
from dataset import PatientList, get_datasets
from trainer import OpthalCT_trainer

import os
from datetime import datetime
import dateutil.tz

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    opt = get_parser()
    device = "gpu" + opt.devicet

    training_start = datetime.now(dateutil.tz.tzlocal())
    timestamp = training_start.strftime('%Y%m%d_%H%M')
    exp_folder_name = f'{timestamp}_{device}'

    base_dir = os.getcwd()
    attr={}
    attr['exp_name'] = opt.exp_name
    attr['exp_folder'] = exp_folder_name

    out_dir, write_dir = get_result_dirs_ready(base_dir, attr)
    patient_list = PatientList(shuffle=True, total_fold=opt.total_fold_num)

    train_dataset, test_dataset = get_datasets(patient_list, out_dir, opt.dataset_path, opt.cross_val, opt.total_fold_num)

    # get dataset for each fold, then train each fold
    for fold in opt.fold_num:
        print(fold, 'th fold ::: training start')
        print(fold, 'th fold train/test seq:: ', len(train_dataset[fold]), len(test_dataset[fold]))
        print(fold, 'th fold train/test patient:: ', train_dataset[fold].get_num_patient(), test_dataset[fold].get_num_patient())

        train_dataloader = DataLoader(train_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = DataLoader(test_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)

        # directory for log, results
        fold_name = f'Fold_{fold}'
        out_fold = make_folder(out_dir, fold_name)
        write_fold = make_folder(write_dir, fold_name)
        writer = SummaryWriter(write_fold)

        # trainer from trainer_multilabel.py
        algo = OpthalCT_trainer(epochs=opt.epoch,
                                gpu=opt.device,
                                batch_size=opt.batch_size,
                                image_size=opt.image_size,
                                learning_rate=opt.lr,
                                output_dir=out_fold,
                                model_type=opt.model_type,
                                pretrained_model_dir=opt.pretrained_model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                writer=writer)

        start_t = datetime.now()
        algo.train()
        end_t = datetime.now()

        print(f'{fold}-th fold ::: total time for training: {end_t - start_t}')






