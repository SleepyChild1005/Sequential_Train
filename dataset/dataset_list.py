from .patient_id_list import PatientList
from .target_dataset import TargetDataset


def get_datasets(patient_list: PatientList, out_dir, dataset_path, is_cross_val:bool=True, total_folds:int=4):
    train_dataset = []
    test_dataset = []

    print('check opt.total_fold_num = ', total_folds)

    for fold in range(total_folds):
        if not is_cross_val:
            print('Shuffle Patients at Dataset')
            patient_list.shuffle_patient_ids()

        patient_list.write_log_text(out_dir, fold)
        train_dataset.append(TargetDataset(dataset_path, 'train', patient_list, fold, total_folds))
        test_dataset.append(TargetDataset(dataset_path, 'test', patient_list, fold, total_folds))
        print(f'Fold {fold} :: train_dataset_cnt :  {train_dataset[fold].get_num_patient()} '
              f':: test_dataset_cnt : {test_dataset[fold].get_num_patient()}')

    return train_dataset, test_dataset
