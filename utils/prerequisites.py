import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help='GPU device number')
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers')
    parser.add_argument("--epoch", type=int, default=500, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--lr', type=float, default=5e-5, help='generator learning rate')
    parser.add_argument('--total_fold_num', type=int, default=4, help='num of total fold')
    parser.add_argument('--fold_num', type=int, nargs='+', default=[0, 1, 2, 3], help='num of target fold')

    parser.add_argument("--num_sequence", type=int, default=3, help='number of sequence')
    parser.add_argument("--image_size", type=int, default=64, help='image size')

    parser.add_argument("--model_type", type=str, default='sensor3d', help='sensor_3d | attention | transfer_learning | sa_unet')

    parser.add_argument("--exp_name", type=str, default='sensor3d_multilabel', help='experiment name')
    parser.add_argument("--cross_val", type=bool, default=True, help='cross validation')
    parser.add_argument("--target_lb", type=int, default=2, help='target label')
    parser.add_argument("--test_plot_prob", type=float, default=0.5, help='test plotting prob cut off value')
    parser.add_argument("--kernel_size", type=int, default=3, help='test plotting prob cut off value')

    parser.add_argument("--pretrained_model", type=str,
                        default='pretrained_model/nodule_sensor3d_attention_final.pth',
                        help='path to pretrained model (pth file)')
    parser.add_argument("--dataset_path", type=str,
                        default='./Data/Sequence_Built/Sequential_Slices_Clipping/',
                        help='path to target dataset path')
    opt = parser.parse_args()

    return opt


def get_result_dirs_ready(path, attr: dict):
    exp_name = attr['exp_name']
    exp_folder = attr['exp_folder']
    Path(f'{path}/{get_result_folder_name()}/').mkdir(exist_ok=True)
    Path(f'{path}/{get_log_folder_name()}/').mkdir(exist_ok=True)

    output_dir = f'{path}/{get_result_folder_name()}/{exp_name}/'
    writer_dir = f'{path}/{get_log_folder_name()}/{exp_name}/'
    Path(output_dir).mkdir(exist_ok=True)
    Path(writer_dir).mkdir(exist_ok=True)

    output_current_dir = f'{output_dir}/{exp_folder}'
    writer_current_dir = f'{writer_dir}/{exp_folder}'
    Path(output_current_dir).mkdir(exist_ok=True)
    Path(writer_current_dir).mkdir(exist_ok=True)

    return output_current_dir, writer_current_dir


def get_result_folder_name():
    folder_name = 'Opthal_CT_Learning_Results'
    return folder_name


def get_log_folder_name():
    folder_name = 'Opthal_CT_Learning_Logs'
    return folder_name


def make_folder(base, folder):
    Path(f'{base}/{folder}').mkdir(exist_ok=True)
    return str(Path(f'{base}/{folder}'))
