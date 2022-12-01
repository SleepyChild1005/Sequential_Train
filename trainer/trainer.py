from model.loss import DiceLoss

from .utils import mul_rgb, color_rgb_filter
from .scores import compute_vs, compute_dice_coeff_train, compute_dice_coeff_test
from .get_model import model_getter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
import time
import cv2
from pathlib import Path


def get_label_mask(masks:torch.Tensor, label_num):
    label_dim = label_num-1
    mask_lb = masks[:, None, label_dim, :, :]
    return mask_lb


def get_loss_weight(masks:torch.Tensor):
    cnt_nodulePixel = masks.nonzero(as_tuple=False).size(0) + 1  # mask = 1 (nodule)
    cnt_allPixel = masks.numel()  # all mask pixel
    loss_weight = cnt_allPixel / cnt_nodulePixel
    return loss_weight


class sequentialSegTrainer(object):
    def __init__(self, epochs, gpu, batch_size, image_size, learning_rate, output_dir, model_type,
                 pretrained_model_dir, train_dataloader, test_dataloader, writer):
        model_folder = 'model'
        result_folder = 'result'
        self.model_dir = f'{output_dir}/{model_folder}'
        result_dir = f'{output_dir}/{result_folder}'
        Path(result_dir).mkdir(exist_ok=True)

        self.train_result_dir = os.path.join(output_dir, result_folder, 'train')
        self.test_result_dir = os.path.join(output_dir, result_folder, 'test')

        Path(self.model_dir).mkdir(exist_ok=True)
        Path(self.train_result_dir).mkdir(exist_ok=True)
        Path(self.test_result_dir).mkdir(exist_ok=True)

        self.epochs = epochs
        self.device = torch.device("cuda:%s" % gpu)
        self.batch_size = batch_size
        self.image_size = image_size

        self.learning_rate = learning_rate

        self.output_dir = output_dir
        self.model_type = model_type
        self.pretrained_model_dir = pretrained_model_dir

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.writer = writer

        self.target_label=1

    def fix_attention_parameters(self, model):
        for param in model.Attention3.parameters():
            param.requires_grad = False
        for param in model.Attention2.parameters():
            param.requires_grad = False
        for param in model.Attention1.parameters():
            param.requires_grad = False

    def train(self):

        sqNet = model_getter(self.device, self.image_size, self.model_type, self.pretrained_model_dir)

        total_param = sum(p.numel() for p in sqNet.parameters())
        train_param = sum(p.numel() for p in sqNet.parameters() if p.requires_grad)

        optimizer = optim.Adam(sqNet.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        criterion = DiceLoss().to(self.device)

        start_t = time.time()

        # variables for each label, total
        total_step = 0
        result_print_step = 10
        loss_print_step = 10
        start_time_step = time.time()
        test_img_export_interval = 5

        best_dice_score_lb = 0
        worst_dice_score_lb = 0

        ### trainer main function
        for epoch in range(self.epochs):
            if epoch == 10:
                result_print_step = 100
                test_img_export_interval = 20
            elif epoch == 100:
                result_print_step = 1000
            start_time_epoch = time.time()

            for data in self.train_dataloader:
                seq_vols, masks = data

                seq_vols = Variable(seq_vols).float().to(self.device)
                masks = Variable(masks).float().to(self.device)

                # loss_weight = get_loss_weight(masks)

                sqNet.train()
                sqNet.requires_grad_(True)
                pred_masks = sqNet(seq_vols)

                mask_lb = get_label_mask(masks, self.target_label).to(self.device)
                predmask_lb = get_label_mask(pred_masks, self.target_label).to(self.device)

                # weight, loss for lbs
                cnt_nodulePixel_lb = mask_lb.nonzero(as_tuple=False).size(0) + 1  # mask = 1 (nodule)
                cnt_allPixel_lb = mask_lb.numel()  # all mask pixel
                loss_weight_lb = cnt_allPixel_lb / cnt_nodulePixel_lb

                # for monitoring
                dice_coeff_lb = compute_dice_coeff_train(predmask_lb, mask_lb)

                # loss for each labels
                loss_lb = loss_weight_lb + criterion(predmask_lb, mask_lb, loss_weight_lb)

                sqNet.zero_grad()
                optimizer.zero_grad()

                loss_lb.backward()

                optimizer.step()
                sqNet.requires_grad_(False)

                if total_step % loss_print_step == 0:
                    end_time_step = time.time()
                    print('[%d / %d]   time : %.2fs' % (epoch, self.epochs, end_time_step - start_time_step))

                    self.writer.add_scalar(f'train/step_dice_coeff_lb{self.target_label}', dice_coeff_lb.item(), total_step)
                    self.writer.add_scalar(f'train/step_dice_loss_lb{self.target_label}', loss_lb.item(), total_step)

                    if total_step % result_print_step == 0:
                        with torch.no_grad():
                            seq_vols = seq_vols.cpu().numpy()
                            seq_vols = np.transpose(seq_vols, (0, 2, 3, 4, 1))
                            seq_vols = np.squeeze(seq_vols)

                            masks = masks.cpu().numpy()
                            masks = np.squeeze(masks)

                            pred_masks = pred_masks.cpu().numpy()
                            pred_masks = np.squeeze(pred_masks)

                            result_imgs = np.array([])

                            for train_result_idx, (gt, m, pred_m) in enumerate(zip(seq_vols, masks, pred_masks)):
                                pred_m = np.where(pred_m > 0.5, 1, 0)
                                ## rgb color filters

                                b_gtmask_lb = color_rgb_filter('gt', 1)
                                b_predmask_lb = color_rgb_filter('pred', 1)


                                # multiply rgb filter
                                gt = cv2.cvtColor(gt[:, :, 1] * 255, cv2.COLOR_GRAY2RGB)
                                b_gtmask_lb = mul_rgb(b_gtmask_lb, m[0, :, :])
                                b_predmask_lb = mul_rgb(b_predmask_lb, pred_m)

                                # save result images
                                temp_result_gtmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                     b_gtmask_lb.astype(np.uint8), 0.5, 0)

                                temp_result_predmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                       b_predmask_lb.astype(np.uint8), 0.5, 0)

                                result_img = np.concatenate((gt, temp_result_gtmask, temp_result_predmask), 1)

                                if train_result_idx == 0:
                                    result_imgs = result_img
                                else:
                                    result_imgs = np.concatenate((result_imgs, result_img), 0)

                                if train_result_idx == 6:
                                    break
                            cv2.imwrite(self.train_result_dir + '/train_epoch_' + str(epoch) + '_step' + str(
                                total_step) + '.png', result_imgs)

                    start_time_step = time.time()
                total_step += 1

            end_time_epoch = time.time()
            print('[%d / %d - %d step] training time : %.5fs ' % (
            epoch, self.epochs, total_step, end_time_epoch - start_time_epoch))

            ##########################################################
            '''
            Testing Start
            '''
            ##########################################################
            ### test
            sqNet.eval()
            sqNet.requires_grad_(False)

            with torch.no_grad():


                total_dice_coeff_lb = []
                total_w_dice_coeff_lb = []
                total_seg_loss_lb = []

                total_vs_lb = []

                total_num_test_batch = []

                for test_idx, data in enumerate(self.test_dataloader):
                    test_seq_vols, test_masks = data

                    test_seq_vols = Variable(test_seq_vols).float().to(self.device)
                    test_masks = Variable(test_masks).float().to(self.device)

                    test_pred_masks = sqNet(test_seq_vols)

                    test_mask_lb = get_label_mask(test_masks, self.target_label).to(self.device)
                    test_predmask_lb = get_label_mask(test_pred_masks, self.target_label).to(self.device)

                    loss_weight = get_loss_weight(test_masks)


                    # dice_coeff
                    dice_coeff_lb = compute_dice_coeff_test(test_predmask_lb, test_mask_lb)

                    # w_dice_coeff
                    w_dice_coeff_lb = compute_dice_coeff_train(test_predmask_lb, test_mask_lb)

                    # loss for each labels
                    seg_loss_lb = loss_weight_lb + criterion(test_predmask_lb, test_mask_lb, loss_weight_lb)

                    vs = compute_vs(test_pred_masks, test_masks)
                    vs_lb = compute_vs(test_predmask_lb, test_mask_lb)

                    #################################################
                    #################################################
                    batch_size = test_seq_vols.shape[0]
                    # total_dice_coeff.append(dice_coeff.item() * batch_size)
                    # total_w_dice_coeff.append(w_dice_coeff.item() * batch_size)

                    total_dice_coeff_lb.append(dice_coeff_lb.item() * batch_size)
                    total_w_dice_coeff_lb.append(w_dice_coeff_lb.item() * batch_size)

                    # total_seg_loss.append(seg_loss.item() * batch_size)
                    total_seg_loss_lb.append(seg_loss_lb.item() * batch_size)

                    total_vs_lb.append(vs_lb)

                    total_num_test_batch.append(batch_size)
                    #################################################
                    #################################################

                    if test_idx == 0 and epoch % test_img_export_interval == 0:
                        test_pred_masks = test_pred_masks.cpu().numpy()
                        test_pred_masks = np.squeeze(test_pred_masks)

                        test_masks = test_masks.cpu().numpy()
                        test_masks = np.squeeze(test_masks)

                        test_seq_vols = test_seq_vols.cpu().numpy()
                        test_seq_vols = np.transpose(test_seq_vols, (0, 2, 3, 4, 1))
                        test_seq_vols = np.squeeze(test_seq_vols)

                        result_imgs = np.array([])

                        for idx, (gt, m, pred_m) in enumerate(zip(test_seq_vols, test_masks, test_pred_masks)):
                            pred_m = np.where(pred_m > 0.5, 1, 0)
                            ## rgb color filters
                            b_gtmask_lb = color_rgb_filter('gt',1)
                            b_predmask_lb = color_rgb_filter('pred',1)


                            gt = cv2.cvtColor(gt[:, :, 1] * 255, cv2.COLOR_GRAY2RGB)
                            # multiply rgb filter
                            b_gtmask_lb = mul_rgb(b_gtmask_lb, m[0, :, :])
                            b_predmask_lb = mul_rgb(b_predmask_lb, pred_m)

                            # save result images
                            temp_result_gtmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                 b_gtmask_lb.astype(np.uint8), 0.5, 0)

                            temp_result_predmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                   b_predmask_lb.astype(np.uint8), 0.5, 0)

                            result_img = np.concatenate((gt, temp_result_gtmask, temp_result_predmask), 1)

                            if idx == 0:
                                result_imgs = result_img
                            else:
                                result_imgs = np.concatenate((result_imgs, result_img), 0)

                            if idx == 6:
                                break
                        cv2.imwrite(
                            self.test_result_dir + '/test_epoch_' + str(epoch) + '_step' + str(total_step) + '.png',
                            result_imgs)

                # calculate dice score for total, each labels
                total_dice_coeff_lb = sum(total_dice_coeff_lb) / sum(total_num_test_batch)

                # calculate wrong dice score for total, each labels
                total_w_dice_coeff_lb = sum(total_w_dice_coeff_lb) / sum(total_num_test_batch)

                # calculate segmantation loss for total, each labels
                total_seg_loss_lb = sum(total_seg_loss_lb) / sum(total_num_test_batch)

                # calculate volume similarity for total, each labels
                total_vs_lb = sum(total_vs_lb) / sum(total_num_test_batch)

                # add scalar to tensorboard

                self.writer.add_scalar(f'test/wrong dice coeff_label{self.target_label}', total_w_dice_coeff_lb, epoch)
                self.writer.add_scalar(f'test/segmentation loss_label{self.target_label}', total_seg_loss_lb, epoch)
                self.writer.add_scalar(f'test/volume similarity_label{self.target_label}', total_vs_lb, epoch)

                # for result (/n th_fold_/model/*)
                # if epoch == self.epochs - 50:
                if epoch == 0:
                    best_dice_score_lb = total_dice_coeff_lb
                    worst_dice_score_lb = total_dice_coeff_lb
                    av_dice_score_lb = total_dice_coeff_lb

                    best_volume_metric_lb = total_vs_lb
                    worst_volume_metric_lb = total_vs_lb
                    av_volume_metric_lb = total_vs_lb

                elif epoch > self.epochs - 50:

                    # save label1, best dice score
                    if best_dice_score_lb < total_dice_coeff_lb:
                        best_dice_score_lb = total_dice_coeff_lb
                        torch.save(sqNet.state_dict(), f'%s/sqNet_best_label{self.target_label}.pth' % (self.model_dir))

                    elif worst_dice_score_lb > total_dice_coeff_lb:
                        worst_dice_score_lb = total_dice_coeff_lb

                    if best_volume_metric_lb < total_vs_lb:
                        best_volume_metric_lb = total_vs_lb

                    elif worst_volume_metric_lb > total_vs_lb:
                        worst_volume_metric_lb = total_vs_lb

                    # av_dice_score += total_dice_coeff
                    av_dice_score_lb += total_dice_coeff_lb
                    # av_volume_metric += total_vs
                    av_volume_metric_lb += total_vs_lb

        # save final result
        torch.save(sqNet.state_dict(), '%s/sqNet_final.pth' % (self.model_dir))
        print("best_dice_score  : ", best_dice_score_lb)
        print("worst_dice_score  : ", worst_dice_score_lb)
        end_t = time.time()
        ## write txt file for summary (/n th_fold_/model/best_worst_dice_score.txt)
        f = open(self.model_dir + f'/best_worst_dice_score_{self.output_dir.split("/")[-3]}_{self.output_dir.split("/")[-2]}_{self.output_dir.split("/")[-1]}.txt', 'w')
        # summary
        f.write("Experiment_Name : " + self.output_dir.split("/")[-3] + "/" + self.output_dir.split("/")[-2] + "/" +
                self.output_dir.split("/")[-1] + "\n")
        f.write("label 1 : Globe \n")
        f.write("label 2 : Extraocular m. \n")
        f.write("label 3 : Optic n. \n")
        # best dice score
        # f.write("best_dice_score : %.5f \n" % (best_dice_score))
        f.write(f"best_dice_score : label{self.target_label} :: %.5f \n" % (best_dice_score_lb))

        # worst dice score
        f.write(f"worst_dice_score : label{self.target_label}  :: %.5f \n" % (worst_dice_score_lb))

        # average dice score
        f.write(f'Average_dice_score : label{self.target_label}  :: %.5f \n' % (av_dice_score_lb / 50))

        # best Volumetric Similarity
        f.write(f"best_Volumetric_Similarity : label{self.target_label}  :: %.5f \n" % (best_volume_metric_lb))

        # worst Volumetric Similarity
        f.write(f"worst_Volumetric_Similarity : label{self.target_label}  :: %.5f \n" % (worst_volume_metric_lb))

        # average Volumetric Similarity
        f.write(f'Average_Volumetric_Similarity : label1 :: %.5f \n' % (av_volume_metric_lb / 50))

        # params, training time
        f.write('Total Param : %i \n' % (total_param))
        f.write('Train Param : %i \n' % (train_param))
        f.write('Total Training Time : %.5f \n' % (end_t - start_t))
        f.close()

    # flattening
    def flatten_outputs(self, fea):
        return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2], fea.shape[3] * fea.shape[4]))

    def extractor_att_fea_map(self, fm_src, fm_tgt):
        fea_loss = torch.tensor(0.).to(self.device)

        b, s, c, h, w = fm_src.shape
        fm_src = self.flatten_outputs(fm_src)
        fm_tgt = self.flatten_outputs(fm_tgt)

        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 3)
        distance = distance ** 2 / (h * w)
        fea_loss += torch.sum(distance)
        return fea_loss