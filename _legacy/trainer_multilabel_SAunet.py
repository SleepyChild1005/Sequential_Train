import os
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import cv2

from model.loss_multilabel import DiceLoss
# used in mainYL

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)


def compute_dice_coeff_train(pred, gt, smooth=1):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = (pred * gt).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    return dice


def compute_dice_coeff_test(pred, gt, smooth=1):
    pred = (pred > 0.5).float()
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = (pred * gt).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    return dice

# all together
def compute_vs(pred, gt):
    pred = (pred > 0.5).float()
    single_vs = []
    for batch in range(pred.shape[0]):
        confusion_vector = pred[batch, :, :, :] / gt[batch, :, :, :]

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        vs = 1 - abs(false_negatives - false_positives) / (
                    2 * true_positives + false_positives + false_negatives + 1e-4)
        single_vs.append(vs)

    total_sum_vs = sum(single_vs)
    return total_sum_vs

# single channel
def compute_vs_each(pred, gt):
    pred = (pred > 0.5).float()
    single_vs = []
    for batch in range(pred.shape[0]):
        confusion_vector = pred[batch, :, :] / gt[batch, :, :]

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        vs = 1 - abs(false_negatives - false_positives) / (
                    2 * true_positives + false_positives + false_negatives + 1e-4)
        single_vs.append(vs)

    total_sum_vs = sum(single_vs)
    return total_sum_vs
# for multiplying rgb np array
def mul_rgb(rgb_mask,m):
    _mask = np.zeros([64,64,3])
    _mask[:,:,:] = rgb_mask[:,:,:]
    for i in range(3):
        _mask[:,:,i] = _mask[:,:,i] * m
    return _mask



class sequentialSegTrainer(object):
    def __init__(self, epochs, gpu, batch_size, image_size, learning_rate, output_dir, bAttention, bTransfer_learning, pretrained_model_dir, train_dataloader, test_dataloader, writer):
        self.model_dir = os.path.join(output_dir, 'model')
        self.train_result_dir = os.path.join(output_dir, 'result', 'train')
        self.test_result_dir = os.path.join(output_dir, 'result', 'test')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.train_result_dir)
            os.makedirs(self.test_result_dir)

        self.epochs = epochs
        self.device = torch.device("cuda:%s" % gpu)
        self.gpu = gpu
        self.batch_size = batch_size
        self.image_size = image_size

        self.learning_rate = learning_rate

        self.output_dir = output_dir
        self.bAttention = bAttention
        self.bTransfer_learning = bTransfer_learning
        self.pretrained_model_dir = pretrained_model_dir

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.writer = writer
    
    def fix_attention_parameters(self, model):
        for param in model.Attention3.parameters():
            param.requires_grad = False
        for param in model.Attention2.parameters():
            param.requires_grad = False
        for param in model.Attention1.parameters():
            param.requires_grad = False
        
    
    def train(self):
        ## 3 modes [ sensor3D Unet, Attention c sensor3D Unet, Domain Adaptation c Attention c sensor3D Unet ]
        # Attention
        if not(self.bAttention) and not(self.bTransfer_learning):
            from model.Torch_SA_Unet_OpthalCT import SA_Unet_Torch
            # print('gpu', self.gpu)
            # print('gpu', str(self.gpu))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            sqNet = SA_Unet_Torch(self.image_size, self.device).to(self.device)
            sqNet.apply(weights_init)
            print("###### Sensor3D + SA Unet Model ######")
        # # just SA Unet Model
        # elif not(self.bAttention) and not(self.bTransfer_learning):
        #     from model.sensor3d_model_multilabel import DeepSequentialNet
        #     sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
        #     sqNet.apply(weights_init)
        #     print("###### Sensor3D model ######")
        # # Domain Adaptation, Attention
        # elif self.bAttention and self.bTransfer_learning:
        #     from model.attention_model import DeepSequentialNet
        #     sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
        #     sqNet.load_state_dict(torch.load(self.pretrained_model_dir, map_location=self.device))
        #     print("###### Sensor3D + Attention model + transfer learning ######")

            
        total_param = sum(p.numel() for p in sqNet.parameters())
        train_param = sum(p.numel() for p in sqNet.parameters() if p.requires_grad)
        
        optimizer = optim.Adam(sqNet.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        criterion = DiceLoss().to(self.device)

        start_t = time.time()

        # variables for each label, total
        total_step = 0
        result_print_step=10
        loss_print_step = 10
        start_time_step = time.time()
        test_img_export_interval = 5
        
        best_dice_score = 0
        best_dice_score_lb1 = 0
        best_dice_score_lb2 = 0
        best_dice_score_lb3 = 0

        worst_dice_score = 0
        worst_dice_score_lb1 = 0
        worst_dice_score_lb2 = 0
        worst_dice_score_lb3 = 0

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
                # print('seq_vols',seq_vols.shape)
                # print('masks',masks.shape)
                # sys.exit("Debugging")
                seq_vols = Variable(seq_vols).float().to(self.device)
                masks = Variable(masks).float().to(self.device)

                cnt_nodulePixel = masks.nonzero(as_tuple=False).size(0) + 1          # mask = 1 (nodule)
                cnt_allPixel = masks.numel()                           # all mask pixel
                loss_weight = cnt_allPixel/cnt_nodulePixel
                # print('masks',masks.shape)
                # seq_vols = torch.squeeze(seq_vols, dim=2)

                sqNet.train()
                sqNet.requires_grad_(True)
                pred_masks = sqNet(seq_vols)

                mask_lb1 = masks[:, 0, :, :].to(self.device)
                mask_lb2 = masks[:, 1, :, :].to(self.device)
                mask_lb3 = masks[:, 2, :, :].to(self.device)

                predmask_lb1 = pred_masks[:, 0, :, :].to(self.device)
                predmask_lb2 = pred_masks[:, 1, :, :].to(self.device)
                predmask_lb3 = pred_masks[:, 2, :, :].to(self.device)

                # weight, loss for lbs
                cnt_nodulePixel_lb1 = mask_lb1.nonzero(as_tuple=False).size(0) + 1  # mask = 1 (nodule)
                cnt_allPixel_lb1 = mask_lb1.numel()  # all mask pixel
                loss_weight_lb1 = cnt_allPixel_lb1 / cnt_nodulePixel_lb1

                cnt_nodulePixel_lb2 = mask_lb2.nonzero(as_tuple=False).size(0) + 1  # mask = 1 (nodule)
                cnt_allPixel_lb2 = mask_lb2.numel()  # all mask pixel
                loss_weight_lb2 = cnt_allPixel_lb2 / cnt_nodulePixel_lb2

                cnt_nodulePixel_lb3 = mask_lb3.nonzero(as_tuple=False).size(0) + 1  # mask = 1 (nodule)
                cnt_allPixel_lb3 = mask_lb3.numel()  # all mask pixel
                loss_weight_lb3 = cnt_allPixel_lb3 / cnt_nodulePixel_lb3

                # for monitoring
                dice_coeff_lb1 = compute_dice_coeff_train(predmask_lb1, mask_lb1)
                dice_coeff_lb2 = compute_dice_coeff_train(predmask_lb2, mask_lb2)
                dice_coeff_lb3 = compute_dice_coeff_train(predmask_lb3, mask_lb3)

                #loss for each labels
                loss_lb1 = loss_weight_lb1 + criterion(predmask_lb1, mask_lb1, loss_weight_lb1)
                loss_lb2 = loss_weight_lb2 + criterion(predmask_lb2, mask_lb2, loss_weight_lb2)
                loss_lb3 = loss_weight_lb3 + criterion(predmask_lb3, mask_lb3, loss_weight_lb3)
                
                sqNet.zero_grad()
                loss_total = loss_lb1 + loss_lb2 +loss_lb3
                loss_total.backward()
                optimizer.step()
                sqNet.requires_grad_(False)

                if total_step % loss_print_step == 0:
                    end_time_step = time.time() 
                    print('[%d / %d]   time : %.2fs' % (epoch, self.epochs, end_time_step - start_time_step))     
                    
                    self.writer.add_scalar('train/step_dice_coeff_lb1', dice_coeff_lb1.item(), total_step)
                    self.writer.add_scalar('train/step_dice_coeff_lb2', dice_coeff_lb2.item(), total_step)
                    self.writer.add_scalar('train/step_dice_coeff_lb3', dice_coeff_lb3.item(), total_step)
                    self.writer.add_scalar('train/step_dice_loss_lb1', loss_lb1.item(), total_step)
                    self.writer.add_scalar('train/step_dice_loss_lb2', loss_lb2.item(), total_step)
                    self.writer.add_scalar('train/step_dice_loss_lb3', loss_lb3.item(), total_step)

                    if total_step % result_print_step == 0:
                        with torch.no_grad():
                            seq_vols = seq_vols.cpu().numpy()
                            # seq_vols = np.transpose(seq_vols, (0, 2, 3, 4, 1))
                            seq_vols = np.squeeze(seq_vols)
                            # print('seq_vols',seq_vols.shape)

                            masks = masks.cpu().numpy()
                            masks = np.squeeze(masks)
                            # print('masks',masks.shape)

                            pred_masks = pred_masks.cpu().numpy()
                            pred_masks = np.squeeze(pred_masks)
                            # print('pred_masks',pred_masks.shape)

                            # = zip(seq_vols, masks, pred_masks)
                            # print('zip',.shape)
                            result_imgs = np.array([])

                            for train_result_idx, (gt, m, pred_m) in enumerate(np.stack([seq_vols, masks, pred_masks],axis=1)):
                                # print('idx',train_result_idx)
                                # print('gt', gt.shape)
                                # print('m', m.shape)
                                # print('pred_m', pred_m.shape)
                                pred_m = np.where(pred_m>0.5, 1, 0)
                                ## rgb color filters
                                # gt lb1 color rgb 128, 24, 185
                                b_gtmask_lb1 = np.zeros([64, 64, 3])
                                b_gtmask_lb1[:,:,0] = 128
                                b_gtmask_lb1[:,:,1] = 24
                                b_gtmask_lb1[:,:,2] = 185
                                # gt lb2 color rgb 19, 114, 87
                                b_gtmask_lb2 = np.zeros([64, 64, 3])
                                b_gtmask_lb2[:,:,0] = 19
                                b_gtmask_lb2[:,:,1] = 114
                                b_gtmask_lb2[:,:,2] = 87
                                # gt lb3 color rgb 13, 101, 177
                                b_gtmask_lb3 = np.zeros([64, 64, 3])
                                b_gtmask_lb3[:,:,0] = 13
                                b_gtmask_lb3[:,:,1] = 101
                                b_gtmask_lb3[:,:,2] = 177

                                # pred lb1 color rgb 227, 39, 186
                                b_predmask_lb1 = np.zeros([64, 64, 3])
                                b_predmask_lb1[:,:,0] = 226
                                b_predmask_lb1[:,:,1] = 39
                                b_predmask_lb1[:,:,2] = 186
                                # pred lb2 color rgb 34, 211, 18
                                b_predmask_lb2 = np.zeros([64, 64, 3])
                                b_predmask_lb2[:,:,0] = 34
                                b_predmask_lb2[:,:,1] = 211
                                b_predmask_lb2[:,:,2] = 18
                                # pred lb3 color rgb 24, 184, 196
                                b_predmask_lb3 = np.zeros([64, 64, 3])
                                b_predmask_lb3[:,:,0] = 24
                                b_predmask_lb3[:,:,1] = 184
                                b_predmask_lb3[:,:,2] = 196

                                # multiply rgb filter
                                gt = cv2.cvtColor(gt[1,:,:]*255, cv2.COLOR_GRAY2RGB)

                                b_gtmask_lb1 = mul_rgb(b_gtmask_lb1, m[0, :, :])
                                b_gtmask_lb2 = mul_rgb(b_gtmask_lb2, m[1, :, :])
                                b_gtmask_lb3 = mul_rgb(b_gtmask_lb3, m[2, :, :])

                                b_predmask_lb1 = mul_rgb(b_predmask_lb1, pred_m[0, :, :])
                                b_predmask_lb2 = mul_rgb(b_predmask_lb2, pred_m[1, :, :])
                                b_predmask_lb3 = mul_rgb(b_predmask_lb3, pred_m[2, :, :])
                                # save result images
                                # print('gt', gt.shape)
                                temp_result_gtmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                         b_gtmask_lb1.astype(np.uint8), 0.5, 0)
                                temp_result_gtmask = cv2.addWeighted(temp_result_gtmask, 0.7,
                                                                         b_gtmask_lb2.astype(np.uint8), 0.5, 0)
                                temp_result_gtmask = cv2.addWeighted(temp_result_gtmask, 0.7,
                                                                         b_gtmask_lb3.astype(np.uint8), 0.5, 0)

                                temp_result_predmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                           b_predmask_lb1.astype(np.uint8), 0.5, 0)
                                temp_result_predmask = cv2.addWeighted(temp_result_predmask, 0.7,
                                                                           b_predmask_lb2.astype(np.uint8), 0.5, 0)
                                temp_result_predmask = cv2.addWeighted(temp_result_predmask, 0.7,
                                                                           b_predmask_lb3.astype(np.uint8), 0.5, 0)

                                result_img = np.concatenate((gt, temp_result_gtmask, temp_result_predmask), 1)

                                if train_result_idx == 0:
                                    result_imgs = result_img
                                else:
                                    result_imgs = np.concatenate((result_imgs, result_img), 0)
                                
                                if train_result_idx == 6:
                                    break
                            cv2.imwrite(self.train_result_dir + '/train_epoch_' + str(epoch) + '_step' + str(total_step) + '.png', result_imgs)
                
                    start_time_step = time.time()    
                total_step += 1

            end_time_epoch = time.time()
            print('[%d / %d - %d step] training time : %.5fs ' % (epoch, self.epochs, total_step, end_time_epoch-start_time_epoch))

            ##########################################################
            ##########################################################
            ### test
            sqNet.eval()
            sqNet.requires_grad_(False)

            with torch.no_grad():
                total_dice_coeff = []
                total_w_dice_coeff = []

                total_dice_coeff_lb1 = []
                total_w_dice_coeff_lb1 = []

                total_dice_coeff_lb2 = []
                total_w_dice_coeff_lb2 = []

                total_dice_coeff_lb3 = []
                total_w_dice_coeff_lb3 = []

                total_seg_loss = []
                total_seg_loss_lb1 = []
                total_seg_loss_lb2 = []
                total_seg_loss_lb3 = []

                total_vs = []
                total_vs_lb1 = []
                total_vs_lb2 = []
                total_vs_lb3 = []

                total_num_test_batch = []
                
                for test_idx, data in enumerate(self.test_dataloader):
                    test_seq_vols, test_masks = data
                    # print('test_idx',test_idx)
                    test_seq_vols = Variable(test_seq_vols).float().to(self.device)
                    test_masks = Variable(test_masks).float().to(self.device)

                    # test_seq_vols = torch.squeeze(test_seq_vols,dim=2)
                    # print('test_seq_vols',test_seq_vols.shape)

                    test_pred_masks = sqNet(test_seq_vols)

                    test_mask_lb1 = test_masks[:, 0, :, :].to(self.device)
                    test_mask_lb2 = test_masks[:, 1, :, :].to(self.device)
                    test_mask_lb3 = test_masks[:, 2, :, :].to(self.device)

                    test_predmask_lb1 = test_pred_masks[:, 0, :, :].to(self.device)
                    test_predmask_lb2 = test_pred_masks[:, 1, :, :].to(self.device)
                    test_predmask_lb3 = test_pred_masks[:, 2, :, :].to(self.device)

                    cnt_nodulePixel = test_masks.nonzero(as_tuple=False).size(0) + 1          # mask = 1 (nodule)
                    cnt_allPixel = test_masks.numel()                           # all mask pixel
                    loss_weight = cnt_allPixel/cnt_nodulePixel
                    w_dice_coeff = compute_dice_coeff_train(test_pred_masks, test_masks)
                    dice_coeff = compute_dice_coeff_test(test_pred_masks, test_masks)                     
                    seg_loss = loss_weight + criterion(test_pred_masks, test_masks, loss_weight)

                    # dice_coeff
                    dice_coeff_lb1 = compute_dice_coeff_test(test_predmask_lb1, test_mask_lb1)
                    dice_coeff_lb2 = compute_dice_coeff_test(test_predmask_lb2, test_mask_lb2)
                    dice_coeff_lb3 = compute_dice_coeff_test(test_predmask_lb3, test_mask_lb3)

                    # w_dice_coeff
                    w_dice_coeff_lb1 = compute_dice_coeff_train(test_predmask_lb1, test_mask_lb1)
                    w_dice_coeff_lb2 = compute_dice_coeff_train(test_predmask_lb2, test_mask_lb2)
                    w_dice_coeff_lb3 = compute_dice_coeff_train(test_predmask_lb3, test_mask_lb3)


                    # loss for each labels
                    seg_loss_lb1 = loss_weight_lb1 + criterion(test_predmask_lb1, test_mask_lb1, loss_weight_lb1)
                    seg_loss_lb2 = loss_weight_lb2 + criterion(test_predmask_lb2, test_mask_lb2, loss_weight_lb2)
                    seg_loss_lb3 = loss_weight_lb3 + criterion(test_predmask_lb3, test_mask_lb3, loss_weight_lb3)

                    # print('test_pred_masks',test_pred_masks.shape)
                    # print('test_masks',test_masks.shape)
                    vs = compute_vs(test_pred_masks, test_masks)
                    vs_lb1 = compute_vs_each(test_predmask_lb1, test_mask_lb1)
                    vs_lb2 = compute_vs_each(test_predmask_lb2, test_mask_lb2)
                    vs_lb3 = compute_vs_each(test_predmask_lb3, test_mask_lb3)
                    

                    #################################################
                    #################################################
                    batch_size = test_seq_vols.shape[0]
                    total_dice_coeff.append(dice_coeff.item() * batch_size)
                    total_w_dice_coeff.append(w_dice_coeff.item() * batch_size)

                    total_dice_coeff_lb1.append(dice_coeff_lb1.item() * batch_size)
                    total_dice_coeff_lb2.append(dice_coeff_lb2.item() * batch_size)
                    total_dice_coeff_lb3.append(dice_coeff_lb3.item() * batch_size)

                    total_w_dice_coeff_lb1.append(w_dice_coeff_lb1.item() * batch_size)
                    total_w_dice_coeff_lb2.append(w_dice_coeff_lb2.item() * batch_size)
                    total_w_dice_coeff_lb3.append(w_dice_coeff_lb3.item() * batch_size)

                    total_seg_loss.append(seg_loss.item() * batch_size)
                    total_seg_loss_lb1.append(seg_loss_lb1.item() * batch_size)
                    total_seg_loss_lb2.append(seg_loss_lb2.item() * batch_size)
                    total_seg_loss_lb3.append(seg_loss_lb3.item() * batch_size)

                    total_vs.append(vs)
                    total_vs_lb1.append(vs_lb1)
                    total_vs_lb2.append(vs_lb2)
                    total_vs_lb3.append(vs_lb3)

                    total_num_test_batch.append(batch_size)
                    #################################################
                    #################################################

                    if test_idx == 0 and epoch%test_img_export_interval == 0:
                        test_pred_masks = test_pred_masks.cpu().numpy()
                        test_pred_masks = np.squeeze(test_pred_masks)
                        # print('test_pred_masks', test_pred_masks.shape)
                        test_masks = test_masks.cpu().numpy()
                        test_masks = np.squeeze(test_masks)
                        # print('test_masks', test_masks.shape)

                        test_seq_vols = test_seq_vols.cpu().numpy()
                        # test_seq_vols = np.transpose(test_seq_vols, (0, 2, 3, 4, 1))
                        test_seq_vols = np.squeeze(test_seq_vols)
                        # print('test_seq_vols', test_seq_vols.shape)
                        result_imgs = np.array([])

                        for idx, (gt, m, pred_m) in enumerate(np.stack([test_seq_vols, test_masks, test_pred_masks], axis=1)):
                            pred_m = np.where(pred_m>0.5, 1, 0)
                            ## rgb color filters
                            # gt lb1 color rgb 128, 24, 185
                            b_gtmask_lb1 = np.zeros([64, 64, 3])
                            b_gtmask_lb1[:, :, 0] = 128
                            b_gtmask_lb1[:, :, 1] = 24
                            b_gtmask_lb1[:, :, 2] = 185
                            # gt lb2 color rgb 19, 114, 87
                            b_gtmask_lb2 = np.zeros([64, 64, 3])
                            b_gtmask_lb2[:, :, 0] = 19
                            b_gtmask_lb2[:, :, 1] = 114
                            b_gtmask_lb2[:, :, 2] = 87
                            # gt lb3 color rgb 13, 101, 177
                            b_gtmask_lb3 = np.zeros([64, 64, 3])
                            b_gtmask_lb3[:, :, 0] = 13
                            b_gtmask_lb3[:, :, 1] = 101
                            b_gtmask_lb3[:, :, 2] = 177

                            # pred lb1 color rgb 227, 39, 186
                            b_predmask_lb1 = np.zeros([64, 64, 3])
                            b_predmask_lb1[:, :, 0] = 226
                            b_predmask_lb1[:, :, 1] = 39
                            b_predmask_lb1[:, :, 2] = 186
                            # pred lb2 color rgb 34, 211, 18
                            b_predmask_lb2 = np.zeros([64, 64, 3])
                            b_predmask_lb2[:, :, 0] = 34
                            b_predmask_lb2[:, :, 1] = 211
                            b_predmask_lb2[:, :, 2] = 18
                            # pred lb3 color rgb 24, 184, 196
                            b_predmask_lb3 = np.zeros([64, 64, 3])
                            b_predmask_lb3[:, :, 0] = 24
                            b_predmask_lb3[:, :, 1] = 184
                            b_predmask_lb3[:, :, 2] = 196

                            gt = cv2.cvtColor(gt[1,:,:] * 255, cv2.COLOR_GRAY2RGB)
                            # multiply rgb filter
                            b_gtmask_lb1 = mul_rgb(b_gtmask_lb1, m[0, :, :])
                            b_gtmask_lb2 = mul_rgb(b_gtmask_lb2, m[1, :, :])
                            b_gtmask_lb3 = mul_rgb(b_gtmask_lb3, m[2, :, :])

                            b_predmask_lb1 = mul_rgb(b_predmask_lb1, pred_m[0, :, :])
                            b_predmask_lb2 = mul_rgb(b_predmask_lb2, pred_m[1, :, :])
                            b_predmask_lb3 = mul_rgb(b_predmask_lb3, pred_m[2, :, :])
                            # save result images
                            temp_result_gtmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                 b_gtmask_lb1.astype(np.uint8), 0.5, 0)
                            temp_result_gtmask = cv2.addWeighted(temp_result_gtmask, 0.7,
                                                                 b_gtmask_lb2.astype(np.uint8), 0.5, 0)
                            temp_result_gtmask = cv2.addWeighted(temp_result_gtmask, 0.7,
                                                                 b_gtmask_lb3.astype(np.uint8), 0.5, 0)

                            temp_result_predmask = cv2.addWeighted(gt.astype(np.uint8), 0.7,
                                                                   b_predmask_lb1.astype(np.uint8), 0.5, 0)
                            temp_result_predmask = cv2.addWeighted(temp_result_predmask, 0.7,
                                                                   b_predmask_lb2.astype(np.uint8), 0.5, 0)
                            temp_result_predmask = cv2.addWeighted(temp_result_predmask, 0.7,
                                                                   b_predmask_lb3.astype(np.uint8), 0.5, 0)

                            result_img = np.concatenate((gt, temp_result_gtmask, temp_result_predmask), 1)

                            if idx == 0:
                                result_imgs = result_img
                            else:
                                result_imgs = np.concatenate((result_imgs, result_img), 0)

                            if idx == 6:
                                break
                        cv2.imwrite(self.test_result_dir + '/test_epoch_' + str(epoch) + '_step' + str(total_step) + '.png', result_imgs)
                
                # calculate dice score for total, each labels
                total_dice_coeff = sum(total_dice_coeff) / sum(total_num_test_batch)
                total_dice_coeff_lb1 = sum(total_dice_coeff_lb1) / sum(total_num_test_batch)
                total_dice_coeff_lb2 = sum(total_dice_coeff_lb2) / sum(total_num_test_batch)
                total_dice_coeff_lb3 = sum(total_dice_coeff_lb3) / sum(total_num_test_batch)
                # calculate wrong dice score for total, each labels
                total_w_dice_coeff = sum(total_w_dice_coeff) / sum(total_num_test_batch)
                total_w_dice_coeff_lb1 = sum(total_w_dice_coeff_lb1) / sum(total_num_test_batch)
                total_w_dice_coeff_lb2 = sum(total_w_dice_coeff_lb2) / sum(total_num_test_batch)
                total_w_dice_coeff_lb3 = sum(total_w_dice_coeff_lb3) / sum(total_num_test_batch)
                # calculate segmantation loss for total, each labels
                total_seg_loss = sum(total_seg_loss) / sum(total_num_test_batch)
                total_seg_loss_lb1 = sum(total_seg_loss_lb1) / sum(total_num_test_batch)
                total_seg_loss_lb2 = sum(total_seg_loss_lb2) / sum(total_num_test_batch)
                total_seg_loss_lb3 = sum(total_seg_loss_lb3) / sum(total_num_test_batch)
                # calculate volume similarity for total, each labels
                total_vs = sum(total_vs) / sum(total_num_test_batch)
                total_vs_lb1 = sum(total_vs_lb1) / sum(total_num_test_batch)
                total_vs_lb2 = sum(total_vs_lb2) / sum(total_num_test_batch)
                total_vs_lb3 = sum(total_vs_lb3) / sum(total_num_test_batch)

                # add scalar to tensorboard
                self.writer.add_scalar('test/dice coeff', total_dice_coeff, epoch)
                self.writer.add_scalar('test/dice coeff_label1', total_dice_coeff_lb1, epoch)
                self.writer.add_scalar('test/dice coeff_label2', total_dice_coeff_lb2, epoch)
                self.writer.add_scalar('test/dice coeff_label3', total_dice_coeff_lb3, epoch)

                self.writer.add_scalar('test/wrong dice coeff', total_w_dice_coeff, epoch)
                self.writer.add_scalar('test/wrong dice coeff_label1', total_w_dice_coeff_lb1, epoch)
                self.writer.add_scalar('test/wrong dice coeff_label2', total_w_dice_coeff_lb2, epoch)
                self.writer.add_scalar('test/wrong dice coeff_label3', total_w_dice_coeff_lb3, epoch)

                self.writer.add_scalar('test/segmentation loss', total_seg_loss, epoch)
                self.writer.add_scalar('test/segmentation loss_label1', total_seg_loss_lb1, epoch)
                self.writer.add_scalar('test/segmentation loss_label2', total_seg_loss_lb2, epoch)
                self.writer.add_scalar('test/segmentation loss_label3', total_seg_loss_lb3, epoch)
                
                self.writer.add_scalar('test/volume similarity', total_vs, epoch)
                self.writer.add_scalar('test/volume similarity_label1', total_vs_lb1, epoch)
                self.writer.add_scalar('test/volume similarity_label2', total_vs_lb2, epoch)
                self.writer.add_scalar('test/volume similarity_label3', total_vs_lb3, epoch)
                # for result (/n th_fold_/model/*)
                if epoch == self.epochs - 50:
                    best_dice_score = total_dice_coeff
                    worst_dice_score = total_dice_coeff
                    av_dice_score = total_dice_coeff

                    best_dice_score_lb1 = total_dice_coeff_lb1
                    worst_dice_score_lb1 = total_dice_coeff_lb1
                    av_dice_score_lb1 = total_dice_coeff_lb1

                    best_dice_score_lb2 = total_dice_coeff_lb2
                    worst_dice_score_lb2 = total_dice_coeff_lb2
                    av_dice_score_lb2 = total_dice_coeff_lb2

                    best_dice_score_lb3 = total_dice_coeff_lb3
                    worst_dice_score_lb3 = total_dice_coeff_lb3
                    av_dice_score_lb3 = total_dice_coeff_lb3

                    best_volume_metric = total_vs
                    worst_volume_metric = total_vs
                    av_volume_metric = total_vs

                    best_volume_metric_lb1 = total_vs_lb1
                    worst_volume_metric_lb1 = total_vs_lb1
                    av_volume_metric_lb1 = total_vs_lb1

                    best_volume_metric_lb2 = total_vs_lb2
                    worst_volume_metric_lb2 = total_vs_lb2
                    av_volume_metric_lb2 = total_vs_lb2

                    best_volume_metric_lb3 = total_vs_lb3
                    worst_volume_metric_lb3 = total_vs_lb3
                    av_volume_metric_lb3 = total_vs_lb3

                elif epoch > self.epochs - 50:
                    # save total, best dice score
                    if best_dice_score < total_dice_coeff:
                        best_dice_score = total_dice_coeff
                        torch.save(sqNet.state_dict(), '%s/sqNet_best.pth' % (self.model_dir))

                    elif worst_dice_score > total_dice_coeff:
                        worst_dice_score = total_dice_coeff

                    if best_volume_metric < total_vs:
                        best_volume_metric = total_vs

                    elif worst_volume_metric > total_vs:
                        worst_volume_metric = total_vs
                    # save label1, best dice score
                    if best_dice_score_lb1 < total_dice_coeff_lb1:
                        best_dice_score_lb1 = total_dice_coeff_lb1
                        torch.save(sqNet.state_dict(), '%s/sqNet_best_label1.pth' % (self.model_dir))

                    elif worst_dice_score_lb1 > total_dice_coeff_lb1:
                        worst_dice_score_lb1 = total_dice_coeff_lb1

                    if best_volume_metric_lb1 < total_vs_lb1:
                        best_volume_metric_lb1 = total_vs_lb1

                    elif worst_volume_metric_lb1 > total_vs_lb1:
                        worst_volume_metric_lb1 = total_vs_lb1
                    # save label2, best dice score
                    if best_dice_score_lb2 < total_dice_coeff_lb2:
                        best_dice_score_lb2 = total_dice_coeff_lb2
                        torch.save(sqNet.state_dict(), '%s/sqNet_best_label2.pth' % (self.model_dir))

                    elif worst_dice_score_lb2 > total_dice_coeff_lb2:
                        worst_dice_score_lb2 = total_dice_coeff_lb2

                    if best_volume_metric_lb2 < total_vs_lb2:
                        best_volume_metric_lb2 = total_vs_lb2

                    elif worst_volume_metric_lb2 > total_vs_lb2:
                        worst_volume_metric_lb2 = total_vs_lb2

                    # save label3, best dice score
                    if best_dice_score_lb3 < total_dice_coeff_lb3:
                        best_dice_score_lb3 = total_dice_coeff_lb3
                        torch.save(sqNet.state_dict(), '%s/sqNet_best_label3.pth' % (self.model_dir))

                    elif worst_dice_score_lb3 > total_dice_coeff_lb3:
                        worst_dice_score_lb3 = total_dice_coeff_lb3

                    if best_volume_metric_lb3 < total_vs_lb3:
                        best_volume_metric_lb3 = total_vs_lb3

                    elif worst_volume_metric_lb3 > total_vs_lb3:
                        worst_volume_metric_lb3 = total_vs_lb3

                    av_dice_score += total_dice_coeff
                    av_dice_score_lb1 += total_dice_coeff_lb1
                    av_dice_score_lb2 += total_dice_coeff_lb2
                    av_dice_score_lb3 += total_dice_coeff_lb3
                    av_volume_metric += total_vs
                    av_volume_metric_lb1 += total_vs_lb1
                    av_volume_metric_lb2 += total_vs_lb2
                    av_volume_metric_lb3 += total_vs_lb3
        # save final result
        torch.save(sqNet.state_dict(), '%s/sqNet_final.pth' % (self.model_dir))
        print("best_dice_score  : ", best_dice_score)
        print("worst_dice_score  : ", worst_dice_score)
        end_t = time.time()
        ## write txt file for summary (/n th_fold_/model/best_worst_dice_score.txt)
        f = open(self.model_dir + '/best_worst_dice_score.txt', 'w')
        # summary
        f.write("Experiment_Name : "+self.output_dir.split("/")[-3]+"/"+self.output_dir.split("/")[-2]+"/"+self.output_dir.split("/")[-1]+"\n")
        f.write("label 1 : Globe \n")
        f.write("label 2 : Extraocular m. \n")
        f.write("label 3 : Optic n. \n")
        # best dice score
        f.write("best_dice_score : %.5f \n" % (best_dice_score))
        f.write("best_dice_score : label1 :: %.5f \n" % (best_dice_score_lb1))
        f.write("best_dice_score : label2 :: %.5f \n" % (best_dice_score_lb2))
        f.write("best_dice_score : label3 :: %.5f \n" % (best_dice_score_lb3))
        # worst dice score
        f.write("worst_dice_score : %.5f \n" % (worst_dice_score))
        f.write("worst_dice_score : label1 :: %.5f \n" % (worst_dice_score_lb1))
        f.write("worst_dice_score : label2 :: %.5f \n" % (worst_dice_score_lb2))
        f.write("worst_dice_score : label3 :: %.5f \n" % (worst_dice_score_lb3))
        # average dice score
        f.write('Average_dice_score : %.5f \n' % (av_dice_score / 50))
        f.write('Average_dice_score : label1 :: %.5f \n' % (av_dice_score_lb1 / 50))
        f.write('Average_dice_score : label2 :: %.5f \n' % (av_dice_score_lb2 / 50))
        f.write('Average_dice_score : label3 :: %.5f \n' % (av_dice_score_lb3 / 50))
        # best Volumetric Similarity
        f.write("best_Volumetric_Similarity : %.5f \n" % (best_volume_metric))
        f.write("best_Volumetric_Similarity : label1 :: %.5f \n" % (best_volume_metric_lb1))
        f.write("best_Volumetric_Similarity : label2 :: %.5f \n" % (best_volume_metric_lb2))
        f.write("best_Volumetric_Similarity : label3 :: %.5f \n" % (best_volume_metric_lb3))
        # worst Volumetric Similarity
        f.write("worst_Volumetric_Similarity : %.5f \n" % (worst_volume_metric))
        f.write("worst_Volumetric_Similarity : label1 :: %.5f \n" % (worst_volume_metric_lb1))
        f.write("worst_Volumetric_Similarity : label2 :: %.5f \n" % (worst_volume_metric_lb2))
        f.write("worst_Volumetric_Similarity : label3 :: %.5f \n" % (worst_volume_metric_lb3))
        # average Volumetric Similarity
        f.write('Average_Volumetric_Similarity : %.5f \n' % (av_volume_metric / 50))
        f.write('Average_Volumetric_Similarity : label1 :: %.5f \n' % (av_volume_metric_lb1 / 50))
        f.write('Average_Volumetric_Similarity : label2 :: %.5f \n' % (av_volume_metric_lb2 / 50))
        f.write('Average_Volumetric_Similarity : label3 :: %.5f \n' % (av_volume_metric_lb3 / 50))
        # params, training time
        f.write('Total Param : %i \n' %(total_param))
        f.write('Train Param : %i \n' %(train_param))
        f.write('Total Training Time : %.5f \n' % ( end_t - start_t))
        f.close()
    # flattening
    def flatten_outputs(self, fea):
        return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2], fea.shape[3]*fea.shape[4]))
        
    def extractor_att_fea_map(self, fm_src, fm_tgt):
        fea_loss = torch.tensor(0.).to(self.device)
        
        b, s, c, h, w = fm_src.shape
        fm_src = self.flatten_outputs(fm_src)
        fm_tgt = self.flatten_outputs(fm_tgt)

        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 3)
        distance = distance ** 2 / (h * w)
        fea_loss += torch.sum(distance)
        return fea_loss      



