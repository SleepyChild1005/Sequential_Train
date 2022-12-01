import torch
import torch.nn as nn
# used in trainer

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def forward(self, pred, gt, weight, smooth=1.0):

        pred = torch.reshape(pred,(-1,))
        gt = torch.reshape(gt,(-1,))
        intersection = (pred * gt).sum()           
        dice = -1 * weight * (2.0*intersection + smooth)/(pred.sum() + gt.sum() + smooth)  
        
        return dice    

