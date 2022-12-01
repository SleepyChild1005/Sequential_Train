import torch


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