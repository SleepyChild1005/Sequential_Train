
import numpy as np

# for multiplying rgb np array
def mul_rgb(rgb_mask, m):
    _mask = np.zeros([64, 64, 3])
    _mask[:, :, :] = rgb_mask[:, :, :]
    for i in range(3):
        _mask[:, :, i] = _mask[:, :, i] * m
    return _mask

def create_RGB_filter(red:int,green:int,blue:int):
    rgb_filter = np.zeros([64, 64, 3])
    rgb_filter[:, :, 0] = red
    rgb_filter[:, :, 1] = green
    rgb_filter[:, :, 2] = blue
    return rgb_filter

def color_rgb_filter(type:str='gt',label:int=1):
    dict_rgb = {}
    rgb_gt = []
    rgb_gt.append((128, 24, 185))
    rgb_gt.append((19, 114, 87))
    rgb_gt.append((13, 101, 177))

    rgb_pred = []
    rgb_pred.append((227, 39, 186))
    rgb_pred.append((34, 221, 18))
    rgb_pred.append((24, 184, 196))

    dict_rgb['gt']=rgb_gt
    dict_rgb['pred']=rgb_pred

    (r,g,b) = dict_rgb[type][label - 1]
    rgb_filter = create_RGB_filter(r,g,b)
    return rgb_filter

