# Inspired by
# https://www.kaggle.com/coldfir3/efficient-coco-dataset-generator?scriptVersionId=79100851

# The comp encoding is rowise and every odd index represent the absolute
# begining of the mask. In the other hand, coco format expects it to be
# encoded by columns and the odd indexes are relative to the last end of the mask.

#
#     1. Decode rle (competition) to binary mask
#     2 .Encode the binary mask to rle (coco) using pycocotools
#     3 .Save to .json



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./sartorius-cell-instance-segmentation/train.csv')

def rle2mask(rle, img_w, img_h):

    '''Function decodes rle (for sartorius comp) to
    binary mask'''

    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1

    ## decompressing the rle encoding (ie, turning [3, 1, 10, 2] into [3, 4, 10, 11, 12])
    # for faster mask construction

    starts, lenghts = array
    mask_decompressed = np.concatenate([np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)])
    ## Building the binary mask
    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))
    msk_img = np.asfortranarray(msk_img)  ## This is important so pycocotools can handle this object

    return msk_img

