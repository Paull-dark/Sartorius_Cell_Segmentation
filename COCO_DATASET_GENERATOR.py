# Inspired by
# https://www.kaggle.com/coldfir3/efficient-coco-dataset-generator?scriptVersionId=79100851

# The comp encoding is rowise and every odd index represent the absolute
# begining of the mask. In the other hand, coco format expects it to be
# encoded by columns and the odd indexes are relative to the last end of the mask.

#
#     1. Decode rle (competition) to binary mask
#     2 .Encode the binary mask to rle (coco) using pycocotools
#     3.Optional (Clean broken masks)
#     4 .Save to .json


import numpy as np
import pandas as pd

import cv2
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi

from tqdm.notebook import tqdm
from pycocotools import mask as maskUtils
from joblib import Parallel, delayed
import json

from conf import TH, TRAIN_CSV

df = pd.read_csv(TRAIN_CSV)


def clean_mask(mask):
    '''
    Function is called to identify whether the mask is broken
    returns True or False state and also returns a mask
    '''
    mask = mask > threshold_otsu(np.array(mask).astype(np.uint8))
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    # New code for mask acceptance
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[0][:, 0]
    diff = c - np.roll(c, 1, 0)
    # find horizontal lines longer than threshold
    targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= TH)

    return mask, (True in targets)

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
    msk_img, broken_mask = clean_mask(msk_img)
    if broken_mask:
        broken_zero = np.zeros_like(msk_img)
        return np.asfortranarray(broken_zero)
    msk_img = np.asfortranarray(msk_img)  ## This is important so pycocotools can handle this object

    return msk_img

def build_mask(labels, input_shape, colors=False):
    height, width = input_shape
    masks = np.zeros((width,height))
    #masks = np.zeros((height,width), dtype=np.uint8)
    for i, label in enumerate(labels):
        a_mask = rle2mask(label, height,width)
        a_mask = np.array(a_mask) > 0
# #         a_mask, broken_mask = clean_mask(a_mask)
# #         if broken_mask:
# #             continue
        masks += a_mask
#     masks = masks.clip(0,1)
    return masks


def annotate(idx, row, cat_ids):
    '''
    Function is called to build json file
    '''

    # Binary mask
    mask = rle2mask(row['annotation'], row['width'], row['height'])
    # Encoding it back to rle (coco format)
    c_rle = maskUtils.encode(mask)
    # converting from binary to utf-8
    c_rle['counts'] = c_rle['counts'].decode('utf-8')
    # calculating the area
    area = maskUtils.area(c_rle).item()
    # calculating the bboxes
    bbox = maskUtils.toBbox(c_rle).astype(int).tolist()
    annotation = {
        'segmentation': c_rle,
        'bbox': bbox,
        'area': area,
        'image_id': row['id'],
        'category_id': cat_ids[row['cell_type']],
        'iscrowd': 0,
        'id': idx
    }
    return annotation


def coco_structure(df, workers=2):
    ## Building the header
    cat_ids = {name: id + 1 for id, name in enumerate(df.cell_type.unique())}
    cats = [{'name': name, 'id': id} for name, id in cat_ids.items()]
    images = [{'id': id, 'width': row.width, 'height': row.height, 'file_name': f'{id}.png'} for id, row in
              df.groupby('id').agg('first').iterrows()]

    ## Building the annotations
    annotations = Parallel(n_jobs=workers)(
        delayed(annotate)(idx, row, cat_ids) for idx, row in tqdm(df.iterrows(), total=len(df)))

    return {'categories': cats, 'images': images, 'annotations': annotations}


root = coco_structure(df)
with open('annotations_train.json', 'w', encoding='utf-8') as f:
    json.dump(root, f, ensure_ascii=True, indent=4)