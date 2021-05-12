import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import os
from tqdm import tqdm

IMG_PATH = './results_Guided_BP'
LABEL_PATH = '../data/affordance/labels.json'

def get_foreground_mask(img_name):
    BLUR = 11
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 30
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,0.0) # In BGR format

    img = cv2.imread(img_name)

    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel) # this is the mask

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop
    x,y,w,h = cv2.boundingRect(cnt)
    mask1 = morphed #[y:y+h, x:x+w]
    mask1 = cv2.GaussianBlur(mask1, (BLUR, BLUR), 0)

    masked_img = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], mask1])
    return mask1

def grab_cut(img_name, mask_img_name, save_path):
    img = cv2.imread(img_name)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask_gray = cv2.imread(mask_img_name, 0)
    assert mask.shape == mask_gray.shape

    mask_fgd = get_foreground_mask(img_name)
    mask_fgd[mask_fgd <= 128] = 0
    mask_fgd[mask_fgd >= 128] = 1

    # cv.GC_BGD (0): obvious background pixels
    # cv.GC_FGD (1): obvious foreground pixels
    # cv.GC_PR_BGD (2): possible background pixels
    # cv.GC_PR_FGD (3): possible foreground pixels
    # mask[mask_gray >= 64] = cv.GC_PR_BGD
    mask[mask_gray >= 64] = cv2.GC_PR_FGD
    mask[mask_gray >= 128] = cv2.GC_FGD

    # bgdModel and fgdModel are arrays used internally by OpenCV
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    mask, _, _ = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 30, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    img = img * mask_fgd[:, :, np.newaxis]
    
    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    with open(LABEL_PATH, 'r') as fp:
        labels = json.load(fp)

    img_id_arr = [k for k in labels]

    hand_move_arr = ['clench', 'palm', 'pinch', 'poke']

    os.makedirs('./segmentation_results', exist_ok=True)
    for img_id in tqdm(img_id_arr[:30]):
        if os.path.isfile(os.path.join(IMG_PATH, img_id + '.png')):
            original_img_name = os.path.join(IMG_PATH, img_id + '.png')
            for hand_move in hand_move_arr:
                mask_img_name = os.path.join(IMG_PATH, f'{img_id}_Guided_BP_gray_{hand_move}.jpg')
                assert os.path.isfile(mask_img_name)

                grab_cut(original_img_name, mask_img_name,
                         f'./segmentation_results/{img_id}_{hand_move}.png')

            img = cv2.imread(original_img_name)
            cv2.imwrite(f'./segmentation_results/{img_id}.png', img)

    # img_id = '11r0o+qOxwL'
    # original_img_name = os.path.join(IMG_PATH, img_id + '.png')
    # mask_img_name = os.path.join(IMG_PATH, f'{img_id}_Guided_BP_gray_clench.jpg')

    # grab_cut(original_img_name, mask_img_name)
    # remove_background(original_img_name)