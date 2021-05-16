import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import os
from tqdm import tqdm

ORIGINAL_IMG_PATH = './IIT_AFF_processed/rgb'
IMG_PATH = './pytorch-cnn-visualizations/results_IIT'
LABEL_PATH = './pytorch-cnn-visualizations/results_IIT/predictions.json'

def resize(img, target_w, target_h):
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    if target_w > target_h:
        w = 512
        h = 512/target_w*target_h
        h_start = int((512-h)/2)
        h_end = int(h_start + h)

        img = img[h_start:h_end, :]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        w = 512/target_h*target_w
        h = 512
        w_start = int((512-w)/2)
        w_end = int(w_start + w)

        img = img[:, w_start:w_end]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return img

def get_foreground_mask(img):
    BLUR = 11
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 30
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,0.0) # In BGR format

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

def grab_cut(img, mask_gray, avg_saliency_mask, std_saliency_mask, save_path):
    mask = np.zeros(img.shape[:2], np.uint8)
    std_saliency_mask += 1e-32
    mask_gray = (mask_gray - avg_saliency_mask)/std_saliency_mask
    mask_gray = cv2.blur(mask_gray, (20,20))

    assert mask.shape == mask_gray.shape

    mask_fgd = get_foreground_mask(img)
    mask_fgd[mask_fgd <= 128] = 0
    mask_fgd[mask_fgd >= 128] = 1

    # cv.GC_BGD (0): obvious background pixels
    # cv.GC_FGD (1): obvious foreground pixels
    # cv.GC_PR_BGD (2): possible background pixels
    # cv.GC_PR_FGD (3): possible foreground pixels
    # mask[mask_gray >= 64] = cv.GC_PR_BGD

    mask[mask_gray >= .3] = cv2.GC_PR_FGD
    mask[mask_gray >= .6] = cv2.GC_FGD

    # bgdModel and fgdModel are arrays used internally by OpenCV
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    if np.sum(mask) > 0:
        mask, _, _ = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 30, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = np.ones(img.shape[:2], np.uint8)
    img = img * mask
    img = img * mask_fgd

    np.save(open(save_path+'.npy', 'wb'), img)
    cv2.imwrite(save_path+'.jpg', img)

if __name__ == '__main__':
    with open(LABEL_PATH, 'r') as fp:
        labels = json.load(fp)

    img_id_arr = [k[:-4] for k in labels]

    hand_move_arr = ['clench', 'palm', 'pinch', 'poke']

    os.makedirs('./segmentation/segmentation_results_sub_mean_IIT', exist_ok=True)
    for img_id in tqdm(img_id_arr):
        if os.path.isfile(os.path.join(IMG_PATH, img_id + '.jpg')):
            original_img_name = os.path.join(ORIGINAL_IMG_PATH, img_id + '.jpg')
            original_h, original_w = cv2.imread(original_img_name).shape[:2]

            img_name = os.path.join(IMG_PATH, img_id + '.jpg')
            mask_img_names = [os.path.join(IMG_PATH, f'{img_id}_Guided_BP_gray_{hand_move}.jpg') for hand_move in hand_move_arr]

            #TODO
            # get original_w and original_h

            all_saliency_mask = [cv2.imread(mask_img_name, 0) for mask_img_name in mask_img_names]
            avg_saliency_mask = np.mean(all_saliency_mask, axis=0)
            std_saliency_mask = np.std(all_saliency_mask, axis=0)

            avg_saliency_mask = resize(avg_saliency_mask, original_w, original_h)
            std_saliency_mask = resize(std_saliency_mask, original_w, original_h)

            img = cv2.imread(img_name)
            img = resize(img, original_w, original_h)

            for hand_move in hand_move_arr:
                mask_img_name = os.path.join(IMG_PATH, f'{img_id}_Guided_BP_gray_{hand_move}.jpg')
                assert os.path.isfile(mask_img_name)
                mask_gray = cv2.imread(mask_img_name, 0)
                mask_gray = resize(mask_gray, original_w, original_h)

                grab_cut(img, mask_gray, avg_saliency_mask, std_saliency_mask,
                         f'./segmentation/segmentation_results_sub_mean_IIT/{img_id}_{hand_move}')

            cv2.imwrite(f'./segmentation/segmentation_results_sub_mean_IIT/{img_id}.jpg', img)

    # img_id = '11r0o+qOxwL'
    # original_img_name = os.path.join(IMG_PATH, img_id + '.png')
    # mask_img_name = os.path.join(IMG_PATH, f'{img_id}_Guided_BP_gray_clench.jpg')

    # grab_cut(original_img_name, mask_img_name)
    # remove_background(original_img_name)
