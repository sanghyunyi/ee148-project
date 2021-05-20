#!/usr/bin/env python3
import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import ndarray as Array

"""
Make a dataset from all the images in a directory, to display during mturk.
"""

#------------------------------------------------------------------------------
# Define global settings.

ACTIONS = ["clench", "pinch", "palm", "poke"]
font = cv2.FONT_HERSHEY_SIMPLEX
data_dir = '/Users/sharon/data/EE148/affordance/segmentation_amazon_mturk'
output_dir = '/Users/sharon/data/EE148/affordance/segmentation_amazon_mturk_combined'


#------------------------------------------------------------------------------
# Define functions to create combined image dataset.

def make_combined_images(directory):
    """Load the original image names from a directory, and make combined images."""

    print(directory)
    image_names = []
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            path = os.path.join(root, fname)
            try:
                if fname[-5:] == '_.png':
                    make_combined_image(fname[:-4])
            except:
                continue


def make_combined_image(image_name):
    """Make images where original image is combined with the segmented images."""

    img1 = cv2.imread('%s/%s.png' % (data_dir, image_name))
    for action in ACTIONS:
        try:
            #img2 = cv2.imread('%s/%s_pos_sal_%s.jpg' % (data_dir, image_name, action))
            img2 = cv2.imread('%s/%s_%s.png' % (data_dir, image_name, action))

            # vertically concatenates images
            # of same width
            im_v = cv2.vconcat([img1, img2])
            
            cv2.imwrite('%s/%s_%s_combined.png' % 
                    (output_dir, image_name, action), im_v)

            if random.uniform(0, 1) < .05:
                # make catch image
                random_action = random.choice(ACTIONS)
                text1 = 'To show that you are'
                text2 = 'paying attention,'
                text3 = 'select %s.' % random_action
                font = cv2.FONT_HERSHEY_SIMPLEX
                textsize1 = cv2.getTextSize(text1, font, 1, 2)[0]
                textsize2 = cv2.getTextSize(text2, font, 1, 2)[0]
                textsize3 = cv2.getTextSize(text3, font, 1, 2)[0]
                textX1 = (img2.shape[1] - textsize1[0] // 2) // 2
                textY1 = (img2.shape[0] - textsize1[1]) // 2
                textX2 = (img2.shape[1] - textsize2[0] // 2) // 2
                textY2 = (img2.shape[0] + textsize2[1]) // 2
                textX3 = (img2.shape[1] - textsize3[0] // 2) // 2
                textY3 = (img2.shape[0] + 3*textsize3[1]) // 2
                cv2.putText(img2, text1, (textX1, textY1), font, .5, (255, 255, 255), 2)
                cv2.putText(img2, text2, (textX2, textY2), font, .5, (255, 255, 255), 2) 
                cv2.putText(img2, text3, (textX3, textY3), font, .5, (255, 255, 255), 2) 
                im_v = cv2.vconcat([img1, img2])
                cv2.imwrite('%s/%s_%s_catch-%s.png' % 
                        (output_dir, image_name, action, random_action), im_v)
                print('%s/%s_%s_catch-%s.png' % 
                        (output_dir, image_name, action, random_action))
        except Exception as e:
            continue


#------------------------------------------------------------------------------
# Run the program.

if __name__ == "__main__":
    image_names = make_combined_images(data_dir)
