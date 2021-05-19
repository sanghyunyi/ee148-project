#!/usr/bin/env python3
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray as Array

"""
Make a dataset from all the images in a directory, to display during mturk.
"""

#------------------------------------------------------------------------------
# Define global settings.

ACTIONS = ["clench", "pinch", "palm", "poke"]
data_dir = '/Users/sharon/data/EE148/affordance/results_amazon'
output_dir = '/Users/sharon/data/EE148/affordance/results_amazon_combined'


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
            img2 = cv2.imread('%s/%s_pos_sal_%s.jpg' % (data_dir, image_name, action))
            #img2 = cv2.imread('%s/%s_%s.png' % (data_dir, image_name, action))

            # vertically concatenates images
            # of same width
            im_v = cv2.vconcat([img1, img2])
            
            cv2.imwrite('%s/%s_%s_combined.png' % 
                    (output_dir, image_name, action), im_v)
        except Exception as e:
            continue


#------------------------------------------------------------------------------
# Run the program.

if __name__ == "__main__":
    image_names = make_combined_images(data_dir)
