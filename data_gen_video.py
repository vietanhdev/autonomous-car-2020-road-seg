#! /usr/bin/env python

"""
Data Generator

"""

import cv2
import numpy as np
import argparse
import os

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Data Generator')

argparser.add_argument(
    '-v',
    '--video', default="video.mp4",
    help='path to video file')

argparser.add_argument(
    '-i',
    '--image_folder', default="./image",
    help='path to output rgb image folder')

argparser.add_argument(
    '-l',
    '--label_folder', default="./gt_image",
    help='path to output segmentation label folder')

argparser.add_argument(
    '-height',
    '--height', default="240",
    help='image height')

argparser.add_argument(
    '-width',
    '--width', default="320",
    help='image width')

argparser.add_argument(
    '-s',
    '--steps', default="3",
    help='number of steps between frames')


def _main_(args):
    """
    :param args: command line argument
    """

    # image count
    count = 0

    image_width = int(args.width)
    image_height = int(args.height)

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    if not os.path.exists(args.label_folder):
        os.makedirs(args.label_folder)

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(args.video)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        for i in range(int(args.steps)):
            ret, frame = cap.read()
        
        if not ret:
            break

        image = cv2.resize(frame, (image_width, image_height))

        cv2.imwrite(os.path.join(args.image_folder, "image_" + str(count).zfill(5) + ".png"), image)

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seg_image = gen_segmentation_watershed(image)

        cv2.imwrite(os.path.join(args.label_folder, "image_" + str(count).zfill(5) + ".png"), seg_image)

        print(count)
        count += 1
        

def gen_segmentation(im_in):

    connectivity = 4 # or 8?
    newMaskVal = 255
    flags = connectivity + (newMaskVal << 8)

    hsv = cv2.cvtColor(im_in, cv2.COLOR_BGR2HSV)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = hsv.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(hsv, mask, (160, 230), 255, loDiff=(5, 40, 100), upDiff=(5, 40, 100), flags=flags)

    return mask W

if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
