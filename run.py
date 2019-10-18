#! /usr/bin/env python

"""
Lane Detection

    Usage: python3 run.py  --conf=./config.json

"""

import cv2
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import argparse
import json

from src.frontend import Segment

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Train and validate Kitti Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config.json",
    help='path to configuration file')

argparser.add_argument(
    '-m',
    '--model', default="UNET_trained.h5",
    help='path to model file')

argparser.add_argument(
    '-v',
    '--video', default="video.mp4",
    help='path to video file')


def _main_(args):
    """
    :param args: command line argument
    """

    # parse command line argument
    config_path = args.conf

    # open and load the config json
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # parse the json to retrieve the training configuration
    backend = config["model"]["backend"]
    input_size = (config["model"]["im_width"], config["model"]["im_height"])
    classes = config["model"]["classes"]

    # define the model and train
    segment = Segment(backend, input_size, classes)
   
    model = segment.feature_extractor

    # Load best model
    model.load_weights(args.model)

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
        ret, frame = cap.read()
        if ret == True:
            raw = cv2.resize(frame, (input_size[0], input_size[1]))

            # Sub mean
            # Because we use it with the training samples, I put it here
            # See in ./src/data/data_utils/data_loader
            img = raw.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            img = img[ : , : , ::-1 ]

            net_input = np.expand_dims(img, axis=0)
            preds = model.predict(net_input, verbose=1)
            pred_1 = preds[:,:,:,1].reshape((input_size[1], input_size[0]))
            pred_1[pred_1 < 0.2] = 0
            # print(pred_1)
            cv2.imshow("Raw", raw)
            cv2.waitKey(1)
            cv2.imshow("pred_1", pred_1)
            cv2.waitKey(1)
            print(preds.shape)


if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
