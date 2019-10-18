#! /usr/bin/env python

"""
Lane Detection

    Usage: python3 test.py  --conf=./config.json

"""

import cv2
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import argparse
import json
from tqdm import tqdm 
import glob
import os
from src.data_utils.data_loader import get_pairs_from_paths, get_segmentation_arr, get_image_arr
from src.frontend import Segment
from src.metrics import get_iou

# define command line arguments
argparser = argparse.ArgumentParser(
    description='Evaluate Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config.json",
    help='path to configuration file')


def _main_(args):
    """
    :param args: command line argument
    """

    # Parse command line argument
    config_path = args.conf

    # Open and load the config json
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
    model.load_weights(config['test']['model_file'])

    ious = []
    for inp, ann  in tqdm( get_pairs_from_paths(config['test']['test_images'], config['test']['test_annotations']) ):
        net_input = np.expand_dims(get_image_arr(inp, input_size[0], input_size[1]), axis=0)
        pred_raw = model.predict(net_input)
        ground_truth = get_segmentation_arr( ann , config['model']['classes'],  config['model']['out_width'], config['model']['out_height']  )
        pred = pred_raw[:,:,:,:].reshape((pred_raw.shape[1], pred_raw.shape[2], config['model']['classes']))
        pred[pred>0.5] = 1
        iou = get_iou( ground_truth, pred, config['model']['classes'] )
        print("IoU of class 1 (road) in current image: ", iou[1])
        ious.append( iou[1] )
    
    ious = np.array( ious )
    print("Mean IoU of class 1 (road): "  ,  np.mean(ious ))


if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
