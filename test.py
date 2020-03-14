#! /usr/bin/env python

"""
Lane Detection

    Usage: python3 test.py  --conf=./config.json

"""


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
from src.data_utils.DataSequence import DataSequence
import time
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm

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

    # Data sequence for testing
    test_gen = DataSequence( config["test"]["test_images"], config["test"]["test_annotations"],  config["test"]["test_batch_size"],  config["model"]["classes"] , config["model"]['im_height'] , config["model"]['im_width'] , config["model"]['out_height'] , config["model"]['out_width'], do_augment=False)


    iou = sm.metrics.IOUScore(threshold=0.5)
    fscore = sm.metrics.FScore(threshold=0.5)
    metrics = [iou, fscore]
    model.compile(optimizer=Adam(0.1), loss="binary_crossentropy",
        metrics=metrics)

    model.evaluate(test_gen)
    scores = model.evaluate_generator(test_gen)
    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("mean {}: {:.5}".format(metric.__name__, value))


if __name__ == '__main__':
    # parse the arguments
    args = argparser.parse_args()
    _main_(args)
