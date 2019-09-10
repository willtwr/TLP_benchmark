#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Support integration with OTB benchmark"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time
import tensorflow as tf

# Code root absolute path
CODE_ROOT = '/home/william/SiamFC-TensorFlow'

# Checkpoint for evaluation
CHECKPOINT = '/home/william/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/SiamFC-2base-dense-nofbgs'

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import auto_select_gpu, load_cfgs
from inference import inference_wrapper_nl as inference_wrapper
from inference.tracker_nl import Tracker
from utils.infer_utils import Rectangle

# Set GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.logging.set_verbosity(tf.logging.DEBUG)


def run_SiamFC_nl(seq, rp, bSaveImage):
  checkpoint_path = CHECKPOINT
  logging.info('Evaluating {}...'.format(checkpoint_path))

  # Read configurations from json
  model_config, _, track_config = load_cfgs(checkpoint_path)

  track_config['log_level'] = 1  # Skip verbose logging for speed
  track_config['scale_step'] = 1.025
  track_config['scale_damp'] = 1.
  track_config['window_influence'] = 0.20
  
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint_path)
  g.finalize()
  
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  
  with tf.Session(graph=g, config=sess_config) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    tracker = Tracker(model, model_config, track_config)
    tic = time.time()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)

    first_name = frames[0]
    first_split = first_name.split('/')
    dir_name = os.path.join('/home/william/TLP_benchmark/results/samples', first_split[-3])
    if not os.path.exists(dir_name):
       os.mkdir(dir_name)

    trajectory_py = tracker.track(sess, init_bb, frames, logdir=dir_name)
    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.time() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result
