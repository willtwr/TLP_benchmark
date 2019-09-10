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

# Code root absolute path
CODE_ROOT = '/media/william/data/SiamFC-workspace/TLP_benchmark/trackers/CREST'

sys.path.insert(0, CODE_ROOT)

import cv2
from cv2 import imwrite
import numpy as np
from os.path import realpath, dirname, join
import collections


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def run_CREST(seq, rp, bSaveImage):
  logging.info('Evaluating {}...'.format('DaSiamRPN-raw'))

  tic = time.clock()
  frames = seq.s_frames
  init_rect = seq.init_rect
  x, y, width, height = init_rect  # OTB format
  init_bb = Rectangle(x + width // 2, y + height // 2, width, height)

  first_name = frames[0]
  first_split = first_name.split('/')
  dir_name = os.path.join('/media/william/data/SiamFC-workspace/TLP_benchmark/results/samples', first_split[-3])
  if not os.path.exists(dir_name):
     os.mkdir(dir_name)
  
  video = first_split[-3]
  #video = list(video)
  #video[0] = video[0].lower()
  #video = "".join(video)
  #video = video.lower()
  video_path = os.path.join(CODE_ROOT, video + '.txt')
  with open(video_path, 'rb') as f:
    res_list = f.readlines()

  include_first = True
  reported_bboxs = []
  for i, filename in enumerate(frames):
    if i > 0 or include_first:
      image_file = filename
      if not image_file:
	break

      im = cv2.imread(image_file)  # HxWxC
      
      res = res_list[i].rstrip()
      res = res.split(' ')
      res = np.array(res).astype(np.float64)

      reported_bbox = Rectangle(res[0], res[1], res[2], res[3])
      reported_bboxs.append(reported_bbox)

      xmin = reported_bbox.x.astype(np.int32)
      ymin = reported_bbox.y.astype(np.int32)
      xmax = xmin + reported_bbox.width.astype(np.int32)
      ymax = ymin + reported_bbox.height.astype(np.int32)
      #cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
      #imwrite(os.path.join(dir_name, 'image_cropped{}.jpg'.format(i)), im)

  duration = time.clock() - tic

  trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in reported_bboxs]  # x, y add one to match OTB format

  result = dict()
  result['res'] = trajectory
  result['type'] = 'rect'
  result['fps'] = round(seq.len / duration, 3)
  return result
