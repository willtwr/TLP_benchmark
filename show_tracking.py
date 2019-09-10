#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Show tracking results
Platform:
  Windows or Linux

Dependencies:
  pillow
  sacred
  bilib

Usages:
  1. Set the absolute path of the tracker_benchmark (from: https://github.com/bilylee/tracker_benchmark)
  2. Unzip SiamFC-lu.zip in tracker_benchmark/results/OPE/
  3. Run: python show_tracking.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import sys
import os
import os.path as osp
import json
from PIL import Image, ImageDraw, ImageFont
from sacred import Experiment
from bilib import mkdir_p, videofig, draw_bounding_box_on_image
from bilib.visualization_utils import STANDARD_COLORS

ex = Experiment()

TRACKER_BENCHMARK_DIR = r''

@ex.config
def configs():
  #videoname = 'Jogging-1'
  
  testnames = [
    'SiamFC_nonlinear_upNiter_maxmean_base', 'MDNet', 'ECO', 'SiamFC'
  ]
  #testnames = [
  #  'SiamFC_bg_l', 'SiamFC_bg_baset', 'SiamFC_multi_temp'
  #]
  #testnames = [
  #  'SiamFC_nonlinear_upNiter_maxmean_base'
  #]
  ids = [None] * len(testnames)

  colors = ['Red', 'Blue', 'Green', 'Magenta']
  
  log_dir = osp.join(TRACKER_BENCHMARK_DIR, 'results/OPE/')
  data_dir = osp.join(TRACKER_BENCHMARK_DIR, 'data/')

  videonames = os.listdir(data_dir)
  videonames = [v for v in videonames if osp.isdir(osp.join(data_dir, v))]
  videonames = ['PolarBear2']

  save = True # Save tracking figures
  save_format = 'image' # video or image
  save_dir = osp.join('.', 'results/samples')


def get_image_with_bb(idx, ids, colors, img_paths, gt_result, tracker_results):
  #img = get_image_with_bb.imgs.get(idx)
  img = None
  if img is None:
    img = Image.open(img_paths[idx]).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Add frame number
    # font = ImageFont.truetype("FreeSerif.ttf", 25) # fonts are located at /usr/share/fonts/
    # draw.text((0, 0), '#{0:04d}'.format(idx + 1), font=font, fill='yellow')
    draw.text((0, 0), '#{0:04d}'.format(idx + 1), fill='yellow')

    # Add gt box to image
    id_ = 'GT'
    box = gt_result['gtRect'][idx]
    xmin, ymin, w, h = box
    xmax = xmin + w - 1
    ymax = ymin + h - 1
    draw_bounding_box_on_image(img, ymin, xmin, ymax, xmax,
                               color=STANDARD_COLORS[-1],
                               display_str_list=[],
                               thickness=2,
                               use_normalized_coordinates=False)

    # Add bounding boxes to image
    for res_idx, result in enumerate(tracker_results):
      # tracker = result['tracker']
      id_ = ids[res_idx]
      #print(len(result['res']))
      box = result['res'][idx]
      xmin, ymin, w, h = box
      xmax = xmin + w - 1
      ymax = ymin + h - 1
      display_str_list = () if id_ is None else [id_]
      draw_bounding_box_on_image(img, ymin, xmin, ymax, xmax,
                                 color=colors[res_idx],
                                 display_str_list=display_str_list,
                                 thickness=2,
                                 use_normalized_coordinates=False)

    get_image_with_bb.imgs[idx] = img
  return img
get_image_with_bb.imgs = {}


@ex.automain
def main(testnames, videonames, ids, colors, save, save_format, log_dir, data_dir, save_dir):
  if not isinstance(testnames, list):
    testnames = [testnames]
    ids = ['ours']
  
  for videoname in videonames:
    #if osp.exists(osp.join(save_dir, videoname)):
    #  continue

    # Read ground truth boxes
    json_path = osp.join(data_dir, videoname, 'cfg.json')
    with open(json_path) as f:
      gt_result = json.load(f)
  
    # Read bounding boxes of trackers
    tracker_results = []
    for testname in testnames:
      json_path = osp.join(log_dir, testname, '{}.json'.format(videoname))
      with open(json_path) as f:
        res, = json.load(f)
        tracker_results.append(res)
  
    # Read image path
    if videoname == 'Tiger1':
      startFrame = 5
      gt_result['gtRect'] = gt_result['gtRect'][5:]
    else:
      startFrame = gt_result['startFrame']
    
    endFrame = gt_result['endFrame']
    img_paths = [osp.join(data_dir, videoname, 'img', gt_result['imgFormat'].format(idx)) for idx in
                 range(startFrame, endFrame + 1)]
  
    if save:
      mkdir_p(osp.join(save_dir, videoname + '_full'))
      if save_format == 'video':
        # You need ffmpeg installed for saving videos
        from skvideo.io import FFmpegWriter
        # FFmpegWriter uses str instead of unicode as input
        videopath = osp.join(save_dir, videoname, '{}.mp4'.format(videoname)).encode('ascii', 'replace')
        writer = FFmpegWriter(videopath)
  
      #print(len(img_paths))
      for idx in range(0, len(img_paths)):
        if idx > 4000:
          break
        img = get_image_with_bb(idx, ids, colors, img_paths, gt_result, tracker_results)
        path = osp.join(save_dir, videoname + '_full', gt_result['imgFormat'].format(startFrame + idx))
        if save_format == 'video':
          writer.writeFrame(img)
        else:
          img.save(path)
      print('Result saved to {}'.format(osp.join(save_dir, videoname)))
    else:
      # Show images with bounding box
      def redraw_fn(f, axes):
        img = get_image_with_bb(f, ids, colors, img_paths, gt_result, tracker_results)
        if not redraw_fn.initialized:
          im = axes.imshow(img, animated=True)
          redraw_fn.im = im
          redraw_fn.initialized = True
        else:
          redraw_fn.im.set_array(img)
      redraw_fn.initialized = False
      videofig(len(img_paths), redraw_fn, play_fps=30)
