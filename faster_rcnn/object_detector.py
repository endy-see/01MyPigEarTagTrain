#!/usr/bin/env python

# --------------------------------------------------------
# API of Tensorflow Faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
import os, cv2
import pprint
import math

from nms_wrapper import nms
from config import cfg_default
from generate_anchors import generate_anchors_pre
from proposal_layer import proposal_layer
import helpers


class ObjectDetector:
    def __init__(self, rpn_model_path, rcnn_model_path, cfg_file,
                 classes=None, conf_thresh=0.3, nms_thresh=0.5, iou_thresh=0.5, net='res50'):
        cfg = cfg_default()

        self._rpn_model_path = rpn_model_path
        self._rcnn_model_path = rcnn_model_path
        if classes is None:
            classes = ['background', 'object']
        self._classes = classes
        self._num_classes = len(self._classes)
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh
        self._iou_thresh = iou_thresh

        # load network configuration
        # pprint.pprint(cfg)
        self._feat_stride = cfg.cfg.FEAT_STRIDE
        cfg.cfg_from_file(cfg_file)
        # cfg_default.cfg_from_file(cfg_file)
        self.cfg = cfg
        self._anchor_scales = cfg.cfg.ANCHOR_SCALES
        self._num_scales = len(self._anchor_scales)
        self._anchor_ratios = cfg.cfg.ANCHOR_RATIOS
        self._num_ratios = len(self._anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios

        with tf.gfile.FastGFile(self._rpn_model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="rpn")
                self._images = graph.get_tensor_by_name('rpn/Placeholder:0')
                if net == 'mobile':
                    self._rpn_cls_prob = graph.get_tensor_by_name('rpn/MobilenetV1_2/rpn_cls_prob/transpose_1:0')
                    self._rpn_bbox_pred = graph.get_tensor_by_name('rpn/MobilenetV1_2/rpn_bbox_pred/BiasAdd:0')
                    self._rpn_feature_map = graph.get_tensor_by_name('rpn/MobilenetV1_1/Conv2d_11_pointwise/Relu6:0')
                elif net == 'res50':
                    self._rpn_cls_prob = graph.get_tensor_by_name('rpn/resnet_v1_50_3/rpn_cls_prob/transpose_1:0')
                    self._rpn_bbox_pred = graph.get_tensor_by_name('rpn/resnet_v1_50_3/rpn_bbox_pred/BiasAdd:0')
                    self._rpn_feature_map = graph.get_tensor_by_name(
                        'rpn/resnet_v1_50_2/block3/unit_6/bottleneck_v1/Relu:0')
                self._rpn_graph = graph

        with tf.gfile.FastGFile(self._rcnn_model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="rcnn")
                self._rcnn_feature_map = graph.get_tensor_by_name('rcnn/Placeholder_3:0')
                self._rois = graph.get_tensor_by_name('rcnn/Placeholder_4:0')
                if net == 'mobile':
                    self._cls_prob = graph.get_tensor_by_name('rcnn/MobilenetV1_2/cls_prob:0')
                    self._bbox_pred = graph.get_tensor_by_name('rcnn/add:0')
                elif net == 'res50':
                    self._cls_prob = graph.get_tensor_by_name('rcnn/resnet_v1_50_2/cls_prob:0')
                    self._bbox_pred = graph.get_tensor_by_name('rcnn/add:0')
                self._rcnn_graph = graph

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self._rpn_sess = tf.Session(graph=self._rpn_graph, config=tfconfig)
        self._rcnn_sess = tf.Session(graph=self._rcnn_graph, config=tfconfig)
        # self.printOperation(self._rpn_sess)
        # self.printOperation(self._rcnn_sess)

    def generate_anchors(self, im_info):
        height = int(math.ceil(im_info[0, 0] / np.float32(self._feat_stride[0])))
        width = int(math.ceil(im_info[0, 1] / np.float32(self._feat_stride[0])))
        anchors, anchor_length = generate_anchors_pre(height, width, self._feat_stride, self._anchor_scales,
                                                      self._anchor_ratios)

        anchors = np.reshape(anchors, [-1, 4])
        anchor_length = np.reshape(anchor_length, [-1])
        return anchors, anchor_length

    def im_detect(self, im):
        blobs, im_scales = helpers.get_blobs(self.cfg.cfg, im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        anchors, anchor_length = self.generate_anchors(blobs['im_info'])

        feed = {
            self._images: blobs['data']
        }
        rpn_cls_prob, rpn_bbox_pred, rpn_feature_map = self._rpn_sess.run(
            [self._rpn_cls_prob, self._rpn_bbox_pred, self._rpn_feature_map], feed)

        rois, _ = proposal_layer(self.cfg.cfg, rpn_cls_prob, rpn_bbox_pred, blobs['im_info'],
                                 self._feat_stride, anchors, self._num_anchors)
        rois = np.reshape(rois, [-1, 5])

        feed = {
            self._rcnn_feature_map: rpn_feature_map,
            self._rois: rois
        }
        scores, bbox_pred = self._rcnn_sess.run([self._cls_prob, self._bbox_pred], feed)

        boxes = rois[:, 1:5] / im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = helpers.bbox_transform_inv(boxes, box_deltas)
        pred_boxes = helpers.clip_boxes(pred_boxes, im.shape)

        return scores, pred_boxes

    def printOperation(self, sess):
        for op in sess.graph.get_operations():
            print(str(op.name))

    '''
    return is dictionary of regions info:
    {
        'class_1': [[x1, y1, x2, y2, score], [...]]
        'class_2': ...
    }
    '''

    def detect(self, image):
        scores, boxes = self.im_detect(image)
        all_regions = {}

        # process all classes except the background
        for cls_ind in range(1, self._num_classes):
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            detections = np.hstack((cls_boxes,
                                    cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(detections, self._nms_thresh)
            detections = detections[keep, :]

            regions = []
            #import pdb; pdb.set_trace()
            for detection in detections:
                overlap, idx = helpers.iou(np.asarray(regions), detection)
                if overlap < self._iou_thresh and detection[4] > self._conf_thresh:
                    region = (
                        int(detection[0]),
                        int(detection[1]),
                        int(math.ceil(detection[2])),
                        int(math.ceil(detection[3])),
                        detection[4]
                    )
                    regions.append(region)

            object_class = cls_ind
            if object_class < len(self._classes):
                object_class = self._classes[object_class]
            all_regions[object_class] = regions

        return all_regions
