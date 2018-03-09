#encoding:utf-8
from __future__ import print_function

import os
import os.path as osp
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import misc

lib_path = osp.join(osp.dirname(__file__), 'faster_rcnn')
sys.path.insert(0, lib_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cow_recognizer import CowRecognizer
from cow_face_aligner import CowFaceAligner
import faster_rcnn.object_detector as FasterRCNN


def create_recognizer(config_file_name='configs.yml'):
    this_dir = osp.dirname(osp.abspath(__file__))
    config_file = osp.join(this_dir, config_file_name)
    configs = yaml.load(open(config_file))

    detector = None
    if configs.has_key('detector'):
        config_file_path = osp.join(this_dir, configs['detector']['config_file_path'])
        rpn_model_path = osp.join(this_dir, configs['detector']['rpn_model_path'])
        rcnn_model_path = osp.join(this_dir, configs['detector']['rcnn_model_path'])
        detector = FasterRCNN.ObjectDetector(rpn_model_path, rcnn_model_path, config_file_path,
                                                classes=configs['detector']['classes'],
                                                net=configs['detector']['net'])


    model_path = osp.join(this_dir, configs['reognizer']['model_path'])
    image_size=182
    if configs['reognizer'].has_key('image_size'):
        image_size=configs['reognizer']['image_size']
    recognizer = CowRecognizer(model_path, detector, image_size)
    return recognizer


def create_aligner(config_file_name='configs.yml'):
    this_dir=osp.dirname(osp.abspath(__file__))
    config_file = osp.join(this_dir, config_file_name)
    configs = yaml.load(open(config_file))
    aligner=None
    if configs.has_key('face_aligner'):
        aligner_model_path = osp.join(this_dir, configs['face_aligner']['model_path'])
        aligner=CowFaceAligner(aligner_model_path,182)

    return aligner

if __name__ == '__main__':

    recognizer = create_recognizer()
    aligner=create_aligner()

    image = misc.imread('a.jpg')

    # Detect head
    head, region = recognizer.detectCowHead(image)
    if head is not None:

        # misc image to pil
        head = Image.fromarray(head)

        # Detect cow face landmarks
        landmarks = aligner.DetectLandmarks(head,head.size[0],head.size[1])

        # Draw landmarks on head image
        # predicts = aligner.Landmarks2KeyPoints(landmarks)
        # draw = ImageDraw.Draw(head)
        # diam = 1
        # bbox_1 = (predicts[0] - diam, predicts[1] - diam, predicts[0] + diam, predicts[1] + diam)
        # bbox_2 = (predicts[2] - diam, predicts[3] - diam, predicts[2] + diam, predicts[3] + diam)
        # bbox_3 = (predicts[4] - diam, predicts[5] - diam, predicts[4] + diam, predicts[5] + diam)
        # bbox_4 = (predicts[6] - diam, predicts[7] - diam, predicts[6] + diam, predicts[7] + diam)
        # bbox_5 = (predicts[8] - diam, predicts[9] - diam, predicts[8] + diam, predicts[9] + diam)
        # bbox_6 = (predicts[10] - diam, predicts[11] - diam, predicts[10] + diam, predicts[11] + diam)
        # draw.ellipse(bbox_1, fill='red')
        # draw.ellipse(bbox_2, fill='red')
        # draw.ellipse(bbox_3, fill='red')
        # draw.ellipse(bbox_4, fill='red')
        # draw.ellipse(bbox_5, fill='red')
        # draw.ellipse(bbox_6, fill='red')
        # head.show()

        # Make align on face image
        aligned_head = aligner.AlignFace(head, landmarks,
                                         eye_left_dest=(40, 65),
                                         eye_right_dest=(142, 65)
                                         )

        #pil image to misc
        aligned_head=np.array(aligned_head)

        # head feature extraction
        feature = recognizer.extractFeatures([aligned_head])

    print ('lucky')

