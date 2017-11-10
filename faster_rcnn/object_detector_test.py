"""
Test the faster-rcnn detector

python object_detector_test.py model_path [classes_string]

The classes string is split by ',' and starts with 'background'. It is optional

Call example:
python object_detector_test.py ./configs/text.yml \
    ../assets/models/driver_license/tf_faster_rcnn/mobilenet_0.25_driver_text_detector_rpn.pb \
    ../assets/models/driver_license/tf_faster_rcnn/mobilenet_0.25_driver_text_detector_rcnn.pb \
    'background,text,seal,photo'

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os, cv2
import time
from object_detector import ObjectDetector

def show(im):
    msg = 'press any key to continue'
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.imshow(msg, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # load text line recongize model
    config_file = sys.argv[1]
    rpn_model_path = sys.argv[2]
    rcnn_model_path = sys.argv[3]
    classes = sys.argv[4].split(',')
    net = sys.argv[5]
    #classes = ['background', 'text', 'seal', 'photo']

    detector = ObjectDetector(rpn_model_path, rcnn_model_path, config_file, classes, conf_thresh=0.01, net=net)

    while True:
        path = raw_input("Image path (press 0 to exit):")
        if path == '0':
            break
        
        if not os.path.isfile(path):
            print("Image not exists!")
            continue
        
        img = cv2.imread(path)

        # detect text regions
        start = time.time()
        all_regions = detector.detect(img)
        end = time.time()
        print("Cost time {}".format(end-start))
        for key in all_regions.keys():
            boxes = all_regions[key]
            for i in range(len(boxes)):
                print("Boxes {}: {}".format(key, boxes[i]))
                cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=(255, 255, 0), thickness=1)
                
        show(img)