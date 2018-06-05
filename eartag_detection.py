from __future__ import print_function

import sys, time
import math
import os
import os.path as osp
import yaml
import numpy as np
import cv2
import xml.etree.cElementTree as ET
import pandas as pd
from tqdm import tqdm
import csv

from IOU import calculateIoU

lib_path = osp.join(osp.dirname(__file__), 'faster_rcnn')
sys.path.insert(0, lib_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cow_recognizer import CowRecognizer
import faster_rcnn.object_detector as FasterRCNN


def create_recognizer(config_file_name='configs.yml'):
    this_dir = osp.dirname(osp.abspath(__file__))
    config_file = osp.join(this_dir, config_file_name)
    configs = yaml.load(open(config_file))

    detector = None
    if 'detector' in configs:
        config_file_path = osp.join(this_dir, configs['detector']['config_file_path'])
        rpn_model_path = osp.join(this_dir, configs['detector']['rpn_model_path'])
        rcnn_model_path = osp.join(this_dir, configs['detector']['rcnn_model_path'])
        detector = FasterRCNN.ObjectDetector(rpn_model_path, rcnn_model_path, config_file_path,
                                             classes=configs['detector']['classes'],
                                             net=configs['detector']['net'])

    return detector


def eval(recognizer, image_files, objects, thredholds):
    nrof_images = len(image_files)
    nrof_thredholds = len(thredholds)

    tn = np.zeros(nrof_thredholds)
    fn = np.zeros(nrof_thredholds)
    tp = np.zeros(nrof_thredholds)
    fp = np.zeros(nrof_thredholds)

    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, image_files[i]))
    print('')

    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %2d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%2d ' % i, end='')
        for j in range(nrof_images):
            status, distance, region1, region2 = recognizer.compareImageFiles(image_files[i], image_files[j])
            if status != 0:
                print('   NaN   ', end='')
            else:
                print('  %1.5f  ' % distance, end='')
                issame = objects[i] == objects[j]
                for k in range(nrof_thredholds):
                    positive = distance < thredholds[k]

                    if positive:
                        if issame:
                            tp[k] += 1
                        else:
                            fp[k] += 1
                    else:
                        if issame:
                            tn[k] += 1
                        else:
                            fn[k] += 1

        print('')

    print('')
    print('                  Accuarcy    Validate     Recall')
    same_count = tp[0] + tn[0]
    total_count = tp[0] + tn[0] + fp[0] + fn[0]
    for k in range(nrof_thredholds):
        print('Thredhold %1.2f:    %1.4f,     %1.4f,    %1.4f' % (
            thredholds[k], 1.0 * (tp[k] + fn[k]) / total_count,
            1.0 * tp[k] / (tp[k] + fp[k]),
            1.0 * tp[k] / same_count))


def get_images(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    objects = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for root, dirnames, _ in os.walk(data_path):
        for dirname in dirnames:
            folder_path = os.path.join(root, dirname)
            for parent, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    for ext in exts:
                        if filename.endswith(ext):
                            files.append(os.path.join(parent, filename))
                            objects.append(dirname)
                            break
    print('Find {} images'.format(len(files)))
    return files, objects


if __name__ == '__main__':
    # result_dir = './result/'
    result_dir = './result1/'
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        image_folder = osp.dirname(__file__)

    detector = create_recognizer()

    pics = os.listdir('./ttfix/')

    ious = []
    debugs = []

    for p in tqdm(pics):
        # read pics
        img = cv2.imread('./ttfix/' + p)
        # try:
        # result = detector.detect(img)['text']
        result = detector.detect(img)['head']
        # print(result)
        # except:
        # print('error', p)
        # continue
        # print(result[0])

        # read xmls
        address = './aafix/' + p[:-4] + '.xml'
        tree = ET.ElementTree(file=address)
        root = tree.getroot()

        gtxmin = int(root[6][4][0].text)
        gtymin = int(root[6][4][1].text)
        gtxmax = int(root[6][4][2].text)
        gtymax = int(root[6][4][3].text)
        # gt box: green
        cv2.rectangle(img, (gtxmin, gtymin), (gtxmax, gtymax), (0, 255, 0), 1)

        if result:
            xmin = result[0][0]
            ymin = result[0][1]
            xmax = result[0][2]
            ymax = result[0][3]
            score = result[0][4]
            # prediction box: blue
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

            wPred = xmax - xmin
            hPred = ymax - ymin
            wGt = gtxmax - gtxmin
            hGt = gtymax - gtymin

            (h, w) = img.shape[:2]
            # cv2.putText(img, str(score), (xmax, ymax), font, 2, (0,0,255), 1)
            print("output path: "+result_dir + p)
            cv2.imwrite(result_dir + p, img)
            iou = calculateIoU((xmin, ymin, xmax, ymax), (gtxmin, gtymin, gtxmax, gtymax))

            debugs.append([iou, wPred, hPred, wGt, hGt, w, h])
        else:
            print(p)
            cv2.imwrite('./exception/'+p, img)
            iou = 0


        ious.append(iou)
        # print(result, iou)

    picNumber = len(pics)
    d1 = pd.Series(ious)
    print('min:', d1.min(), 'max:', d1.max(), 'mean:', d1.mean(), 'median:', d1.median())

    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    cat = pd.cut(ious, bins)
    print(cat.categories)
    print(pd.value_counts(cat) / picNumber)

    output = [str(r)+'\n' for r in ious]
    print(output)
    with open('./logs/mobileear_v11n.txt', 'w') as f:
        f.writelines(output)

    with open('./debug/mobileear_v11n_debug.csv', 'w') as f:
        writer = csv.writer(f)
        # 写入多行用writerows
        writer.writerows(debugs)