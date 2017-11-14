from __future__ import print_function

from scipy import misc
import sys, time
import math
import os
import os.path as osp
import yaml
import numpy as np

lib_path = osp.join(osp.dirname(__file__), 'faster_rcnn')
sys.path.insert(0, lib_path)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cow_recognizer import CowRecognizer
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

def eval(recognizer, image_files, objects, thredholds):
    nrof_images = len(image_files)
    nrof_thredholds = len(thredholds)

    tn = np.zeros(nrof_thredholds)
    fn = np.zeros(nrof_thredholds)
    tp = np.zeros(nrof_thredholds)
    fp = np.zeros(nrof_thredholds)

    print("Extract Features:")
    features = []
    for i in range(nrof_images):
        print(image_files[i])
        image = misc.imread(os.path.expanduser(image_files[i]))
        head, region = recognizer.detectCowHead(image)
        if head is None:
            features.append(None)
        else:
            feature = recognizer.extractFeatures([head])
            features.append(feature[0])

    print('Compare Images:')
    for i in range(nrof_images):
        print(image_files[i])
        for j in range(nrof_images):    
            if features[i] is not None and features[j] is not None:
                distance = recognizer.compareFeatures(features[i], features[j])
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
                            fn[k] += 1
                        else:
                            tn[k] +=1

        print('')

    
    same_count = tp[0] + fn[0]
    total_count = tp[0] + tn[0] + fp[0] + fn[0]
    print("Total: {}, same: {}".format(total_count, same_count))
    print('')
    print('                  Accuarcy    Validate     Recall      F-Score      FPR')
    for k in range(nrof_thredholds):
        precision = 1.0*tp[k]/(tp[k]+fp[k])
        TPR=recall = 1.0*tp[k]/same_count
        fscore = 2*precision*recall/(precision + recall)
        FPR = fp[k] / (fp[k] + tn[k])
        print('Thredhold %1.2f:    %1.4f,     %1.4f,    %1.4f,    %1.4f,    %1.4f' % (
            thredholds[k], 
            1.0*(tp[k] + tn[k]) / total_count, 
            precision, recall, fscore, FPR))

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
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        image_folder = osp.dirname(__file__)

    recognizer = create_recognizer()

    files, objects = get_images(image_folder)
    eval(recognizer, files, objects, np.arange(0, 1.3, 0.01))