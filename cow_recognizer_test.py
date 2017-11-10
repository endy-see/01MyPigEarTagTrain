import sys, time
import cv2, math
import os.path as osp
import yaml

lib_path = osp.join(osp.dirname(__file__), 'faster_rcnn')
sys.path.insert(0, lib_path)

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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        image_folder = osp.dirname(__file__)

    recognizer = create_recognizer()

    while True:
        path1 = raw_input("Image path 1:")
        if path1 == 'exit' or path1 == '0': break
        if path1[0] != '/': path1 = image_folder + path1
        if not osp.isfile(path1):
            print("Image not exists!")
            continue

        path2 = raw_input("Image path 2:")
        if path2 == 'exit' or path2 == '0': break
        if path2[0] != '/': path2 = image_folder + path2
        if not osp.isfile(path2):
            print("Image not exists!")
            continue

        status, distance, region1, region2 = recognizer.compareImageFiles(path1, path2)

        if status == 0:
            print("Distance: {}".format(distance))
        else:
            print("image {} no found cow head".format(status))
