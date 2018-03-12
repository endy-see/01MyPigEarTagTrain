import os.path, sys
from flask import Flask, request, jsonify
import decimal, simplejson
import json
import numpy as np
import logging
import time
import yaml
import uuid
import base64
from PIL import Image
from scipy import misc

this_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(this_dir, 'faster_rcnn')
sys.path.insert(0, lib_path)

from cow_recognizer import CowRecognizer
from cow_face_aligner import CowFaceAligner
import faster_rcnn.object_detector as FasterRCNN

config_file = os.path.join(this_dir, './server.yml')
config = yaml.load(open(config_file))
server_config = config['server']

class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)

def create_recognizer(configs):
    detector = None
    if configs.has_key('detector'):
        config_file_path = os.path.join(this_dir, configs['detector']['config_file_path'])
        rpn_model_path = os.path.join(this_dir, configs['detector']['rpn_model_path'])
        rcnn_model_path = os.path.join(this_dir, configs['detector']['rcnn_model_path'])
        detector = FasterRCNN.ObjectDetector(rpn_model_path, rcnn_model_path, config_file_path,
                                                classes=configs['detector']['classes'],
                                                net=configs['detector']['net'])


    model_path = os.path.join(this_dir, configs['reognizer']['model_path'])
    image_size=182
    if configs['reognizer'].has_key('image_size'):
        image_size=configs['reognizer']['image_size']
    recognizer = CowRecognizer(model_path, detector, image_size)
    return recognizer

def create_aligner(configs):
    aligner=None
    if configs.has_key('face_aligner'):
        aligner_model_path = os.path.join(this_dir, configs['face_aligner']['model_path'])
        aligner=CowFaceAligner(aligner_model_path,182)

    return aligner

sdk_configs = yaml.load(open(os.path.join(this_dir, 'configs.yml')))
compare_threshold = sdk_configs['reognizer']['compare_threshold']
recognizer = create_recognizer(sdk_configs)
aligner = create_aligner(sdk_configs)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = server_config['upload_folder']
app.logger.setLevel(logging.INFO)

def face_extract(path):
    image = misc.imread(path)
    head, region = recognizer.detectCowHead(image)
    if head is None:
        return None, region
    # misc image to pil
    head = Image.fromarray(head)
    # Detect cow face landmarks
    landmarks = aligner.DetectLandmarks(head,head.size[0],head.size[1])
    # Make align on face image
    aligned_head = aligner.AlignFace(head, landmarks,eye_left_dest=(40, 65),eye_right_dest=(142, 65))
    #pil image to misc
    aligned_head=np.array(aligned_head)
    # head feature extraction
    feature = recognizer.extractFeatures([aligned_head])
    
    return feature, region

def cow_verify(path1, path2):
    start = time.time()
    feature1, region1 = face_extract(path1)
    if feature1 is None:
        return 1

    feature2, region2 = face_extract(path2)
    if feature2 is None:
        return 2

    dist = recognizer.compareFeatures(feature1, feature2)
    print("Total cost time: {0}".format(time.time() - start))
    return 0, dist, region1, region2

@app.route('/')
def hello():
    return jsonify({
        'status': 'OK'
    })

@app.route('/api/compareCows', methods=['POST'])
def compareCows():
    request_id = uuid.uuid1()
    img1 = request.files.get('image1')
    if img1 is None:
        return jsonify({'status': 'PARAMETER_ERROR', 'error': 'image1 param is required'}), 400

    img2 = request.files.get('image2')
    if img2 is None:
        return jsonify({'status': 'PARAMETER_ERROR', 'error': 'image2 param is required'}), 400

    try:
        name1 = "%s_1.jpg"%(request_id)
        save_path1 = os.path.join(this_dir, app.config['UPLOAD_FOLDER'], name1)
        img1.save(save_path1)

        name2 = "%s_2.jpg"%(request_id)
        save_path2 = os.path.join(this_dir, app.config['UPLOAD_FOLDER'], name2)
        img2.save(save_path2)
    except IOError as e:
        app.logger.error('I/O error(%s): %s, request_id: %s', e.errno, e.strerror, request_id)
        return jsonify({'status': 'SAVE_FILE_ERROR', 'error': e.strerror}), 500
    except TypeError as e:
        # if os.path.isfile(save_path):
        #     os.unlink(save_path)
        app.logger.error('TypeError: %s, request_id: %s', e, request_id)
        return jsonify({'status': 'FILE_DECODE_ERROR', 'error': 'file content is invalid'}), 500

    app.logger.debug('Begin compareCows, request_id: %s', request_id)

    try:
        #status, distance, region1, region2 = recognizer.compareImageFiles(save_path1, save_path2)
        status, distance, region1, region2 = cow_verify(save_path1, save_path2)
    finally:
        if os.path.isfile(save_path1):
            # os.unlink(save_path1)
            pass

    thredhold = compare_threshold
    if status == 0: # succuss
        score = CowRecognizer.convert2Score(distance, thredhold)
        result = {
            'score': score,
            'distance': str(distance),
            'thredhold': str(thredhold),
            'region1': region1[:4],
            'region2': region2[:4]
        }
        #print(result)
        app.logger.debug('End driver_license ocr, info: %s, request_id: %s', result, request_id)
        # return jsonify({ 'status': 'OK', 'result': { 'region_info': region_info, 'info': texts } })
        return jsonify({ 'status': 'OK', 'result': result })
        # return simplejson.dumps(result, cls=DecimalJSONEncoder)
    elif status == 1: 
        return jsonify({'status': 'NO_COW_HEAD_DETECTED_IN_IMAGE1'})
        app.logger.debug("Not detect cow head in image1!")
    elif status == 2:
        return jsonify({'status': 'NO_COW_HEAD_DETECTED_IN_IMAGE2'})
        app.logger.debug("Not detect cow head in image2!")
    else:
        return jsonify({'status': 'NOT_RECOGNIZABLE'})
        app.logger.debug("Recognized nothing!")


if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host=server_config['address'], port=server_config['port'])
