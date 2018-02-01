import os.path, sys
from flask import Flask, request, jsonify
import decimal, simplejson
import json
import numpy as np
import logging
import yaml
import uuid
import base64
from scipy import misc

this_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(this_dir, 'faster_rcnn')
sys.path.insert(0, lib_path)

from cow_recognizer import CowRecognizer
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

recognizer_configs = yaml.load(open(os.path.join(this_dir, 'configs.yml')))
compare_threshold = recognizer_configs['reognizer']['compare_threshold']
recognizer = create_recognizer(recognizer_configs)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config['server']['upload_folder']
app.logger.setLevel(logging.INFO)

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
        status, distance, region1, region2 = recognizer.compareImageFiles(save_path1, save_path2)
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
        app.logger.debug('End compareCows, info: %s, request_id: %s', result, request_id)
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

@app.route('/api/detect_face', methods=['POST'])
def detect_face():
    request_id = uuid.uuid1()
    img = request.files.get('image_file')
    if img is None:
        return jsonify({'status': 'PARAMETER_ERROR', 'error': 'image_file param is required'}), 400

    try:
        name = "%s.jpg"%(request_id)
        save_path = os.path.join(this_dir, app.config['UPLOAD_FOLDER'], name)
        img.save(save_path)
    except IOError as e:
        app.logger.error('I/O error(%s): %s, request_id: %s', e.errno, e.strerror, request_id)
        return jsonify({'status': 'SAVE_FILE_ERROR', 'error': e.strerror}), 500
    except TypeError as e:
        # if os.path.isfile(save_path):
        #     os.unlink(save_path)
        app.logger.error('TypeError: %s, request_id: %s', e, request_id)
        return jsonify({'status': 'FILE_DECODE_ERROR', 'error': 'file content is invalid'}), 500

    app.logger.debug('Begin detect_face, request_id: %s', request_id)

    try:
        oimg = misc.imread(os.path.expanduser(save_path))
        _, region = recognizer.detectCowHead(oimg)
    finally:
        if os.path.isfile(save_path):
            # os.unlink(save_path1)
            pass

    #print(region)
    if region is None:
        return jsonify({'status': 'NOT_RECOGNIZABLE'})
        app.logger.debug("Recognized nothing!")

    result = np.array(region).tolist()
    result = { 'face_rect': result[:4], 'confidence': result[4]}
    app.logger.debug('End detect_face, info: %s, request_id: %s', result, request_id)
    return jsonify({ 'status': 'OK', 'result': result })


if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host=server_config['address'], port=server_config['port'])
