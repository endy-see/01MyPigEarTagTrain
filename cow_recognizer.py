from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import time
import facenet

S_SUCCESS = 0
S_COW1_NO_HEAD = 1
S_COW2_NO_HEAD = 2

class CowRecognizer():

    def __init__(self, recognizer_model_path, detector = None, cow_head_image_size=182):
        self._detector = detector
        self._cow_head_image_size = cow_head_image_size

        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                facenet.load_model(recognizer_model_path)

                # Get input and output tensors
                self._images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self._embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self._phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self._sess = sess


    # crop the cow head from the raw image
    def cropImage(self, im, region):
        try:
            im_h, im_w, _ = im.shape
            left, top, right, bottom, score = region
            width = right - left
            height = bottom - top
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            if width > height:
                delta = (width - height) / 2
                top = np.maximum(top - delta, 0)
                bottom = np.minimum(bottom + delta, im_h)
            else:
                delta = (height - width) / 2
                left = np.maximum(left - delta, 0)
                right = np.minimum(right + delta, im_w)
            
            cropped = im[top:bottom, left:right,:]
            return cropped
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return None
    
    def extractFeatures(self, images):
        feedImages = []
        for img in images:
            img_size = np.asarray(img.shape)[0:2]
            img = misc.imresize(img, (self._cow_head_image_size, self._cow_head_image_size), interp='bilinear')
            img = facenet.prewhiten(img)
            feedImages.append(img)

        feed_dict = { self._images_placeholder: np.stack(feedImages), self._phase_train_placeholder: False }
        embs = self._sess.run(self._embeddings, feed_dict=feed_dict)
        return embs

    def compareFeatures(self, feature1, feature2):
        dist = np.sqrt(np.sum(np.square(np.subtract(feature1, feature2))))
        return dist

    def detectCowHead(self, image):
        if self._detector is None:
            return image, None

        regions = self._detector.detect(image)
        if regions.has_key('head'):
            # select top 1 regions which has the highest score
            if len(regions['head']) > 0:
                region = regions['head'][0]
                return self.cropImage(image, region), region
        
        return None, None

    # compare two cow images, returns:
    # status: 0 success, 1 image1 no head detected, 2 image2 no head detected
    # distance between the two cow
    # head region of cow 1
    # head region of cow 2
    def compareImages(self, image1, image2):
        # step 1, detect the cow head in the image
        start = time.time()
        head1, region1 = self.detectCowHead(image1)
        # print("Detection cow 1 cost time: {0}".format(time.time() - start))
        if head1 is None:
            return S_COW1_NO_HEAD, None, None, None

        # detect_start = time.time()
        head2, region2 = self.detectCowHead(image2)
        # print("Detection cow 2 cost time: {0}".format(time.time() - detect_start))
        if head2 is None:
            return S_COW2_NO_HEAD, None, None, None

        # extract_start = time.time()
        features = self.extractFeatures([head1, head2])
        # print("Extract features cost time: {0}".format(time.time() - extract_start))

        dist = self.compareFeatures(features[0], features[1])

        print("Total cost time: {0}".format(time.time() - start))

        return S_SUCCESS, dist, region1, region2

    def compareImageFiles(self, image_file_path1, image_file_path2):
        image1 = misc.imread(os.path.expanduser(image_file_path1))
        image2 = misc.imread(os.path.expanduser(image_file_path2))
        return self.compareImages(image1, image2)

    # convert the distance to 0-100 score
    # higher is similar
    @staticmethod
    def convert2Score(distance, thredhold=1.0):
        assert thredhold > 0.5 and thredhold < 1.6
        assert distance >= 0

        if distance <= thredhold:
            return int(50 +  50 * (thredhold - distance) / (thredhold - 0.5))
        else:
            return max(0, int(50 - 50 * (distance - thredhold)/(1.6 - thredhold)))




