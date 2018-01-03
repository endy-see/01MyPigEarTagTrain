# coding=UTF-8
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image

class  CowFacePosesDetector():

    def __init__(self,face_pose_model_path):
        self._image_size=120
        self._face_pose_model_path=face_pose_model_path
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with gfile.FastGFile(self._face_pose_model_path, 'rb') as f:
              graph_def.ParseFromString(f.read())
              tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess:

                self._sess=sess
                self._input=tf.get_default_graph().get_tensor_by_name("input_image:0")
                self._predictions=tf.get_default_graph().get_tensor_by_name("probabilities:0")


    # Get landmarks in a cropped face image
    def DetectPose(self, image):
        '''
        Detect pose of given cattle face image.
        :param image: Image of cropped cattle face,no restriction on image size.
        :return: Score of being a front face.      
        '''

        try:

            prediction_result = self._sess.run(self._predictions, feed_dict={self._input: image})
            return prediction_result[0][1]

        except:

            return None


if __name__=='__main__':

    pose_detector=CowFacePosesDetector('cow_face_pose_model.pb') 
    image=Image.open(image_path)
    front_pose_score=pose_detector.DetectPose(image)
    
    print 'lucky'