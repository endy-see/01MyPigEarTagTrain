# coding=UTF-8
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image, ImageDraw
import os
import math

# Stores landmarks
class Landmarks:
    '''
    Stores 6 key-points of cattle face.
    usage:
            landmarks = LandMarks()
            landmarks.right_ear = (1, 2)
            landmarks.right_eye = (kps[2], kps[3])
            landmarks.right_nose = (kps[4], kps[5])
            landmarks.left_nose = (kps[6], kps[7])
            landmarks.left_eye = (kps[8], kps[9])
            landmarks.left_ear = (kps[10], kps[11])
    '''
    pass

class  CowFaceAligner():

    def __init__(self,landmark_model_path):
        self._image_size=182
        self._landmarks_count=12
        self._lower_margin_rate=0.1
        self._landmark_model_path=landmark_model_path
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with gfile.FastGFile(self._landmark_model_path, 'rb') as f:
              graph_def.ParseFromString(f.read())
              tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess:

                self._sess=sess
                self._input=tf.get_default_graph().get_tensor_by_name("input_image:0")
                self._predictions=tf.get_default_graph().get_tensor_by_name("prediction:0")

    # Distance of two points
    def __Distance(self,p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    # Perform affine transformation by rotating with left eye as center.
    def __ScaleRotateTranslate(self,image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
        if (scale is None) and (center is None):
            return image.rotate(angle=angle, resample=resample)
        nx, ny = x, y = center
        sx = sy = 1.0
        if new_center:
            (nx, ny) = new_center
        if scale:
            (sx, sy) = (scale, scale)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine / sx
        b = sine / sx
        c = x - nx * a - ny * b
        d = -sine / sy
        e = cosine / sy
        f = y - nx * d - ny * e
        return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

    def __TupleMinus(self,a, b):

        if len(a) != 2 or len(b) != 2:
            return None
        return (a[0] - b[0], a[1] - b[1])

    def __TuplePlus(self,a, b):

        if len(a) != 2 or len(b) != 2:
            return None
        return (int(a[0] + b[0]), int(a[1] + b[1]))

    def __TupleMulti(self,a, multiplier):

        if len(a) != 2:
            return None
        return (int(a[0] * multiplier), int(a[1] * multiplier))

    # Convert landmarks obj to list obj
    def Landmarks2KeyPoints(self,landmarks):

        if not landmarks:
            return None

        kps=[-1 for x in range(12)]

        if landmarks.right_ear:
            kps[0] = landmarks.right_ear[0]
            kps[1] = landmarks.right_ear[1]

        if landmarks.right_eye:
            kps[2]=landmarks.right_eye[0]
            kps[3]=landmarks.right_eye[1]

        if landmarks.right_nose:
            kps[4]=landmarks.right_nose[0]
            kps[5]=landmarks.right_nose[1]

        if landmarks.left_nose:
            kps[6]=landmarks.left_nose[0]
            kps[7]=landmarks.left_nose[1]

        if landmarks.left_eye:
            kps[8]=landmarks.left_eye[0]
            kps[9]=landmarks.left_eye[1]

        if landmarks.left_ear:
            kps[10]=landmarks.left_ear[0]
            kps[11]=landmarks.left_ear[1]

        if kps.__contains__(-1):
            return None
        else:
            return kps

    # Convert list obj to landmarks obj
    def KeyPoints2Landmarks(self,kps):

        if not kps:
            return None

        if len(kps)!=self._landmarks_count:
            return None

        landmarks = Landmarks()
        landmarks.right_ear = (kps[0], kps[1])
        landmarks.right_eye = (kps[2], kps[3])
        landmarks.right_nose = (kps[4], kps[5])
        landmarks.left_nose = (kps[6], kps[7])
        landmarks.left_eye = (kps[8], kps[9])
        landmarks.left_ear = (kps[10], kps[11])

        return landmarks

    # Get landmarks in a cropped face image
    def DetectLandmarks(self, image, image_width=182, image_height=182):
        '''
        Detate landmarks of given cattle face image.
        :param image: Image of cropped cattle face,no restriction on image size.
        :param image_width: input image width.
        :param image_height: input image height.
        :return: Land marks of input image.
        '''

        try:

            prediction_result = self._sess.run(self._predictions, feed_dict={self._input: image})

            scale_x=float(image_width)/182
            scale_y= float(image_height) / 182

            kps=[]
            for index,value in enumerate(prediction_result):
                if index%2==0:
                    value=value*scale_x
                else:
                    value=value*scale_y

                kps.append(int(round(value)))

            if len(kps) != self._landmarks_count:
                return None

            landmarks = self.KeyPoints2Landmarks(kps)

            return landmarks

        except:

            return None


    def AlignFace(self, image, landmarks, dest_sz=(182, 182),
                  eye_left_dest=(15, 30), eye_right_dest=(167, 30)):
        '''
        Align cattle face,eye_left_dest and eye_right_dest 
        must be symmetry in x axis and equal in y axis .
        :param image: Input image.
        :param landmarks: Landmarks of related image.
        :param dest_sz: Output size of aligned cattle face image.
        :param eye_left_dest: Left eye location on aligned face image.
        :param eye_right_dest: Right eye location on aligned face iamge.
        :return: Aligned cattle face image.
        '''

        if (eye_right_dest[0] <= eye_left_dest[0]):
            return None

        if eye_right_dest[0] > dest_sz[0]:
            return None

        src_img_size = image.size[0]

        eye_vector = self.__TupleMinus(landmarks.right_eye, landmarks.left_eye)

        # Cal rotate angle.
        rotation = -math.atan2(float(eye_vector[1]), float(eye_vector[0]))
        cosin = math.cos(rotation)
        sin = math.sin(rotation)

        # Cal eye distance。
        dist = self.__Distance(landmarks.left_eye, landmarks.right_eye)
        dest_eye_distance = eye_right_dest[0] - eye_left_dest[0]
        scale = float(dest_eye_distance) / dist

        # Rotate with left eye as center
        image = self.__ScaleRotateTranslate(image, center=landmarks.left_eye, angle=rotation, scale=1)

        left_nose_dest_relative = self.__TupleMinus(landmarks.left_nose, landmarks.left_eye)
        right_nose_dest_relative = self.__TupleMinus(landmarks.right_nose, landmarks.left_eye)


        nose_left_dest = self.__TuplePlus((left_nose_dest_relative[0] * cosin - left_nose_dest_relative[1] * sin
                                    , left_nose_dest_relative[0] * sin + left_nose_dest_relative[1] * cosin),
                                          landmarks.left_eye)

        nose_right_dest = self.__TuplePlus((right_nose_dest_relative[0] * cosin - right_nose_dest_relative[1] * sin
                                     , right_nose_dest_relative[0] * sin + right_nose_dest_relative[1] * cosin),
                                           landmarks.left_eye)

        eye_right_caled = (landmarks.left_eye[0] + dist, landmarks.left_eye[1])

        src_boarder_left = int(eye_left_dest[0] / scale)
        src_boarder_right = int((dest_sz[0] - eye_right_dest[0]) / scale)
        src_boarder_upper = (eye_left_dest[1] * (nose_left_dest[1] + src_img_size * self._lower_margin_rate - landmarks.left_eye[1])) \
                            / (dest_sz[1] - eye_left_dest[1])
        src_boarder_lower = int(src_img_size * self._lower_margin_rate)

        # Cal face box
        top_x = max(0, landmarks.left_eye[0] - src_boarder_left)
        top_y = max(landmarks.left_eye[1] - src_boarder_upper, 0)
        bottom_x = min(image.size[0], eye_right_caled[0] + src_boarder_right)
        bottom_y = min(max(nose_right_dest[1], nose_left_dest[1]) + src_boarder_lower, image.size[1])

        # Crop and resize
        image = image.crop((int(top_x), int(top_y), int(bottom_x), int(bottom_y)))
        image = image.resize(dest_sz, Image.ANTIALIAS)

        return image


if __name__=='__main__':

    aligner=CowFaceAligner('frozen_face_align_model.pb')
    image=Image.open('/000001_U_01_fliped.jpg')
    landmarks=aligner.DetectLandmarks(image)
    aligned_image=aligner.AlignFace(image,landmarks)
    aligned_image.show()
    print 'lucky'