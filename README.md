# Cow Recognize 

## 1.  Recognize API

### 1.1 Constructor

Create the cow recognizer from config files

```python
import sys
import os.path as osp
import yaml

lib_path = osp.join(osp.dirname(__file__), 'faster_rcnn')
sys.path.insert(0, lib_path)

from cow_recognizer import CowRecognizer
import faster_rcnn.object_detector as FasterRCNN


def create_recognizer(config_file_name='configs.yml'):
    # load config file
    this_dir = osp.dirname(osp.abspath(__file__))
    config_file = osp.join(this_dir, config_file_name)
    configs = yaml.load(open(config_file))

    # create cow head detector
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
    
    # create cow recognizer
    recognizer = CowRecognizer(model_path, detector, image_size)
    return recognizer


# create the cow recognizer by the default config files
recognizer = create_recognizer()

```


### 1.2  Compare Cows From Image Files

```python
status, distance, region1, region2 = recognizer.compareImageFiles(image1_path, image2_path)
```

`status`: 0 success, 1 no cow head detect in image1, 2 no cow head detected in image2          
`distance`: float, distance of the two cow, must be >= 0.0, smaller the similar       
`region1` and `region2`: the region for the head in the image, format [left, top, right, bottom, score]

If `status != 0` the `distance`, `region1` and `region2` is `None`!


### 1.3 Compare Cows From Image Objects

```python
from scipy import misc

image1 = misc.imread(os.path.expanduser(image_file_path1))
image2 = misc.imread(os.path.expanduser(image_file_path2))
return recognizer.compareImages(image1, image2)
```

The output is the same as `compareImageFiles`


### 1.4 Detect Cow Head

```python
headImg, region = recognizer.detectCowHead(image)
```

Return the top 1 head detected in the image.        
`headImg` the head image cropped from the cow image          
`region` head region in the cow image, format [left, top, right, bottom, score]  


### 1.5 Extract Features Fom Head Image List


```python
headImgs = [headImg1, headImag2, ...]
features = recognizer.extractFeatures(headImgs)
```


### 1.6 Count The Distance Between Two Head Features

```python
distance = recognizer.compareFeatures(features[0], features[1])
```


### 1.7 Static Method: convert distance to score

Convert the distance of two cow head features to score in [0,100], the higher the similar.

```python
score = CowRecognizer.convert2Score(distance, thredhold=1.0)
```

***Please set the thredhold carefully, recommend to use the thredhold base the the evalution reports***


## 2.  Face Align API

### 2.1 Constructor
```python
def create_aligner(config_file_name='configs.yml'):
    this_dir=osp.dirname(osp.abspath(__file__))
    config_file = osp.join(this_dir, config_file_name)
    configs = yaml.load(open(config_file))
    aligner=None
    if configs.has_key('face_aligner'):
        aligner_model_path = osp.join(this_dir, configs['face_aligner']['model_path'])
        aligner=CowFaceAligner(aligner_model_path,182)

    return aligner


```

### 2.2 Get Head Landmarks from Detected Head Image
```python
 landmarks = aligner.DetectLandmarks(head,head.size[0],head.size[1])

```
### 2.3 Get Aligned Head Image
Positions of both eyes in dest aligned head image should be set according to landmark model. 
 
```python
aligned_head = aligner.AlignFace(head, landmarks,
                                         eye_left_dest=(40, 65),
                                         eye_right_dest=(142, 65)
                                         )



```

## 3 Notes

1. 尽可能的传奶牛正脸图片(能看到牛的两个眼睛为准）；     
1. 照片中最好只有一头牛，尽量减少出现多头牛的情况；     
1. 奶牛在照片中不能太小，占照片总面积的30%-70%为佳；
1. 照片分辨率最好不要低于：640*640， 牛头部分不要低于182*182
1. 为了提高处理速度，照片大小控制在5M之内 或者1600*1600个像素内（非必须）  