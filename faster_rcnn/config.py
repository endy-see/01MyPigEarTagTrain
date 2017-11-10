from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


class cfg_default:
    def __init__(self):
        # self.__C = edict()
        # Consumers can get config by:
        #   from fast_rcnn_config import cfg
        # self.cfg = self.__C
        self.cfg = edict()
        _C = self.cfg
        #
        # Testing options
        #
        _C.TEST = edict()

        # Scale to use during testing (can NOT list multiple scales)
        # The scale is the pixel size of an image's shortest side
        _C.TEST.SCALES = (600,)

        # Max pixel size of the longest side of a scaled input image
        _C.TEST.MAX_SIZE = 1000

        # Overlap threshold used for non-maximum suppression (suppress boxes with
        # IoU >= this threshold)
        _C.TEST.NMS = 0.3

        # Experimental: treat the (K+1) units in the cls_score layer as linear
        # predictors (trained, eg, with one-vs-rest SVMs).
        _C.TEST.SVM = False

        # Test using bounding-box regressors
        _C.TEST.BBOX_REG = True

        # Propose boxes
        _C.TEST.HAS_RPN = False

        # Test using these proposals
        _C.TEST.PROPOSAL_METHOD = 'gt'

        ## NMS threshold used on RPN proposals
        _C.TEST.RPN_NMS_THRESH = 0.7
        ## Number of top scoring boxes to keep before apply NMS to RPN proposals
        _C.TEST.RPN_PRE_NMS_TOP_N = 6000

        ## Number of top scoring boxes to keep after applying NMS to RPN proposals
        _C.TEST.RPN_POST_NMS_TOP_N = 300

        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        # __C.TEST.RPN_MIN_SIZE = 16

        # Testing mode, default to be 'nms', 'top' is slower but better
        # See report for details
        _C.TEST.MODE = 'nms'

        # Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
        _C.TEST.RPN_TOP_N = 5000

        #
        # ResNet options
        #

        _C.RESNET = edict()

        # Option to set if max-pooling is appended after crop_and_resize.
        # if true, the region will be resized to a square of 2xPOOLING_SIZE,
        # then 2x2 max-pooling is applied; otherwise the region will be directly
        # resized to a square of POOLING_SIZE
        _C.RESNET.MAX_POOL = False

        # Number of fixed blocks during training, by default the first of all 4 blocks is fixed
        # Range: 0 (none) to 3 (all)
        _C.RESNET.FIXED_BLOCKS = 1

        #
        # MobileNet options
        #

        _C.MOBILENET = edict()

        # Whether to regularize the depth-wise filters during training
        _C.MOBILENET.REGU_DEPTH = False

        # Number of fixed layers during training, by default the first of all 14 layers is fixed
        # Range: 0 (none) to 12 (all)
        _C.MOBILENET.FIXED_LAYERS = 5

        # Weight decay for the mobilenet weights
        _C.MOBILENET.WEIGHT_DECAY = 0.00004

        # Depth multiplier
        _C.MOBILENET.DEPTH_MULTIPLIER = 0.25

        #
        # MISC
        #

        # The mapping from image coordinates to feature map coordinates might cause
        # some boxes that are distinct in image space to become identical in feature
        # coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
        # for identifying duplicate boxes.
        # 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
        _C.DEDUP_BOXES = 1. / 16.

        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # We use the same pixel mean for all networks even though it's not exactly what
        # they were trained with
        _C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

        # For reproducibility
        _C.RNG_SEED = 3

        # A small number that's used many times
        _C.EPS = 1e-14

        # Root directory of project
        _C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

        # Data directory
        _C.DATA_DIR = osp.abspath(osp.join(_C.ROOT_DIR, 'data'))

        # Name (or path to) the matlab executable
        _C.MATLAB = 'matlab'

        # Place outputs under an experiments directory
        _C.EXP_DIR = 'default'

        # Use GPU implementation of non-maximum suppression
        _C.USE_GPU_NMS = False

        # Default GPU device id
        _C.GPU_ID = 0

        # Default pooling mode, only 'crop' is available
        _C.POOLING_MODE = 'crop'

        # Size of the pooled region after RoI pooling
        _C.POOLING_SIZE = 7

        # Anchor scales for RPN
        _C.ANCHOR_SCALES = [8, 16, 32]

        # Anchor ratios for RPN
        _C.ANCHOR_RATIOS = [0.5, 1, 2]

        # output feature map stride
        _C.FEAT_STRIDE = [16]

    def _merge_a_into_b(self, a, b):
        """Merge config dictionary a into config dictionary b, clobbering the
        options in b whenever they are also specified in a.
        """
        if type(a) is not edict:
            return

        for k, v in a.items():
            # a must specify keys that are in b
            if k not in b:
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(type(b[k]),
                                                                   type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    self._merge_a_into_b(a[k], b[k])
                except:
                    print(('Error under config key: {}'.format(k)))
                    raise
            else:
                b[k] = v

    def cfg_from_file(self, filename):
        """Load a config file and merge it into the default options."""
        import yaml

        with open(filename, 'r') as f:
            yaml_cfg = edict(yaml.load(f))

        self._merge_a_into_b(yaml_cfg, self.cfg)

    def cfg_from_list(self, cfg_list):
        """Set config keys via list (e.g., from command line)."""
        from ast import literal_eval
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = k.split('.')
            d = self.__C
            for subkey in key_list[:-1]:
                assert subkey in d
                d = d[subkey]
            subkey = key_list[-1]
            assert subkey in d
            try:
                value = literal_eval(v)
            except:
                # handle the case when v is a string literal
                value = v
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(
                    type(value), type(d[subkey]))
            d[subkey] = value

