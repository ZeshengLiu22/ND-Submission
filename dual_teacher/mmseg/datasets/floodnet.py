import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class FloodNetDataset(CustomDataset):
    """FloodNet dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_lab.png' for FloodNet dataset.
    """

    CLASSES = ('background', 'building-flooded', 'building-non-flooded',
               'road-flooded', 'road-non-flooded', 'water', 'tree',
               'vehicle', 'pool', 'grass')

    PALETTE = [[0, 0, 0], [255, 0, 0], [180, 120, 120], [160, 150, 20],
               [140, 140, 140], [61, 230, 250], [0, 82, 255], [255, 0, 245],
               [255, 235, 0], [4, 250, 7]]

    def __init__(self, **kwargs):
        super(FloodNetDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_lab.png',
            **kwargs)
    
    

