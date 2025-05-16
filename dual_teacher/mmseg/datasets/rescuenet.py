import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class RescueNetDataset(CustomDataset):
    """RescueNet dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_lab.png' for RescueNet dataset.
    """

    CLASSES = ('background', 'water', 'building-no-damage',
               'building-medium-damage', 'building-major-damage', 'building-total-destruction', 
               'vehicle', 'road-clear', 'road-blocked', 'tree', 'pool')

    PALETTE = [[0, 0, 0], [61, 230, 250], [180, 120, 120], [235, 255, 7],
               [255, 184, 6], [255, 0, 0], [255, 0, 245], [140, 140, 140],
               [160, 150, 20], [4, 250, 7], [255, 235, 0]]

    def __init__(self, **kwargs):
        super(RescueNetDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_lab.png',
            **kwargs)
    
    

