from .base import *
from .coco import COCOSegmentation
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CitySegmentation
from .cocostuff import CocostuffSegmentation
from .pcontext3 import ContextSegmentation3
from .pcontext5 import ContextSegmentation5
from .pcontext_br import ContextSegmentation_br
datasets = {
    'coco': COCOSegmentation,
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'citys': CitySegmentation,
    'cocostuff': CocostuffSegmentation,
    'pcontext3': ContextSegmentation3,
    'pcontext5': ContextSegmentation5,
    'pcontext_br': ContextSegmentation_br,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
