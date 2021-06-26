from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .cfpn_gsf import *

from .fpn import *
from .dfcn import *
from .pam import *
from .gcn import *
from .refinenet import *
from .Upernet import *
from .ocnet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'cfpn_gsf': get_cfpn_gsf,
        'fpn': get_fpn,
        'dfcn': get_dfcn,
        'pam': get_pam,
        'gcn': get_gcn,
        'refinenet': get_refinenet,
        'upernet': get_Upernet,
        'ocnet': get_ocnet,
        
    }
    return models[name.lower()](**kwargs)
