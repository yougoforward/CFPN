from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *

from .cfpn import *
from .cfpn_gsf import *
from .dfpn8_gsf import *
from .dfpn82_gsf import *
from .dfpn83_gsf import *
from .dfpn84_gsf import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,

        'cfpn': get_cfpn,
        'cfpn_gsf': get_cfpn_gsf,
        'dfpn8_gsf': get_dfpn8_gsf,
        'dfpn82_gsf': get_dfpn82_gsf,
        'dfpn83_gsf': get_dfpn83_gsf,
        'dfpn84_gsf': get_dfpn84_gsf,

    }
    return models[name.lower()](**kwargs)
