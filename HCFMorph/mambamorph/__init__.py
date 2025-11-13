# ---- voxelmorph ----
# unsupervised learning for image registration

import os

# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
# import neurite
# minv = '0.2'
# curv = getattr(neurite, '__version__', None)
# if curv is None or version.parse(curv) < version.parse(minv):
#     raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
#                       f'but found version {curv}')

# move on the actual voxelmorph imports
from VMambaMorph.mambamorph import generators
from VMambaMorph.mambamorph import py
from VMambaMorph.mambamorph.py.utils import default_unet_features
import torch

# import backend-dependent submodules
# backend = py.utils.get_backend()
backend = 'pytorch'

if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    os.environ['NEURITE_BACKEND'] = 'pytorch'

    from VMambaMorph.mambamorph import torch
    from VMambaMorph.mambamorph.torch import layers
    from VMambaMorph.mambamorph.torch import networks
    from VMambaMorph.mambamorph.torch import losses

else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this voxelmorph backend')

    os.environ['NEURITE_BACKEND'] = 'tensorflow'

    # ensure valid tensorflow version is available
    minv = '2.4'
    curv = getattr(tensorflow, '__version__', None)
    if curv is None or version.parse(curv) < version.parse(minv):
        raise ImportError(f'voxelmorph requires tensorflow version {minv} or greater, '
                          f'but found version {curv}')

    from VMambaMorph.mambamorph import tf
    from VMambaMorph.mambamorph.tf import layers
    from VMambaMorph.mambamorph.tf import networks
    from VMambaMorph.mambamorph.tf import losses
    from VMambaMorph.mambamorph.tf import utils
