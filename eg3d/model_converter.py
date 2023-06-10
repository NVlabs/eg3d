import sys
import pickle

import torch

import dnnlib
import legacy
from torch_utils import misc
from training.triplane import TriPlaneGenerator



assert len(sys.argv) == 3, "Script expects format: `python model_converter.py <Input file> <output file>"

input_f = sys.argv[1]
output_f = sys.argv[2]


# load data
try:
    with dnnlib.util.open_url(input_f, verbose=False) as f:
        data = legacy.load_network_pkl(f)
    print('Done loading model.')
except:
    print('Failed to load model!')

    
# reload data
### only reload G and G_ema.
for module_name in ['G' and 'G_ema']:
    module = data[module_name]
    with torch.no_grad():
        module_new = TriPlaneGenerator(*module.init_args, **module.init_kwargs).eval().requires_grad_(False).to('cpu')
        misc.copy_params_and_buffers(module, module_new, require_all=True)
    module_new.neural_rendering_resolution = module.neural_rendering_resolution
    assert(module.rendering_kwargs == module_new.rendering_kwargs)
    
    data[module_name] = module_new



# Save data
with open(output_f, 'wb') as f:
    pickle.dump(data, f)


print('Successfully converted!')
