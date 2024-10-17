import torch

from facelib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet

import os.path as osp
now_dir = osp.dirname(osp.abspath(__file__))
models_dir = osp.join(osp.dirname(osp.dirname(now_dir)),"..","..","models")
aifsh_dir = osp.join(models_dir,"AIFSH")
hallo_dir = osp.join(aifsh_dir,"HALLO")

def init_parsing_model(model_name='bisenet', half=False, device='cuda'):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth'
        model_path = osp.join(hallo_dir,"facelib/parsing_bisenet.pth")
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
        model_path = osp.join(hallo_dir,"facelib/parsing_parsenet.pth")
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # model_path = load_file_from_url(url=model_url, model_dir='pretrained_models/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
