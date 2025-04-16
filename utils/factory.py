import torch
from models.genguide import GeNGuide

def get_model(model_name, args):
    name = model_name.lower()
    if 'genguide' in name:
        return GeNGuide(args)
    else:
        assert 0
