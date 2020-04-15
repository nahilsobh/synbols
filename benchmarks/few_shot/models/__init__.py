from .protonet import protonet
from .MAML import MAML

def get_model(exp_dict):
    if exp_dict["benchmark"] == 'few_shot': 
        if exp_dict["model"] == 'protonet':
            return protonet(exp_dict) 
        elif exp_dict["model"] == 'MAML':
            return MAML(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])