import torch
import transformers
import math
from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        assert lora_dim > 0
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        in_feature, out_feature = self.weight.shape
        self.lora_right_weight = torch.nn.Parameter(torch.zeros((out_feature, lora_dim)))
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((lora_dim, in_feature)))

        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        #
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        self.lora_right_weight.data.normal_(mean=0.0, std=0.02)
        self.lora_left_weight.data.zero_()

        raise NotImplementedError
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        output=torch.nn.functional.linear(input, self.weight,self.bias)
        lora_output=torch.matmul(self.lora_left_weight.T, self.lora_right_weight.T)
        output+=lora_output*self.lora_scaling
        return output
        raise NotImplementedError
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
    raise NotImplementedError
    ##############################################################################

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading

    state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name:
            state_dict[name] = param.data
    return state_dict
    raise NotImplementedError
    ########################################################