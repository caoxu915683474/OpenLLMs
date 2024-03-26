import gc
import os
from typing import TYPE_CHECKING, Dict, Tuple
import torch
from peft import PeftModel
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList, PreTrainedModel
from transformers.utils import (SAFE_WEIGHTS_NAME,
                                WEIGHTS_NAME,
                                is_torch_bf16_gpu_available,
                                is_torch_cuda_available,
                                is_torch_mps_available,
                                is_torch_npu_available,
                                is_torch_xpu_available)
from transformers.utils.versions import require_version


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False
    

def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32

def get_current_device() -> torch.device:
    """ Gets the current available device. """
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"
    return torch.device(device)

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """ Returns the number of trainable parameters and number of all parameters in the model. """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param