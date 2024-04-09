import sys

sys.path.append("../")
from params.data_args import DataArguments
from params.model_args import ModelArguments
from params.finetuning_args import FinetuningArguments
from params.generating_args import GeneratingArguments

def verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments"):
    """ verify_model_args """
    if model_args.adapter_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")
    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")
        if model_args.adapter_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")
        if model_args.adapter_path is not None and len(model_args.adapter_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")
