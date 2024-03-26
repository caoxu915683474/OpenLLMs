import sys
import torch
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import PreTrainedModel
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

sys.path.append("../")
from extra.logger import get_logger
from model.utils import find_all_linear_modules, find_expanded_modules


logger = get_logger(__name__)


class LMAdapterWapper:
    """ LMAdapter """
    def __init__(self, 
                 adapter_name_or_path: str,
                 finetuning_type: str,
                 pure_bf16: bool,
                 use_llama_pro: bool,
                 num_layer_trainable: int,
                 use_dora: bool,
                 offload_folder: str,
                 is_trainable: bool) -> None:
        """ __init__ """
        self.adapter_name_or_path = adapter_name_or_path
        self.finetuning_type = finetuning_type
        self.pure_bf16 = pure_bf16
        self.use_llama_pro = use_llama_pro
        self.num_layer_trainable = num_layer_trainable
        self.use_dora = use_dora
        self.offload_folder = offload_folder
        self.is_trainable = is_trainable
    
    def no_adapter(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ no_adapter """
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model
    
    def full_param(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ full_param """
        logger.info("Fine-tuning method: Full")
        if not self.pure_bf16:
            model = model.float()
        return model
    
    def freeze(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ freeze """
        logger.info("Fine-tuning method: Freeze")
        num_layers = (getattr(model.config, "num_hidden_layers", None) 
                      or getattr(model.config, "num_layers", None) 
                      or getattr(model.config, "n_layer", None))
        if not num_layer:
            raise ValueError("Current model does not support freeze tuning.")
        if self.use_llama_pro:
            if num_layers % self.num_layer_trainable != 0:
                raise ValueError("`num_layers` {} should be divisible by `num_layer_trainable` {}.".\
                                 format(num_layers, finetuning_args.num_layer_trainable))
            stride = num_layers // self.num_layer_trainable
            trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
        elif self.num_layer_trainable > 0:
            # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = range(num_layers - self.num_layer_trainable, num_layers)
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = range(-self.num_layer_trainable)
        freeze_modules = {"all"}
        for name, _ in model.named_modules():
            if ".0." in name:
                freeze_modules.add(name.split(".0.")[-1].split(".")[0])
        trainable_layers = []
        for module_name in self.name_module_trainable:
            if module_name not in freeze_modules:
                raise ValueError("Module {} is not found, please choose from {}".format(module_name, ", ".join(freeze_modules)))
            for idx in trainable_layer_ids:
                trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))
        for name, param in model.named_parameters():
            if any(trainable_layer in name for trainable_layer in trainable_layers):
                if not self.pure_bf16:
                    param.data = param.data.to(torch.float32)
            else:
                param.requires_grad_(False)
        logger.info("Set trainable layers: {}".format(",".join(map(str, trainable_layer_ids))))
    
    def lora(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ lora """
        logger.info("Fine-tuning method: {}".format("DoRA" if self.use_dora else "LoRA"))
        adapter_to_resume = None
        if self.adapter_name_or_path is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable.
                assert len(self.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                is_mergeable = False
            if is_deepspeed_zero3_enabled():
                assert len(self.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            if (self.is_trainable and not self.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = self.adapter_name_or_path[:-1]
                adapter_to_resume = self.adapter_name_or_path[-1]
            else:
                adapter_to_merge = self.adapter_name_or_path
            for adapter in adapter_to_merge:
                model: "LoraModel" = PeftModel.from_pretrained( model, adapter, offload_folder=self.offload_folder)
                model = model.merge_and_unload()
            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))
            if adapter_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(model, 
                                                  adapter_to_resume, 
                                                  is_trainable=is_trainable, 
                                                  offload_folder=self.offload_folder)
        if self.is_trainable and adapter_to_resume is None:  # create new lora weights while training
            if len(self.lora_target) == 1 and self.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model)
            else:
                target_modules = self.lora_target
            if self.use_llama_pro:
                target_modules = find_expanded_modules(model, target_modules, self.num_layer_trainable)
            if self.use_dora and getattr(model, "quantization_method", None) is not None:
                if getattr(model, "quantization_method", None) != "bitsandbytes":
                    raise ValueError("DoRA is not compatible with PTQ-quantized models.")
            peft_kwargs = {"r": self.lora_rank,
                           "target_modules": target_modules,
                           "lora_alpha": self.lora_alpha,
                           "lora_dropout": self.lora_dropout,
                           "use_rslora": self.use_rslora}
            if self.use_unsloth:
                from unsloth import FastLanguageModel  # type: ignore
                unsloth_peft_kwargs = {"model": model, "max_seq_length": self.model_max_length}
                model = FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
            else:
                lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                         inference_mode=False,
                                         modules_to_save=self.additional_target,
                                         use_dora=self.use_dora,
                                         **peft_kwargs)
                model = get_peft_model(model, lora_config)
        if not self.pure_bf16:
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                param.data = param.data.to(torch.float32)
        if model_args.adapter_name_or_path is not None:
            logger.info("Loaded adapter(s): {}".format(",".join(self.adapter_name_or_path)))
        return model
    
    def __call__(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ __call__ """
        if (not self.is_trainable) and self.adapter_name_or_path is None:
            model = self.no_adapter(model)
        elif self.finetuning_type == "full" and self.is_trainable:
            model = self.full_param(model)
        elif self.finetuning_type == "freeze" and self.is_trainable:
            model = self.freeze(model)
        elif self.finetuning_type == "lora":
            model = self.lora(model)
        return model