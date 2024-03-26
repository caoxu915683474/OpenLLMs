import sys
from typing import Dict, Any
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PretrainedConfig

sys.path.append("../")
from extra.logger import get_logger

logger = get_logger(__name__)


class LModel:
    """ LModel """
    def __init__(self, 
                 path: str, 
                 config: "PretrainedConfig",
                 is_trainable: bool,
                 init_kwargs: Dict[str, Any]) -> None:
        """ __init__ """
        self.path = path
        self.config = config
        self.is_trainable = is_trainable
        self.init_kwargs = init_kwargs
    
    def load(self) -> "PreTrainedModel":
        """ load """
        model = None
        if self.is_trainable and self.use_unsloth:
            from unsloth import FastLanguageModel  # type: ignore
            unsloth_kwargs = {"model_name": self.path,
                              "max_seq_length": self.model_max_length,
                              "dtype": self.compute_dtype,
                              "load_in_4bit": self.quantization_bit == 4,
                              "device_map": {"": get_current_device()},
                              "rope_scaling": getattr(config, "rope_scaling", None)}
            try:
                model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
            except NotImplementedError:
                logger.warning("Unsloth does not support model type {}.".\
                               format(getattr(config, "model_type", None)))
                model_args.use_unsloth = False
            if model_args.adapter_name_or_path:
                model_args.adapter_name_or_path = None
                logger.warning("Unsloth does not support loading adapters.")
        if model is None:    
            model = AutoModelForCausalLM.from_pretrained(self.path, 
                                                         config=config, 
                                                         **self.init_kwargs)
        return tokenizer
