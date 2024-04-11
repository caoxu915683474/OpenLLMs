import sys
from typing import Dict, Any, Literal
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM

sys.path.append("../")
from extras.logger import get_logger
from extras.misc import get_current_device

logger = get_logger(__name__)


@dataclass
class LModel:
    """ LModel """
    path: str
    config: "PretrainedConfig"
    is_trainable: bool
    use_unsloth: bool
    model_max_length: int
    compute_dtype: Literal[torch.float16, torch.bfloat16, torch.float32]
    quantization_bit: int
    use_adapter: bool
    init_kwargs: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_model()
    
    def set_model(self) -> None:
        """ set_model """
        if self.is_trainable and self.use_unsloth:
            from unsloth import FastLanguageModel  # type: ignore
            unsloth_kwargs = {"model_name": self.path,
                              "max_seq_length": self.model_max_length,
                              "dtype": self.compute_dtype,
                              "load_in_4bit": self.quantization_bit == 4,
                              "device_map": {"": get_current_device()},
                              "rope_scaling": getattr(self.config, "rope_scaling", None)}
            try:
                self.model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
            except NotImplementedError:
                logger.warning("Unsloth does not support model type {}.".\
                               format(getattr(config, "model_type", None)))
                self.use_unsloth = False
            if self.use_adapter:
                self.use_adapter = False
                logger.warning("Unsloth does not support loading adapters.")
        else:   
            self.model = AutoModelForCausalLM.from_pretrained(self.path, 
                                                              config=self.config, 
                                                              **self.init_kwargs)
        
    def get_model(self) -> "PreTrainedModel":
        """ get_model """
        return self.model
