import sys
from dataclasses import dataclass
from typing import Dict, Any
from transformers import AutoConfig

sys.path.append("../")
from extras.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LMConfig:
    """ LMConfig """
    path: str
    init_kwargs: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.set_config()
    
    def set_config(self) -> None:
        """ set_config """
        self.config = AutoConfig.from_pretrained(self.path, **self.init_kwargs)
    
    def get_config(self) -> "PretrainedConfig":
        """ get_config """
        return self.config