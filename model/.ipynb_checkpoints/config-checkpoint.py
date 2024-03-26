import sys
from typing import Dict, Any
from transformers import PretrainedConfig

sys.path.append("../")
from extra.logger import get_logger

logger = get_logger(__name__)


class LMConfig:
    """ LMConfig """
    def __init__(self, path: str, init_kwargs: Dict[str, Any]) -> None:
        """ __init__ """
        self.path = path
        self.init_kwargs = init_kwargs
    
    def load(self) -> "PretrainedConfig":
        """ load """
        config = AutoAutoConfigTokenizer.from_pretrained(self.path, **self.init_kwargs)
        return config
