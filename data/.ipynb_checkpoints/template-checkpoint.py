import sys
from itertools import chain
from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional

sys.path.append("../")
from data import ROLE
from params.data_args import DataArguments
from extra.logger import get_logger
from extra.constant import IGNORE_INDEX


logger = get_logger(__name__)


@dataclass
class SubTemper(ABC):
    """ SubTemper """
    pattern: str
    
    @abstractmethod
    def apply(self, **kwargs) -> str:
        """ apply """
        ...


@dataclass
class StringSubTemper(SubTemper):
    """ StringSubTemper """
    def apply(self, **kwargs) -> str:
        """ apply """
        for name, value in kwargs.items():
            locals()[name] = value
        text = eval("f'" + self.pattern + "'")
        return text


@dataclass
class SFTTemplate:
    """ SFTTemplate """
    user: Optional[str] = None
    assistant: Optional[str] = None
    separator: Optional[str] = None
    system: Optional[str] = None
    stop_words: List[str] = None
    efficient_eos: bool = False
    replace_eos: bool = False
    force_system: bool = False
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.user_temper = StringSubTemper(self.user) if self.user else None
        self.assistant_temper = StringSubTemper(self.assistant) if self.assistant else None
    
    def write(self, examples) -> Dict[str, Any]:
        """ format """
        outputs = {"messages_elements": []}
        for i in range(len(examples["prompt"])):
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
                continue
            messages = examples["prompt"][i] + examples["response"][i]
            messages_elements = []
            for j, message in enumerate(messages):
                elements = []
                if j == 0 and (self.system and self.force_system):
                    elements.append(self.system)
                elif j > 0 and j % 2 == 0:
                    elements.append(self.separator)
                if message["role"] == ROLE["USER"]:
                    user_text = self.user_temper.apply(content=message["content"], idx=str(j // 2))
                    elements.append(user_text)
                elif message["role"] == ROLE["ASSISTANT"]:
                    assistant_text = self.assistant_temper.apply(content=message["content"])
                    elements.append(assistant_text)
                else:
                    raise NotImplementedError("Unexpected role: {}".format(message["role"]))
                messages_elements.append(elements)
            outputs["messages_elements"].append(messages_elements)
        return outputs
