import sys
from dataclasses import dataclass


@dataclass
class ChatHelper:
    """ ChatHelper """
    model:  Union["PreTrainedModel", "nn.Module"]
    tokenizer: "PreTrainedTokenizer"
    finetuning_args: "FinetuningArguments"
    generating_args: "GeneratingArguments"
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_chater()
    
    def set_chater(self) -> None:
        """ set_chater """
        # TODO
    
    def get_chater(self) -> "Chater":
        """ get_chater """
        return self.chater