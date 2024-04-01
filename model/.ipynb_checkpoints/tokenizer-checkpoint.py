import sys
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, AutoTokenizer

sys.path.append("../")
from data import TEMPLATE
from extras.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LMTokenizer:
    """ LMTokenizer """
    path: str
    template: str
    use_fast_tokenizer: bool
    split_special_tokens: bool
    padding_side: str
    trust_remote_code: bool
    cache_dir: str
        
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_template()
        self.set_tokenizer()
        self.set_special_tokens()
    
    def set_template(self) -> None:
        """ set_tempalte """
        self.template = TEMPLATE[self.template]
    
    def set_tokenizer(self) -> None:
        """ set_tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,
                                                       use_fast=self.use_fast_tokenizer,
                                                       split_special_tokens=self.split_special_tokens,
                                                       padding_side=self.padding_side, 
                                                       trust_remote_code=self.trust_remote_code, 
                                                       cache_dir=self.cache_dir)
    
    def set_special_tokens(self) -> None:
        """ set_special_tokens """
        if self.template["replace_eos"] and self.template["stop_words"]:
            self.tokenizer.add_special_tokens({"eos_token": self.template["stop_words"][0]})
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_tokenizer(self) -> "PreTrainedTokenizer":
        """ get_tokenizer """
        return self.tokenizer