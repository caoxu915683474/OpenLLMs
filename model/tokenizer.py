import sys
from transformers import PreTrainedTokenizer

sys.path.append("../")
from extra.logger import get_logger

logger = get_logger(__name__)


class LMTokenizer:
    """ LMTokenizer """
    def __init__(self, 
                 path: str, 
                 use_fast_tokenizer: bool, 
                 split_special_tokens: bool,
                 padding_side: str, 
                 trust_remote_code: bool, 
                 cache_dir: str) -> None:
        """ __init__ """
        self.path = path
        self.use_fast_tokenizer = use_fast_tokenizer
        self.split_special_tokens = split_special_tokens
        self.padding_side = padding_side
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
    
    def load(self) -> "PreTrainedTokenizer":
        """ load """
        tokenizer = AutoTokenizer.from_pretrained(self.path,
                                                  use_fast=self.use_fast_tokenizer,
                                                  split_special_tokens=self.split_special_tokens,
                                                  padding_side=self.padding_side, 
                                                  trust_remote_code=self.trust_remote_code, 
                                                  cache_dir=self.cache_dir)
        return tokenizer
