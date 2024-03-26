import sys
from transformers import PreTrainedTokenizer

sys.path.append("../")
from model.patch import TokenizerPatcher
from model.tokenizer import LMTokenizer
from params.model_args import ModelArguments


class TokenizerHelper:
    """ TokenizerHelper """
    def __init__(self, args: "ModelArguments") -> None:
        """ __init__ """
        self.args = args
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_patcher()
        self.set_tokenizer()
    
    def set_patcher(self) -> None:
        """ set_patcher """
        self.patcher = TokenizerPatcher()
        
    def set_tokenizer(self) -> None:
        """ set_tokenizer """
        tokenizer = LMTokenizer(path=self.args.path, 
                                use_fast_tokenizer=self.args.use_fast_tokenizer, 
                                split_special_tokens=self.args.split_special_tokens, 
                                padding_side=self.args.padding_side, 
                                trust_remote_code=self.args.trust_remote_code, 
                                cache_dir=self.args.cache_dir).load()
        self.tokenizer = self.patcher(tokenizer)
    
    def get_tokenizer(self) -> "PreTrainedTokenizer":
        """ get_tokenizer """
        return self.tokenizer
        