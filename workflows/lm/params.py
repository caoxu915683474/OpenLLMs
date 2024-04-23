import sys
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

sys.path.append("../")
from params import (DataArguments, 
                    ModelArguments, 
                    FinetuningArguments, 
                    GeneratingArguments)


class ParamHelper:
    """ ParamHelper """
    def __init__(self) -> None:
        """ __init__ """
        self.setup()
        
    def setup(self) -> None:
        """ setup """
        self.set_args()

    def set_args(self) -> None:
        """ set_args """
        parser = HfArgumentParser((DataArguments, 
                                   ModelArguments, 
                                   Seq2SeqTrainingArguments,
                                   FinetuningArguments,
                                   GeneratingArguments))
        (self.data_args,
         self.model_args,
         self.training_args,
         self.finetuning_args,
         self.generating_args) = parser.parse_args_into_dataclasses()
        
    def get_data_args(self) -> "DataArguments":
        """ get_data_args """
        return self.data_args
    
    def get_model_args(self) -> "ModelArguments":
        """ get_model_args """
        return self.model_args
    
    def get_training_args(self) -> "Seq2SeqTrainingArguments":
        """ get_training_args """
        return self.training_args
    
    def get_finetuning_args(self) -> "FinetuningArguments":
        """ get_finetuning_args """
        return self.finetuning_args
    
    def get_generating_args(self) -> "GeneratingArguments":
        """ get_generating_args """
        return self.generating_args