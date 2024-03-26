import sys
from transformers import Seq2SeqTrainingArguments

sys.path.append("../")
from params.data_args import DataArguments
from params.model_args import ModelArguments
from params.finetuning_args import FinetuningArguments
from params.generating_args import 


class ParamHelper:
    """ ParamHelper """
    def __init__(self) -> None:
        """ __init__ """
        self.setup()
        
    def setup(self) -> None:
        """ setup """
        self.set_data_args()
        self.set_model_args()
        self.set_training_args()
        self.set_finetuning_args()
        self.set_generating_args()
        
    def set_data_args(self) -> None:
        """ set_data_args """
        self.data_args = DataArguments()
    
    def set_model_args(self) -> None:
        """ set_model_args """
        self.model_args = ModelArguments()
        
    def set_training_args(self) -> None:
        """ set_training_args """
        self.training_args = Seq2SeqTrainingArguments()
    
    def set_finetuning_args(self) -> None:
        """ set_finetuning_args """
        self.finetuning_args = FinetuningArguments()
    
    def set_generating_args(self) -> None:
        """ set_generating_args """
        self.generating_args = GeneratingArguments()
    
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