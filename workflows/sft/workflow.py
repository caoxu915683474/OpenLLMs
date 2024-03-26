import sys
from transformers import Seq2SeqTrainingArguments, TrainerCallback

sys.path.append("../")
from train.sft.tokenizer import TokenizerHelper
from train.sft.data import DatasetHelper
from train.sft.model import ModelHelper
from train.sft.trainer import TrainerHelper


class WorkFlow:
    """ WorkFlow """
    def __init__(self) -> None:
        """ __init__ """
        self.setup()

    def setup(self) -> None:
        """ setup """
        self.set_params()
        self.set_tokenizer()
        self.set_datasets()
        self.set_model()
        self.set_trainer()
    
    def set_params(self) -> None:
        """ set_param """
        param_helper = ParamHelper()
        self.data_args = param_helper.get_data_args()
        self.model_args = param_helper.get_model_args()
        self.training_args = param_helper.get_training_args()
        self.finetuning_args = param_helper.get_finetuning_args()
        self.generating_args = param_helper.get_generating_args()
        
    def set_tokenizer(self) -> None:
        """ set_tokenizer """
        tokenizer_helper = TokenizerHelper(args=self.model_args)
        self.tokenizer = tokenizer_helper.get_tokenizer()
    
    def set_datasets(self) -> None:
        """ set_datasets """
        dataset_helper = DatasetHelper(args=self.data_args, tokenizer=self.tokenizer)
        self.dataset = dataset_helper.get_dataset()
        self.train_dataset = self.dataset.get("train_dataset", None)
        self.eval_dataset = self.dataset.get("eval_dataset", None)
    
    def set_model(self) -> None:
        """ set_model """
        model_helper = ModelHelper(args=self.model_args, tokenizer=self.tokenizer)
        self.model = model_helper.get_model()
    
    def set_trainer(self) -> None:
        """ set_trainer """
        trainer_helper = TrainerHelper(model=self.model, 
                                       tokenizer=self.tokenizer,
                                       train_dataset=self.train_dataset, 
                                       eval_dataset=self.eval_dataset,
                                       trainning_args=self.trainning_args, 
                                       finetuning_args=self.finetuning_args)
        self.trainer = trainer_helper.get_trainer()    
    
    def start(self) -> None:
        """ start """
        self.trainer.train()
    
if __name__ == '__main__':
    workflow = WorkFlow()
    workflow.start()