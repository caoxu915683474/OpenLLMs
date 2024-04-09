import sys
from transformers import Seq2SeqTrainingArguments, TrainerCallback

sys.path.append("../")
from workflows.lm.tokenizer import TokenizerHelper
from workflows.lm.data import DatasetHelper
from workflows.lm.model import ModelHelper
from workflows.lm.trainer import TrainerHelper
from workflows.lm.params import ParamHelper


class LMFlow:
    """ LMFlow: The workflow for Pretrain and Supervised Fine-Tun-ing """
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
        tokenizer_helper = TokenizerHelper(args=self.model_args, template=self.data_args.template)
        self.tokenizer = tokenizer_helper.get_tokenizer()
    
    def set_datasets(self) -> None:
        """ set_datasets """
        dataset_helper = DatasetHelper(args=self.data_args, 
                                       tokenizer=self.tokenizer, 
                                       seed=self.training_args.data_seed, 
                                       do_train=self.training_args.do_train)
        self.dataset = dataset_helper.get_dataset()
        self.train_dataset = self.dataset.get("train_dataset", None)
        self.eval_dataset = self.dataset.get("eval_dataset", None)
    
    def set_model(self) -> None:
        """ set_model """
        self.model = ModelHelper(args=self.model_args,
                                 finetuning_args=self.finetuning_args,
                                 tokenizer=self.tokenizer, 
                                 is_trainable=self.training_args.do_train).get_model()
    
    def set_trainer(self) -> None:
        """ set_trainer """
        self.trainer = TrainerHelper(model=self.model, 
                                     tokenizer=self.tokenizer,
                                     train_dataset=self.train_dataset, 
                                     eval_dataset=self.eval_dataset,
                                     training_args=self.training_args, 
                                     finetuning_args=self.finetuning_args).get_trainer()
    
    def start(self) -> None:
        """ start """
        self.trainer.train()
    
    
if __name__ == '__main__':
    workflow = LMFlow()
    workflow.start()