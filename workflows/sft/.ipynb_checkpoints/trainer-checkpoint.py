import sys
from torch import nn
from datasets import Dataset, IterableDataset
from transformers import (Trainer, 
                          TrainingArguments, 
                          DataCollatorForSeq2Seq, 
                          PreTrainedModel, 
                          PreTrainedTokenizer)

sys.path.append("../")
from extras.constant import IGNORE_INDEX
from controller.train import LMTrainer
from params.finetuning_args import FinetuningArguments
from controller.metric import LMMetrics
from controller.callback import LogCallback


class TrainerHelper:
    """ TrainerHelper """
    def __init__(self,
                 model:  Union["PreTrainedModel", "Module"],
                 tokenizer: "PreTrainedTokenizer",
                 train_dataset: Union["Dataset", "IterableDataset"],
                 eval_dataset: Union["Dataset", "IterableDataset"],
                 trainning_args: "TrainingArguments"
                 finetuning_args: "FinetuningArguments") -> None:
        """ __init__ """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.trainning_args = trainning_args
        self.finetuning_args = finetuning_args
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.set_trainer()
        
    def set_data_collator(self) -> None:
        """ set_collator """
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    pad_to_multiple_of=8 \ # for shift short attention
                                                        if self.tokenizer.padding_side == "right" else None,
                                                    label_pad_token_id=IGNORE_INDEX)
    
    def set_metrics(self) -> None:
        """ set_metrics """
        self.metrics = LMMetrics(tokenizer=self.tokenizer)
    
    def set_callbacks(self) -> None:
        """ set_callbacks """
        self.callbacks = [LogCallback()]
    
    def set_trainer(self) -> None:
        """ set_trainer """
        self.set_data_collator()
        self.set_metrics()
        self.set_callbacks()
        self.trainer = LMTrainer(model=self.model, 
                                 tokenizer=self.tokenizer,
                                 trainning_args=self.trainning_args, 
                                 finetuning_args=self.finetuning_args, 
                                 data_collator=self.data_collator, 
                                 train_dataset=self.train_dataset, 
                                 eval_dataset=self.eval_dataset, 
                                 callbacks=self.callbacks, 
                                 compute_metrics=self.compute_metrics \
                                     if self.training_args.predict_with_generate else None)
        
    def get_trainer(self) -> "Trainer":
        """ get_trainer """
        return self.trainer