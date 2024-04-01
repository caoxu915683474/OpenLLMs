import sys
from dataclasses import dataclass
from typing import Union
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


@dataclass
class TrainerHelper:
    """ TrainerHelper """
    model:  Union["PreTrainedModel", "nn.Module"]
    tokenizer: "PreTrainedTokenizer"
    train_dataset: Union["Dataset", "IterableDataset"]
    eval_dataset: Union["Dataset", "IterableDataset"]
    training_args: "TrainingArguments"
    finetuning_args: "FinetuningArguments"
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.set_trainer()
        
    def set_data_collator(self) -> None:
        """ set_collator """
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    pad_to_multiple_of=8 \
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
                                 training_args=self.training_args, 
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