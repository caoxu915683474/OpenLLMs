import sys
from typing import Union, Any
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer
from datasets import Dataset, IterableDataset

sys.path.append("../")
from params.data_args import DataArguments
from data.builder import DataBuilder 


@dataclass
class DatasetHelper:
    """ DatasetHelper """
    args: "DataArguments"
    tokenizer: "PreTrainedTokenizer"
    context: Any
    seed: int
    do_train: bool
    num_shards: int
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_data_builder()
        self.set_dataset()
        self.split_dataset()
    
    def set_data_builder(self) -> None:
        """ set_data_builder """
        self.data_builder = DataBuilder(tokenizer=self.tokenizer,
                                        dataset=self.args.dataset, 
                                        mix_strategy=self.args.mix_strategy,
                                        probs=self.args.probs,
                                        template=self.args.template,
                                        streaming=self.args.streaming,
                                        seed=self.seed, 
                                        context=self.context,
                                        num_shards=self.num_shards)
    
    def set_dataset(self) -> None:
        """ set_dataset """
        self.dataset = self.data_builder.get_dataset()
    
    def split_dataset(self) -> None:
        """ split_dataset """
        if self.do_train:
            if self.args.val_size > 1e-6:  # Split the dataset
                if self.args.streaming:
                    val_set = self.dataset.take(int(data_args.val_size))
                    train_set = self.dataset.skip(int(data_args.val_size))
                    self.dataset = self.dataset.shuffle(buffer_size=self.args.buffer_size, seed=self.seed)
                    self.dataset = {"train_dataset": train_set, "eval_dataset": val_set}
                else:
                    val_size = int(self.args.val_size) if self.args.val_size > 1 else self.args.val_size
                    self.dataset = self.dataset.train_test_split(test_size=val_size, seed=self.seed)
                    self.dataset = {"train_dataset": self.dataset["train"], "eval_dataset": self.dataset["test"]}
            else:
                if self.args.streaming:
                    self.dataset = self.dataset.shuffle(buffer_size=self.args.buffer_size, seed=self.seed)    
                self.dataset = {"train_dataset": self.dataset}
        else:  # do_eval or do_predict
            self.dataset = {"eval_dataset": self.dataset}
    
    def get_dataset(self) -> Union["Dataset", "IterableDataset"]:
        """ get_dataset """
        return self.dataset