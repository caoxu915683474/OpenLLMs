from dataclasses import dataclass, field
from typing import Optional
from typing_extensions import Literal


@dataclass
class DataArguments:
    """ DataArguments """
    dataset: Optional[str] = field(default=None, metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."})
    probs: Optional[str] = field(default=None, metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."})
    streaming: Optional[bool] = field(default=False, metadata={"help": "Enable dataset streaming."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    overwrite_cache: Optional[bool] = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."})
    split: Optional[str] = field(default="train", metadata={"help": "Which dataset split to use for training and evaluation."})
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(default="interleave_under", metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."})
    train_on_prompt: Optional[bool] = field(default=False, metadata={"help": "Whether to disable the mask on the prompt or not."})
    sft_packing: Optional[bool] = field(default=False, metadata={"help": "Packing the questions and answers in the supervised fine-tuning stage."})
    cutoff_len: Optional[int] = field(default=1024, metadata={"help": "The cutoff length of the model inputs after tokenization."})
    reserved_label_len: Optional[int] = field(default=1, metadata={"help": "The minimum cutoff length reserved for label after tokenization."})
    seed: Optional[int] = field(default=1234, metadata={"help": "The random seed for dataset."})

    def __post_init__(self) -> None:
        """ __post_init__ """
        self.dataset = [ds.strip() for ds in self.dataset.split(",")] \
                                            if self.dataset is not None else []
        self.probs = [float(prob.strip()) for prob in self.probs.split(",")] \
                                            if self.probs is not None else []