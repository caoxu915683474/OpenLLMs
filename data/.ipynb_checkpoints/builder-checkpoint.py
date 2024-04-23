import sys
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from transformers import PreTrainedTokenizer
from datasets import (Dataset,
                      IterableDataset,
                      concatenate_datasets,
                      interleave_datasets)

sys.path.append("../")
from data import DATA_INFO
from data.factory import (DatasetFactory,
                          PTLMDatasetFactory, 
                          AlpacaSFTLMDatasetFactory, 
                          ShareGptSFTLMDatasetFactory)

        
@dataclass
class DataBuilder:
    """ DataBuilder """
    tokenizer: "PreTrainedTokenizer"
    dataset: List[str]
    mix_strategy: str
    probs: List[float]
    template: str
    streaming: bool
    seed: int
    context: Any
    num_shards: int
    
    def __post_init__(self):
        """ __post_init__ """
        with self.context:
            self.set_factorys()
            self.construction()
    
    def set_factorys(self) -> None:
        """ set_factorys """
        factorys: List["DatasetFactory"] = []
        for name in self.dataset:
            if name not in DATA_INFO:
                raise ValueError("Undefined dataset {} in {}.".format(name, "DATA_INFO"))
            data_info = DATA_INFO[name]
            factory = eval(data_info["factory"])(**data_info, template=self.template, num_shards=self.num_shards)
            factorys.append(factory)
        self.factorys = factorys
    
    def construction(self) -> None:
        """ construction """
        datasets = []
        for factory in self.factorys:
            factory.pipeline(tokenizer=self.tokenizer, streaming=self.streaming)
            dataset = factory.get_dataset()
            datasets.append(dataset)
        if len(datasets) == 1:
            self.dataset= datasets[0]
        elif self.mix_strategy == "concat":
            self.dataset = concatenate_datasets(datasets)
        elif self.mix_strategy.startswith("interleave"):
            self.dataset = interleave_datasets(datasets=datasets,
                                               probabilities=self.probs,
                                               seed=self.seed,
                                               stopping_strategy="first_exhausted" \
                                               if self.mix_strategy.endswith("under") else "all_exhausted")
            
    def get_dataset(self) -> Dict[str, Union["Dataset", "IterableDataset"]]:
        """ get_dataset """
        return self.dataset
        
        