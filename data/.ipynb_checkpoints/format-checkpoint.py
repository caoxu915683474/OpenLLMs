import sys
from typing import Union
from datasets import Dataset, IterableDataset

sys.path.append("../")
from data.template import SFTTemplate


class SFTFormatWrapper:
    """ FormatWrapper """
    def __init__(self, template: "SFTTemplate") -> None:
        """ __init__ """
        self.template = template
    
    def __call__(self, dataset: Union["Dataset", "IterableDataset"]) -> Union["Dataset", "IterableDataset"]:
        """ __call__ """
        dataset = dataset.map(self.template.write, batched=True)
        return dataset
    