import os
import sys
import inspect
from typing import Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, IterableDataset

sys.path.append("../")
from extras.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BasicDataset(ABC):
    """ BasicDataset """
    name: str
    load_from: str
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.load()
    
    @abstractmethod
    def load(self) -> None:
        ...
    
    def get_dataset(self) -> Union["Dataset", "IterableDataset"]:
        """ get_dataset """
        return self.dataset

@dataclass
class LMDataset(BasicDataset):
    """ LMDataset """        
    def load(self) -> None:
        """ load """
        logger.info("Loading dataset {}...".format(self.name))
        data_files = []
        local_path: str = self.load_from
        if os.path.isdir(local_path):
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):
            data_files.append(local_path)
        else:
            raise ValueError("File not found.")
        self.dataset = load_dataset("json",
                                    split="train",
                                    data_files=data_files,
                                    download_mode="force_redownload")


class CVLMDataset(BasicDataset):
    """ CVLMDataset """
    def load(self) -> None:
        """ load """
        ...
