import os
import sys
import inspect
from typing import Union
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, IterableDataset

sys.path.append("../")
from extra.logger import get_logger
from data.info import DatasetAttr

logger = get_logger(__name__)


class BasicDataset(ABC):
    """ BasicDataset """
    name: str
    load_from: str
    
    @abstractmethod
    def load(self) -> Union["Dataset", "IterableDataset"]:
        ...


class LMDataset(BasicDataset):
    """ LMDataset """        
    def load(self) -> Union["Dataset", "IterableDataset"]:
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
        dataset = load_dataset("json",
                               split="train",
                               data_files=data_files,
                               download_mode="force_redownload")
        return dataset


class CVLMDataset(BasicDataset):
    """ CVLMDataset """
    def load(self) -> Union["Dataset", "IterableDataset"]:
        """ load """
        ...
