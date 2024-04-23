import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Union
from datasets import Dataset, IterableDataset

sys.path.append("../")
from data import TEMPLATE
from data.template import SFTTemplate
from data.align import PTAlignWrapper, SFTAlignWrapper
from data.format import SFTFormatWrapper
from data.feature import PTFeatureWrapper, SFTFeatureWrapper
from data.dataset import LMDataset, CVLMDataset

"""
The Dependent Tree of DataFactorys

- DatasetFactory 
    - LMDatasetFactory 
        - PTLMDatasetFactory
        - SFTLMDatasetFactory
            - AlpacaSFTLMDatasetFactory
            - ShareGptSFTLMDatasetFactory
    - CVLMDatasetFactory
        - PTCVLMDatasetFactory
        - SFTCVLMDatasetFactory
"""


@dataclass
class DatasetFactory(ABC):
    """ DatasetFactory """
    factory: str
    load_from: str
    name: str
    cutoff_len: int
    num_shards: int
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.set_dataset()
        self.set_columns()
        
    def __repr__(self) -> str:
        """ __repr__ """
        return self.name
    
    def get_dataset(self) -> Union["Dataset", "IterableDataset"]:
        """ get_dataset """
        return self.dataset
    
    def get_columns(self) -> Union["Dataset", "IterableDataset"]:
        """ get_columns """
        return self.columns
    
    @abstractmethod
    def set_dataset(self) -> None:
        """ set_dataset """
        ...
    
    @abstractmethod
    def set_columns(self) -> None:
        ...
    
    @abstractmethod
    def pipeline(self) -> None:
        """ pipeline """
        ...
        

@dataclass
class LMDatasetFactory(DatasetFactory):
    """ LMDatasetFactory """
    def __post_init__(self) -> None:
        """ __post_init__ """
        super().__post_init__()
        self.modal = "text2text"
        
    def set_dataset(self) -> None:
        """ set_dataset """
        self.dataset = LMDataset(name=self.name, load_from=self.load_from).get_dataset()
        

@dataclass
class PTLMDatasetFactory(LMDatasetFactory):
    """ PTLMDatasetFactory """
    packed: bool
    template: str
    content: Optional[str] = "document"
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        super().__post_init__()
        self.stage = "pt"
        
    def set_columns(self) -> None:
        """ set_columns """
        self.columns = {"content": self.content}
    
    def pipeline(self, tokenizer: "PreTrainedTokenizer", streaming: bool) -> None:
        """ pipeline """
        if streaming: self.dataset = self.dataset.to_iterable_dataset(num_shards=self.num_shards)
        align_wrapper = PTAlignWrapper(columns=self.columns)
        feature_wrapper = PTFeatureWrapper(tokenizer=tokenizer, 
                                           packed=self.packed,
                                           cutoff_len=self.cutoff_len)
        self.dataset = align_wrapper(dataset=self.dataset)
        self.dataset = feature_wrapper(dataset=self.dataset)
        

@dataclass
class SFTLMDatasetFactory(LMDatasetFactory):
    """ SFTLMDatasetFactory """
    packed: bool
    template: str
    reserved_label_len: int
    train_on_prompt: Optional[bool] = False
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        super().__post_init__()
        self.stage = "sft"
        self.tags = {}
    
    def pipeline(self, tokenizer: "PreTrainedTokenizer", streaming: bool) -> None:
        """ pipeline """
        if streaming: self.dataset = self.dataset.to_iterable_dataset(num_shards=self.num_shards)
        align_wrapper = SFTAlignWrapper(columns=self.columns, 
                                        tags=self.tags, 
                                        formatting=self.formatting)
        template = SFTTemplate(**TEMPLATE[self.template])
        format_wrapper = SFTFormatWrapper(template)
        feature_wrapper = SFTFeatureWrapper(tokenizer=tokenizer, 
                                            packed=self.packed,
                                            cutoff_len=self.cutoff_len, 
                                            reserved_label_len=self.reserved_label_len, 
                                            train_on_prompt=self.train_on_prompt)
        self.dataset = align_wrapper(dataset=self.dataset)
        self.dataset = format_wrapper(dataset=self.dataset)
        self.dataset = feature_wrapper(dataset=self.dataset, efficient_eos=template.efficient_eos)
        
        
@dataclass
class AlpacaSFTLMDatasetFactory(SFTLMDatasetFactory):
    """ AlpacaSFTLMDatasetFactory """
    ## columns for the alpaca format
    system: Optional[str] = "system"
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = "history"
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        super().__post_init__()
        self.formatting = "alpaca"
    
    def set_columns(self) -> None:
        """ set_columns """
        self.columns = {"system": self.system, 
                        "prompt": self.prompt, 
                        "query": self.query,  
                        "response": self.response,
                        "history": self.history}

@dataclass
class ShareGptSFTLMDatasetFactory(SFTLMDatasetFactory):
    """ ShareGptSFTLMDatasetFactory """
    ## columns for the sharegpt format
    system: Optional[str] = "system"
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    ## tags for the sharegpt format
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    
    def __post_init__(self):
        """ __post_init__ """
        super().__post_init__()
        self.formatting = "sharegpt"
        self.set_tags()
    
    def set_columns(self) -> None:
        """ set_columns """
        self.columns = {"system": self.system, 
                        "messages": self.messages, 
                        "tools": self.tools}
    
    def set_tags(self) -> None:
        """ set_tags """
        self.tags["role_tag"] = self.role_tag
        self.tags["content_tag"] = self.content_tag
        self.tags["user_tag"] = self.user_tag
        self.tags["assistant_tag"] = self.assistant_tag
        self.tags["observation_tag"] = self.observation_tag
        self.tags["function_tag"] = self.function_tag
        self.tags["system_tag"] = self.system_tag