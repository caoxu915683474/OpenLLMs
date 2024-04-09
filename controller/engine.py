from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Dict, 
                    List, Literal, Optional, Sequence, Union)
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]
    
@dataclass
class BaseEngine(ABC):
    """ BaseEngine """
    tokenizer: "PreTrainedTokenizer"
    model: "PreTrainedModel"
    
    def __post_init__(self, 
                      model_args: "ModelArguments",
                      data_args: "DataArguments",
                      finetuning_args: "FinetuningArguments",
                      generating_args: "GeneratingArguments") -> None: 
        """ __post_init__ """
        ...
    
    @abstractmethod
    async def start(self) -> None: 
        ...
    
    @abstractmethod
    async def chat(self,
                   messages: Sequence[Dict[str, str]],
                   system: Optional[str] = None,
                   tools: Optional[str] = None,
                   **input_kwargs) -> List["Response"]: 
        ...
    
    @abstractmethod
    async def stream_chat(self,
                          messages: Sequence[Dict[str, str]],
                          system: Optional[str] = None,
                          tools: Optional[str] = None,
                          **input_kwargs) -> AsyncGenerator[str, None]: 
        ...

    @abstractmethod
    async def get_scores(self,
                         batch_input: List[str],
                         **input_kwargs) -> List[float]: 
        ...


@dataclass
class HuggingfaceEngine(BaseEngine):
    def __init__(self) -> None:
        """ __init__ """
        