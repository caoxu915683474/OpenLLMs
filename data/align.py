import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Any, Literal
from datasets import Dataset, IterableDataset, Features

sys.path.append("../")
from data import ROLE


@dataclass
class AlignWrapper(ABC):
    """ AlignWrapper """
    columns: List[str]
    tags: Dict[str, Any] = None
    formatting: Literal["alpaca", "sharegpt"] = None
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.set_features_dict()
    
    @abstractmethod
    def set_features_dict(self) -> None:
        """ set_features_dict """
        ...
    
    @abstractmethod
    def align(self) -> Any:
        """ align """
        ...
    
    def __call__(self, dataset: Union["Dataset", "IterableDataset"]) -> Union["Dataset", "IterableDataset"]:
        """ __call__ """
        features = Features.from_dict(self.features_dict)
        remove_columns = list(next(iter(dataset)).keys())
        return dataset.map(self.align, 
                           batched=True, 
                           remove_columns=remove_columns, 
                           features=features)


class PTAlignWrapper(AlignWrapper):
    """ PTAlignWrapper """
    def set_features_dict(self) -> None:
        """ set_features_dict """
        self.features_dict = {"text": {"dtype": "string", "_type": "Value"}}
    
    def align(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """ align """
        outputs = {"text": []}
        for i in range(len(examples[self.columns["content"]])):
            outputs["text"].append(examples[self.columns["content"]][i])
        return outputs

class SFTAlignWrapper(AlignWrapper):
    """ SFTAlignWrapper """
    def set_features_dict(self) -> None:
        """ set_features_dict """
        self.features_dict = {"prompt": [{"role": {"dtype": "string", "_type": "Value"}, 
                                    "content": {"dtype": "string", "_type": "Value"}}],
                              "response": [{"role": {"dtype": "string", "_type": "Value"}, 
                                       "content": {"dtype": "string", "_type": "Value"}}],
                              "system": {"dtype": "string", "_type": "Value"},
                              "tools": {"dtype": "string", "_type": "Value"}}
    
    def align(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """ align """
        if self.formatting == "alpaca":
            return self.alpaca(examples)
        elif self.formatting == "sharegpt":
            return self.sharegpt(examples)
    
    def alpaca(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """ alpaca_align """
        outputs = {"prompt": [], "response": [], "system": [], "tools": []}
        for i in range(len(examples[self.columns["prompt"]])):
            prompt = []
            if self.columns["history"] and self.columns["history"] in examples \
                and isinstance(examples[self.columns["history"]][i], list):
                for old_prompt, old_response in examples[self.columns["history"]][i]:
                    prompt.append({"role": ROLE["USER"], "content": old_prompt})
                    prompt.append({"role": ROLE["ASSISTANT"], "content": old_response})
            content = []
            if self.columns["prompt"] and self.columns["prompt"] in examples \
                and examples[self.columns["prompt"]][i]:
                content.append(examples[self.columns["prompt"]][i])
            if self.columns["query"] and self.columns["query"] in examples \
                and examples[self.columns["query"]][i]:
                content.append(examples[self.columns["query"]][i])
            prompt.append({"role": ROLE["USER"], "content": "\n".join(content)})
            if self.columns["response"] and self.columns["response"] in examples \
                and isinstance(examples[self.columns["response"]][i], list):
                response = [{"role": ROLE["ASSISTANT"], "content": content} \
                                for content in examples[self.columns["response"]][i]]
            elif self.columns["response"] and self.columns["response"] in examples \
                and isinstance(examples[self.columns["response"]][i], str):
                response = [{"role": ROLE["ASSISTANT"], "content": examples[self.columns["response"]][i]}]
            else:
                response = []
            system = examples[self.columns["system"]][i] if self.columns["system"] \
                                                            and self.columns["system"] in examples else ""
            system = ""
            outputs["prompt"].append(prompt)
            outputs["response"].append(response)
            outputs["system"].append(system)
            outputs["tools"].append("")
        return outputs
    
    def sharegpt(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """ sharegpt_align """
        outputs = {"prompt": [], "response": [], "system": [], "tools": []}
        tag_mapping = {self.tags["user_tag"]: ROLE["USER"],
                       self.tags["assistant_tag"]: ROLE["ASSISTANT"],
                       self.tags["observation_tag"]: ROLE["OBSERVATION"],
                       self.tags["function_tag"]: ROLE["FUNCTION"],
                       self.tags["system_tag"]: ROLE["SYSTEM"]}
        odd_tags = (self.tags["user_tag"], self.tags["observation_tag"])
        even_tags = (self.tags["assistant_tag"], self.tags["function_tag"])
        accept_tags = (odd_tags, even_tags)
        for i, messages in enumerate(examples[self.columns["messages"]]):
            if self.tags["system_tag"] and messages[0][self.tags["role_tag"]] == self.tags["system_tag"]:
                system = messages[0][self.tags["content_tag"]]
                messages = messages[1:]
            else:
                system = examples[self.columns["system"]][i] if self.columns["system"] else ""
            messages = messages[: len(messages) // 2 * 2]  # should be multiples of 2
            if len(messages) == 0:
                continue
            aligned_messages = []
            for turn_idx, message in enumerate(messages):
                if message[self.tags["role_tag"]] not in accept_tags[turn_idx % 2]:
                    raise ValueError("Invalid role tag in {}.".format(messages))
                aligned_messages.append({"role": tag_mapping[message[self.tags["role_tag"]]], 
                                         "content": message[self.tags["content_tag"]]})
            tools = examples[self.columns["tools"]][i] if self.columns["tools"] else ""
            outputs["prompt"].append(aligned_messages[:-1])
            outputs["response"].append(aligned_messages[-1:])
            outputs["system"].append(system)
            outputs["tools"].append(tools)
        return outputs