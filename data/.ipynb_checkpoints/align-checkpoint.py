import sys
from typing import Union, Dict, List, Any, Optional


class SFTAlignWrapper:
    """ SFTAlignWrapper """
    def __init__(self, 
                 columns: List[str], 
                 tags: Dict[str, Any], 
                 formatting: str) -> None:
        """ __init__ """
        self.columns = columns
        self.tags = tags
        self.formatting = formatting
    
    def alpaca(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """ alpaca_align """
        outputs = {"prompt": [], "response": [], "system": []}
        for i in range(len(examples[self.colmns["prompt"]])):
            prompt = []
            if self.colmns["history"] and isinstance(examples[self.colmns["history"]][i], list):
                for old_prompt, old_response in examples[self.colmns["history"]][i]:
                    prompt.append({"role": ROLE["USER"], "content": old_prompt})
                    prompt.append({"role": ROLE["ASSISTANT"], "content": old_response})
            content = []
            if self.colmns["prompt"] and examples[self.colmns["prompt"]][i]:
                content.append(examples[self.colmns["prompt"]][i])
            if self.colmns["query"] and examples[self.colmns["query"]][i]:
                content.append(examples[self.colmns["query"]][i])
            prompt.append({"role": ROLE["USER"], "content": "\n".join(content)})
            if self.colmns["response"] and isinstance(examples[self.colmns["response"]][i], list):
                response = [{"role": ROLE["ASSISTANT"], "content": content} for content in examples[self.colmns["response"]][i]]
            elif self.colmns["response"] and isinstance(examples[self.colmns["response"]][i], str):
                response = [{"role": ROLE["ASSISTANT"], "content": examples[self.colmns["response"]][i]}]
            else:
                response = []
            system = examples[self.colmns["system"]][i] if self.colmns["system"] else ""
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
        for i, messages in enumerate(examples[self.colmns["messages"]]):
            if self.tags["system_tag"] and messages[0][self.tags["role_tag"]] == self.tags["system_tag"]:
                system = messages[0][self.tags["content_tag"]]
                messages = messages[1:]
            else:
                system = examples[self.colmns["system"]][i] if self.colmns["system"] else ""
            messages = messages[: len(messages) // 2 * 2]  # should be multiples of 2
            if len(messages) == 0:
                continue
            aligned_messages = []
            for turn_idx, message in enumerate(messages):
                if message[self.tags["role_tag"]] not in accept_tags[turn_idx % 2]:
                    raise ValueError("Invalid role tag in {}.".format(messages))
                aligned_messages.append({"role": tag_mapping[message[self.tags["role_tag"]]], 
                                         "content": message[self.tags["content_tag"]]})
            tools = examples[self.colmns["tools"]][i] if self.colmns["tools"] else ""
            outputs["prompt"].append(aligned_messages[:-1])
            outputs["response"].append(aligned_messages[-1:])
            outputs["system"].append(system)
            outputs["tools"].append(tools)
        return outputs
        
    def __call__(self, dataset: Union["Dataset", "IterableDataset"]) -> Union["Dataset", "IterableDataset"]:
        """ __call__ """
        features_dict = {"prompt": [{"role": {"dtype": "string", "_type": "Value"}, 
                                    "content": {"dtype": "string", "_type": "Value"}}],
                         "response": [{"role": {"dtype": "string", "_type": "Value"}, 
                                       "content": {"dtype": "string", "_type": "Value"}}],
                         "system": {"dtype": "string", "_type": "Value"},
                         "tools": {"dtype": "string", "_type": "Value"}}
        remove_columns = list(next(iter(dataset)).keys())
        return dataset.map(eval("self.{}".format(self.formatting)), 
                           batched=True, 
                           remove_columns=remove_columns, 
                           features=features)