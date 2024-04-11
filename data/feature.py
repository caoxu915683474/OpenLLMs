import sys
from functools import partial
from itertools import chain
from typing import Union, List, Sequence, Tuple, Dict, Any
from transformers import PreTrainedTokenizer

sys.path.append("../")
from extras.constant import IGNORE_INDEX


class PTFeatureWrapper:
    """ PTFeatureWrapper """
    def __init__(self, 
                 tokenizer: "PreTrainedTokenizer", 
                 packed: bool, 
                 cutoff_len: int) -> None:
        """ __init__ """
        self.tokenizer = tokenizer
        self.packed = packed
        self.cutoff_len = cutoff_len
    
    def unpacked_func(self, examples: Dict[str, Any]) -> Dict[str, List[Any]]:
        """ unpacked_func """
        model_inputs = {}
        texts = [text.strip() + self.tokenizer.eos_token for text in examples["text"]]
        result = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.cutoff_len)
        labels = []
        for label in result["input_ids"]:
            label_length = len(label)
            for i in range(label_length):
                if label[i] == self.tokenizer.eos_token_id and i != (label_length - 1):
                    label[i] = IGNORE_INDEX
            labels.append(label)
        model_inputs["input_ids"] = result["input_ids"]
        model_inputs["attention_mask"] = result["attention_mask"]
        model_inputs["labels"] = labels
        return model_inputs
    
    def packed_func(self, examples: Dict[str, Any]) -> Dict[str, List[Any]]:
        """ packed_func """
        model_inputs = {}
        texts = [text.strip() + self.tokenizer.eos_token for text in examples["text"]]
        texts_tokenized = self.tokenizer(texts, add_special_tokens=False)
        texts_tokenized = {k: list(chain(*texts_tokenized[k])) for k in texts_tokenized.keys()}
        total_length = len(texts_tokenized[list(texts_tokenized.keys())[0]])
        block_size = self.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                  for k, t in texts_tokenized.items()}
        model_inputs["input_ids"] = result["input_ids"]
        model_inputs["attention_mask"] = result["attention_mask"]
        model_inputs["labels"] = result["input_ids"].copy()
        return model_inputs        
    
    def __call__(self, dataset: Union["Dataset", "IterableDataset"]) -> Union["Dataset", "IterableDataset"]:
        """ __call__ """
        column_names = list(next(iter(dataset)).keys())
        map_func = self.packed_func if self.packed else self.unpacked_func
        dataset = dataset.map(map_func,
                              batched=True,
                              remove_columns=column_names)
        return dataset
    

class SFTFeatureWrapper:
    """ SFTFeatureWrapper """
    def __init__(self, 
                 tokenizer: "PreTrainedTokenizer", 
                 packed: bool,
                 cutoff_len: int, 
                 reserved_label_len: int, 
                 train_on_prompt: int) -> None:
        """ __init__ """
        self.tokenizer = tokenizer
        self.packed = packed
        self.cutoff_len = cutoff_len
        self.reserved_label_len = reserved_label_len
        self.train_on_prompt = train_on_prompt
    
    def packed_func(self, 
                    examples: Dict[str, Any], 
                    efficient_eos: bool)  -> Dict[str, List[Any]]:
        """ packed_func """
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for i in range(len(examples["messages_elements"])):
            messages_elements = examples["messages_elements"][i]
            encoded_messages = []
            for elements in messages_elements:
                token_ids = []
                for element in elements:
                    token_ids += self.tokenizer.encode(element, add_special_tokens=False)
                encoded_messages.append(token_ids)
            encoded_pairs = self.make_pairs(encoded_messages)
            for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
                if self.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and efficient_eos: 
                    source_mask = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
        if efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
        total_length = len(input_ids)
        block_size = self.cutoff_len
        total_length = (total_length // block_size) * block_size
        for i in range(0, total_length, block_size):
            if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
                model_inputs["input_ids"].append(input_ids[i : i + block_size])
                model_inputs["attention_mask"].append([1] * block_size)
                model_inputs["labels"].append(labels[i : i + block_size])
        return model_inputs

    def unpacked_func(self, 
                      examples: Dict[str, Any], 
                      efficient_eos: bool)  -> Dict[str, List[Any]]:
        """ unpacked_func """
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["messages_elements"])):
            messages_elements = examples["messages_elements"][i]
            encoded_messages = []
            for elements in messages_elements:
                token_ids = []
                for element in elements:
                    token_ids += self.tokenizer.encode(element, add_special_tokens=False)
                encoded_messages.append(token_ids)
            input_ids, labels = [], []
            encoded_pairs = self.make_pairs(encoded_messages)
            for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
                if self.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and efficient_eos: 
                    source_mask = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                if efficient_eos:
                    input_ids += [self.tokenizer.eos_token_id]
                    labels += [self.tokenizer.eos_token_id]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
        return model_inputs

    def make_pairs(self, encoded_messages: Sequence[List[int]]) -> List[Any]:
        """ make_pairs """
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= self.cutoff_len: break
            max_source_len, max_target_len = self.infer_max_len(source_len=len(encoded_messages[i]),
                                                                target_len=len(encoded_messages[i + 1]),
                                                                total_length=total_length)
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))
        return encoded_pairs
    
    def infer_max_len(self, 
                      source_len: int, 
                      target_len: int, 
                      total_length: int) -> Tuple[int, int]:
        """ infer_max_len """
        max_len = self.cutoff_len - total_length
        max_target_len = int(max_len * (target_len / (source_len + target_len)))
        max_target_len = max(max_target_len, self.reserved_label_len)
        max_source_len = max_len - max_target_len
        return max_source_len, max_target_len

    def __call__(self,
                 dataset: Union["Dataset", "IterableDataset"], 
                 efficient_eos: bool) -> Union["Dataset", "IterableDataset"]:
        """ __call__ """
        column_names = list(next(iter(dataset)).keys())
        map_func = self.packed_func if self.packed else self.unpacked_func
        map_func = partial(map_func, efficient_eos=efficient_eos)
        dataset = dataset.map(map_func, 
                              batched=True, 
                              remove_columns=column_names)
        return dataset
    
        
