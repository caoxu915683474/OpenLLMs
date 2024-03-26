import sys
from types import MethodType
from functools import partial
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, interleave_datasets
from transformers import (AutoTokenizer, 
                          PreTrainedTokenizer, 
                          DataCollatorForSeq2Seq, 
                          PreTrainedTokenizerBase)
from typing import Union, List, Sequence, Tuple, Dict, Any

sys.path.append("../")
from data.dataset import LMDataset
from data.info import DatasetAttr, get_dataset_list
from data.align import AlignWrapper
from data.format import FormatWrapper
from data.feature import FeatureWrapper
from params.data_args import DataArguments
IGNORE_INDEX=-100

def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", 
                                          trust_remote_code=True,
                                          use_fast=False,
                                          split_special_tokens=False,
                                          padding_side="right")
tokenizer.add_special_tokens({'eos_token': "<|im_end|>"})
tokenizer.pad_token = tokenizer.eos_token
patch_tokenizer(tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                       pad_to_multiple_of=8,
                                       label_pad_token_id=IGNORE_INDEX)

data_args = DataArguments()
data_args.dataset = "alpaca_data_zh_51k_100p, alpaca_data_zh_51k_100p2"
data_args.interleave_probs = [0.3, 0.7]
dataset_attrs = get_dataset_list(data_args.dataset)
all_datasets = []
for dataset_attr in dataset_attrs:
    conf = dataset_attr.get_confs()
    dataset = LMDataset(name=conf["dataset"]["name"], load_from=conf["dataset"]["load_from"])
    align_wrapper = AlignWrapper(stage=conf["task"]["stage"], 
                                 formatting=conf["task"]["formatting"], 
                                 conf=conf["align"])
    format_wrapper = FormatWrapper(template=conf["format"]["template"])
    feature_wrapper = FeatureWrapper(tokenizer=tokenizer, 
                                     cutoff_len=conf["task"]["cutoff_len"], 
                                     reserved_label_len=conf["task"]["reserved_label_len"], 
                                     train_on_prompt=conf["task"]["train_on_prompt"], 
                                     efficient_eos=format_wrapper.template.efficient_eos)
    dataset = dataset.load()
    dataset = align_wrapper(dataset)
    dataset = format_wrapper(dataset)
    dataset = feature_wrapper(dataset)
    all_datasets.append(dataset)
# dataset = concatenate_datasets(all_datasets)

dataset = interleave_datasets(datasets=all_datasets,
                              probabilities=data_args.interleave_probs,
                              seed=data_args.seed,
                              stopping_strategy="first_exhausted" \
                                      if data_args.mix_strategy.endswith("under") else "all_exhausted")

dataloader = DataLoader(dataset, 
                        batch_size=2,
                        num_workers=1,
                        collate_fn=data_collator,
                        pin_memory=True)
for batch in dataloader:
    print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False))
    print("-" * 100)
    print(tokenizer.decode([id_ for id_ in batch["labels"][0] if id_ != -100], skip_special_tokens=False))
    input()

        
