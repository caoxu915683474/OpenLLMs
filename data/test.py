import sys
from torch.utils.data import DataLoader
from transformers import (DataCollatorForSeq2Seq, 
                          HfArgumentParser, 
                          Seq2SeqTrainingArguments, 
                          DataCollatorForLanguageModeling)

sys.path.append("../")
from extras.constant import IGNORE_INDEX
from params import DataArguments, ModelArguments
from data.builder import DataBuilder
from model.patch import TokenizerPatcher
from model.tokenizer import LMTokenizer


if __name__ == '__main__':
    parser = HfArgumentParser((DataArguments, ModelArguments, Seq2SeqTrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    patcher = TokenizerPatcher()
    tokenizer = LMTokenizer(path=model_args.path, 
                            template=data_args.template,
                            use_fast_tokenizer=model_args.use_fast_tokenizer, 
                            split_special_tokens=model_args.split_special_tokens, 
                            padding_side=model_args.padding_side, 
                            trust_remote_code=model_args.trust_remote_code, 
                            cache_dir=model_args.cache_dir).get_tokenizer()
    tokenizer = patcher(tokenizer)
    data_builder = DataBuilder(tokenizer=tokenizer,
                               dataset=data_args.dataset,
                               mix_strategy=data_args.mix_strategy,
                               probs=data_args.probs,
                               template=data_args.template,
                               streaming=data_args.streaming,
                               seed=training_args.seed,
                               context=training_args.main_process_first(),
                               num_shards=data_args.num_shards)
    dataset = data_builder.get_dataset()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           pad_to_multiple_of=8,
                                           label_pad_token_id=IGNORE_INDEX)
    dataloader = DataLoader(dataset, 
                            batch_size=2,
                            num_workers=1,
                            collate_fn=data_collator,
                            pin_memory=True)
    for batch in dataloader:
        assert len(batch["input_ids"][0]) == len(batch["labels"][0])
        assert len(batch["input_ids"][1]) == len(batch["labels"][1])
        print("".join(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False).split()))
        print("-" * 100)
        print(tokenizer.decode([id_ for id_ in batch["labels"][0] if id_ != -100], skip_special_tokens=False))
        input()