import sys
from typing import Dict, Union, Optional, Tuple, Any, List
from torch import nn
from datasets import Dataset, IterableDataset
from transformers import (PreTrainedModel,
                          Seq2SeqTrainer, 
                          TrainingArguments, 
                          TrainerCallback, 
                          PreTrainedTokenizer, 
                          EvalPrediction)

sys.path.append("../")
from params.finetuning_args import FinetuningArguments
from conroller.utils import create_custom_optimizer


class LMTrainer(Seq2SeqTrainer):
    """ LMTrainer """
    def __init__(self, 
                 model: Union["PreTrainedModel", "nn.Module"], 
                 tokenizer: "PreTrainedTokenizer",
                 trainning_args: "TrainingArguments",
                 finetuning_args: "FinetuningArguments", 
                 data_collator: "DataCollator",
                 train_dataset: Union["Dataset", "IterableDataset"],
                 eval_dataset: Union["Dataset", "IterableDataset"],
                 callbacks: Optional[List["TrainerCallback"]] = None,
                 compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
                 **kwargs) -> None:
        super().__init__(model=model, 
                         args=trainning_args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         callbacks=callbacks,
                         compute_metrics=compute_metrics,
                         **kwargs)
        self.finetuning_args = finetuning_args
    
    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """ create_optimizer_and_scheduler """
        self.optimizer = create_custom_optimzer(self.model, 
                                                self.args, 
                                                self.finetuning_args, 
                                                num_training_steps)
        if self.optimizer is None:
            self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
    
    def prediction_step(self, 
                        model: "torch.nn.Module", 
                        inputs: Dict[str, Union[torch.Tensor, Any]], 
                        prediction_loss_only: bool, 
                        ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ 
        Removes the prompt part in the generated tokens.
        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            input_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if input_len > label_len:
                inputs["labels"] = self.pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > input_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :input_len]
        loss, generated_tokens, _ = super().prediction_step(model, 
                                                            inputs, 
                                                            prediction_loss_only=prediction_loss_only, 
                                                            ignore_keys=ignore_keys)
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :input_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()
        return loss, generated_tokens, labels
            
    
    def pad_tensors_to_target_len(self, 
                                  src_tensor: torch.Tensor, 
                                  tgt_tensor: torch.Tensor) -> torch.Tensor:
        """ _pad_tensors_to_target_len """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory
        
        