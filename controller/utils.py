

def create_custom_optimizer(model: "PreTrainedModel", 
                            training_args: "Seq2SeqTrainingArguments", 
                            finetuning_args: "FinetuningArguments", 
                            max_steps: int) -> Optional["torch.optim.Optimizer"]:
    """ create_custom_optimizer """
    if finetuning_args.use_galore:
        optimizer = create_galore_optimizer(model, 
                                            training_args, 
                                            finetuning_args, 
                                            max_steps)
        return optimizer
    if finetuning_args.loraplus_lr_ratio is not None:
        optimizer = create_loraplus_optimizer(model, 
                                              training_args, 
                                              finetuning_args)
        return optimizer
    
