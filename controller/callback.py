import os
import sys
import time
import json
from datetime import timedelta
from transformers import (TrainerCallback, 
                          TrainerControl, 
                          TrainerState,
                          TrainingArguments)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length

sys.path.append("../")
from extras.constant import LOG_FILE_NAME
from extras.logger import get_logger

logger = get_logger(__name__)


class LogCallback(TrainerCallback):
    """ LogCallback """
    def __init__(self, runner=None) -> None:
        """ __init__ """
        self.runner = runner
        self.in_trainning =False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
    
    def timing(self):
        """ timing """
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
    
    def on_train_begin(self, args: "TrainingArguments", 
                       state: "TrainerState", 
                       control: "TrainerControl", 
                       **kwargs):
        """ Event called at the beginning of training. """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps
            if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)) and args.overwrite_output_dir:
                logger.warning("Previous log file in this folder will be deleted.")
                os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))
    
    def on_train_end(self, 
                     args: "TrainingArguments", 
                     state: "TrainerState", 
                     control: "TrainerControl", 
                     **kwargs):
        """ Event called at the end of training. """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0
    
    def on_substep_end(self, 
                       args: "TrainingArguments", 
                       state: "TrainerState", 
                       control: "TrainerControl", 
                       **kwargs):
        """ Event called at the end of an substep during gradient accumulation. """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True
    
    def on_step_end(self, 
                    args: "TrainingArguments", 
                    state: "TrainerState", 
                    control: "TrainerControl", 
                    **kwargs):
        """ Event called at the end of a training step. """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True
    
    def on_evaluate(self, 
                    args: "TrainingArguments", 
                    state: "TrainerState", 
                    control: "TrainerControl", 
                    **kwargs):
        """ Event called after an evaluation phase. """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0
    
    def on_predict(self, 
                   args: "TrainingArguments", 
                   state: "TrainerState", 
                   control: "TrainerControl", 
                   *other, 
                   **kwargs):
        """ Event called after a successful prediction. """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0
    
    def on_log(self, 
               args: "TrainingArguments", 
               state: "TrainerState", 
               control: "TrainerControl", 
               **kwargs) -> None:
        """ Event called after logging the last logs. """
        if not state.is_local_process_zero:
            return
        
        logger.info(state.log_history[-1])
        logs = dict(current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    global_step=state.log_history[-1].get("step", None),
                    loss=state.log_history[-1].get("loss", None),
                    eval_loss=state.log_history[-1].get("eval_loss", None),
                    predict_loss=state.log_history[-1].get("predict_loss", None),
                    reward=state.log_history[-1].get("reward", None),
                    learning_rate=state.log_history[-1].get("learning_rate", None),
                    num_input_tokens_seen=state.log_history[-1].get("num_input_tokens_seen", None),
                    train_tokens_per_second=state.log_history[-1].get("train_tokens_per_second", None),
                    epoch=state.log_history[-1].get("epoch", None),
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) \
                                                    if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time)
        if self.runner is None:
            logger.info("{{'epoch': {:.2f}, 'gloabl_step': {}, 'loss': {:.4f}, 'learning_rate': {:2.4e}, 'tokens/s': {:.2f}}}"\
                            .format(logs["epoch"] or 0, logs["global_step"] or 0, logs["loss"] or 0, logs["learning_rate"] or 0, logs["train_tokens_per_second"] or 0))
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")
    
    def on_prediction_step(self, 
                           args: "TrainingArguments", 
                           state: "TrainerState", 
                           control: "TrainerControl", 
                           **kwargs):
        """ Event called after a prediction step. """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()
