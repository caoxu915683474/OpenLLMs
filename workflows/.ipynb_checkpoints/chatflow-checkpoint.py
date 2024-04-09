
import sys

sys.path.append("../")
from workflows.chat.tokenizer import TokenizerHelper
from workflows.chat.data import DatasetHelper
from workflows.chat.model import ModelHelper
from workflows.chat.params import ParamHelper


class ChatFlow:
    """ RLFlow: The workflow for Chat Usage """
    def __init__(self) -> None:
        setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_params()
        self.set_tokenizer()
        self.set_datasets()
        self.set_model()
    
    def set_params(self) -> None:
        """ set_param """
        param_helper = ParamHelper()
        self.data_args = param_helper.get_data_args()
        self.model_args = param_helper.get_model_args()
        self.training_args = param_helper.get_training_args()
        self.finetuning_args = param_helper.get_finetuning_args()
        self.generating_args = param_helper.get_generating_args()
    
    def set_tokenizer(self) -> None:
        """ set_tokenizer """
        tokenizer_helper = TokenizerHelper(args=self.model_args, template=self.data_args.template)
        self.tokenizer = tokenizer_helper.get_tokenizer()
    
    def set_model(self) -> None:
        """ set_model """
        ...
    
    def set_chater() -> None:
        """ set_chater """
        self.chater = ChatHelper()
    
    def start(self) -> None:
        """ start """
        self.chater.chat()
    
if __name__ == '__main__':
    workflow = ChatFlow()
    workflow.start()