import sys
from transformers import (PreTrainedModel, 
                          PreTrainedTokenizer, 
                          PretrainedConfig, 
                          PreTrainedTokenizerBase, 
                          GPTQConfig, 
                          BitsAndBytesConfig)

sys.path.append("../")
from params.model_args import ModelArguments
from model.model import LModel
from model.config import LMConfig
from model.utils import register_autoclass
from model.patch import ConfigPatcher, ModelPatcher
from model.adapter import LMAdapterWapper


class ModelHelper:
    """ ModelHelper """
    def __init__(self, 
                 args: "ModelArguments", 
                 tokenizer: "PreTrainedTokenizer", 
                 is_trainable: bool) -> None:
        """ __init__ """
        self.tokenizer = tokenizer
        self.args = args
        self.is_trainable = is_trainable
        self.init_kwargs = {"trust_remote_code": self.args.trust_remote_code,
                            "revision": self.args.model_revision,
                            "cache_dir": self.args.cache_dir}
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.setup()
    
    def setup(self) -> None:
        """ setup """
        self.set_config_patcher()
        self.set_config()
        self.set_adapter()
        self.set_model_patcher()
        self.set_model()
        
    def set_config_patcher(self) -> None:
        """ set_config_patcher """
        self.config_patcher = ConfigPatcher(tokenizer=self.tokenizer,
                                            compute_dtype=self.args.compute_dtype,
                                            rope_scaling=self.args.rope_scaling, 
                                            model_max_length=self.args.model_max_length,
                                            shift_attn=self.args.shift_attn, 
                                            flash_attn=self.args.flash_attn, 
                                            low_cpu_mem_usage=self.args.low_cpu_mem_usage, 
                                            device_map=self.args.device_map, 
                                            is_trainable=self.is_trainable, 
                                            quantization_dataset=self.args.quantization_dataset,
                                            export_quantization_maxlen=self.args.export_quantization_maxlen,
                                            export_quantization_nsamples=self.args.export_quantization_nsamples,
                                            export_quantization_bit=self.args.export_quantization_bit,
                                            offload_folder=self.args.offload_folder,
                                            init_kwargs=self.init_kwargs)
    
    def set_config(self) -> None:
        """ set_config """
        config_patcher = ConfigPatcher(tokenizer=self.tokenizer,
                                       compute_dtype=self.args.compute_dtype,
                                       rope_scaling=self.args.rope_scaling, 
                                       model_max_length=self.args.model_max_length,
                                       shift_attn=self.args.shift_attn, 
                                       flash_attn=self.args.flash_attn, 
                                       low_cpu_mem_usage=self.args.low_cpu_mem_usage, 
                                       device_map=self.args.device_map, 
                                       is_trainable=self.is_trainable, 
                                       quantization_dataset=self.args.quantization_dataset,
                                       export_quantization_maxlen=self.args.export_quantization_maxlen,
                                       export_quantization_nsamples=self.args.export_quantization_nsamples,
                                       export_quantization_bit=self.args.export_quantization_bit,
                                       offload_folder=self.args.offload_folder,
                                       init_kwargs=self.init_kwargs)
        config = LMConfig(path=self.args.path, init_kwargs=self.init_kwargs).load()
        self.config = config_patcher(self.config)
    
    def set_adapter(self) -> None:
        """ set_adapter """
        self.adapter_wrapper = LMAdapterWapper(adapter_name_or_path=self.args.adapter_name_or_path, 
                                               finetuning_type=self.finetuning_type, 
                                               pure_bf16=self.args.pure_bf16, 
                                               use_llama_pro=self.args.use_llama_pro, 
                                               num_layer_trainable=self.args.num_layer_trainable, 
                                               use_dora=self.args.use_dora, 
                                               offload_folder=self.args.offload_folder, 
                                               is_trainable=self.is_trainable)
    
    def set_model_patcher(self) -> None:
        """ set_model_patcher """
        self.model_patcher = ModelPatcher(tokenizer=self.tokenizer, 
                                          is_trainable=self.is_trainable,
                                          resize_vocab=self.args.resize_vocab, 
                                          upcast_layernorm=self.args.upcast_layernorm, 
                                          disable_gradient_checkpointing=self.args.disable_gradient_checkpointing)
    
    def set_model(self) -> None:
        """ set_model """        
        model = LModel(path=self.args.path, 
                       config=self.config, 
                       is_trainable=self.is_trainable,
                       init_kwargs=self.init_kwargs).load()
        model = self.model_patcher(model)
        register_autoclass(config=self.config, model=self.model, tokenizer=self.tokenizer)
        self.model = self.adapter_wrapper(model)

    def get_model(self) -> "PreTrainedModel":
        """ get_model """
        return self.model