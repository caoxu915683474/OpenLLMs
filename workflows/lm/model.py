import sys
from dataclasses import dataclass
from transformers import (PreTrainedModel, 
                          PreTrainedTokenizer, 
                          PretrainedConfig, 
                          PreTrainedTokenizerBase, 
                          GPTQConfig, 
                          BitsAndBytesConfig)

sys.path.append("../")
from params import ModelArguments, FinetuningArguments
from model.model import LModel
from model.config import LMConfig
from model.utils import register_autoclass
from model.patch import ConfigPatcher, ModelPatcher
from model.adapter import LMAdapterWapper


@dataclass
class ModelHelper:
    """ ModelHelper """
    args: "ModelArguments"
    finetuning_args: "FinetuningArguments"
    tokenizer: "PreTrainedTokenizer"
    is_trainable: bool

    def __post_init__(self) -> None:
        """ __post_init__ """
        self.init_kwargs = {"trust_remote_code": self.args.trust_remote_code,
                            "revision": self.args.model_revision,
                            "cache_dir": self.args.cache_dir}
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
                                            use_cache=self.args.use_cache,
                                            is_trainable=self.is_trainable, 
                                            quantization_bit=self.args.quantization_bit,
                                            double_quantization=self.args.double_quantization,
                                            quantization_type=self.args.quantization_type,
                                            export_quantization_dataset=self.args.export_quantization_dataset,
                                            export_quantization_maxlen=self.args.export_quantization_maxlen,
                                            export_quantization_nsamples=self.args.export_quantization_nsamples,
                                            export_quantization_bit=self.args.export_quantization_bit,
                                            offload_folder=self.args.offload_folder,
                                            init_kwargs=self.init_kwargs)
    
    def set_config(self) -> None:
        """ set_config """
        self.config = LMConfig(path=self.args.path, init_kwargs=self.init_kwargs).get_config()
        self.config = self.config_patcher(self.config)
        self.init_kwargs = self.config_patcher.get_init_kwargs()
    
    def set_adapter(self) -> None:
        """ set_adapter """
        self.adapter_wrapper = LMAdapterWapper(adapter_path=self.args.adapter_path, 
                                               finetuning_type=self.finetuning_args.finetuning_type,
                                               pure_bf16=self.finetuning_args.pure_bf16,
                                               use_llama_pro=self.finetuning_args.use_llama_pro,
                                               num_layer_trainable=self.finetuning_args.num_layer_trainable,
                                               use_dora=self.finetuning_args.use_dora,
                                               offload_folder=self.args.offload_folder, 
                                               create_new_adapter=self.finetuning_args.create_new_adapter, 
                                               lora_rank=self.finetuning_args.lora_rank,
                                               lora_target=self.finetuning_args.lora_target, 
                                               lora_alpha=self.finetuning_args.lora_alpha, 
                                               lora_dropout=self.finetuning_args.lora_dropout, 
                                               use_rslora=self.finetuning_args.use_rslora, 
                                               additional_target=self.finetuning_args.additional_target, 
                                               use_unsloth=self.args.use_unsloth, 
                                               is_trainable=self.is_trainable)
    
    def set_model_patcher(self) -> None:
        """ set_model_patcher """
        self.model_patcher = ModelPatcher(tokenizer=self.tokenizer, 
                                          is_trainable=self.is_trainable,
                                          resize_vocab=self.args.resize_vocab, 
                                          upcast_layernorm=self.args.upcast_layernorm, 
                                          disable_gradient_checkpointing=self.args.disable_gradient_checkpointing,
                                          upcast_lmhead_output=self.args.upcast_lmhead_output)
    
    def set_model(self) -> None:
        """ set_model """        
        self.model = LModel(path=self.args.path, 
                            config=self.config, 
                            is_trainable=self.is_trainable,
                            use_unsloth=self.args.use_unsloth,
                            model_max_length=self.args.model_max_length,
                            compute_dtype=self.args.compute_dtype,
                            quantization_bit=self.args.quantization_bit,
                            use_adapter=self.args.use_adapter,
                            init_kwargs=self.init_kwargs).get_model()
        self.model = self.model_patcher(self.model)
        register_autoclass(config=self.config, model=self.model, tokenizer=self.tokenizer)
        self.model = self.adapter_wrapper(self.model)

    def get_model(self) -> "PreTrainedModel":
        """ get_model """
        return self.model
