import sys
import math
import random
from types import MethodType
from contextlib import nullcontext
from typing import Dict, Optional, Any
import torch
from datasets import concatenate_datasets, interleave_datasets
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import (PreTrainedModel, 
                          PreTrainedTokenizer, 
                          PretrainedConfig, 
                          PreTrainedTokenizerBase, 
                          GPTQConfig, 
                          BitsAndBytesConfig)

sys.path.append("../")
from data.dataset import LMDataset
from data.info import DatasetAttr, get_dataset_list
from extra.logger import get_logger
from extra.misc import get_current_device
from extra.packages import is_flash_attn2_available
from model.llama_patch import apply_llama_patch
from model.mixtral_patch import patch_mixtral_replace_moe_impl

logger = get_logger(__name__)
SUPPORTED_CLASS_FOR_S2ATTN = ["llama"]


class TokenizerPatcher:
    """ TokenizerPatcher """
    def __init__(self) -> None:
        """ __init__ """
        ...
    
    def __call__(self, tokenizer: "PreTrainedTokenizer") -> "PreTrainedTokenizer":
        """ __call__ """
        if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
            tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)


class ConfigPatcher:
    """ ConfigPatcher """
    def __init__(self, 
                 tokenizer: "PreTrainedTokenizer",
                 compute_dtype: Optional[torch.float16, torch.bfloat16, torch.float32],
                 rope_scaling: bool,
                 model_max_length: int,
                 shift_attn: bool,
                 flash_attn: bool,
                 low_cpu_mem_usage bool, 
                 device_map: Dict[str, Any],
                 is_trainable: bool,
                 quantization_dataset: str,
                 export_quantization_maxlen: int,
                 export_quantization_nsamples: int,
                 export_quantization_bit: int,
                 offload_folder: str,
                 init_kwargs: Dict[str, Any]) -> None:
        """ __init__ """
        self.tokenizer = tokenizer
        self.compute_dtype = compute_dtype
        self.rope_scaling = rope_scaling
        self.model_max_length = model_max_length
        self.shift_attn = shift_attn
        self.flash_attn = flash_attn
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.is_trainable = is_trainable
        self.quantization_dataset = quantization_dataset
        self.export_quantization_maxlen = export_quantization_maxlen
        self.export_quantization_nsamples = export_quantization_nsamples
        self.export_quantization_bit = export_quantization_bit
        self.offload_folder = offload_folder
        self.init_kwargs = init_kwargs
    
    def configure_attn_implementation(self) -> None:
        """ configure_attn_implementation """
        if self.flash_attn:
            if is_flash_attn2_available():
                logger.info("Using FlashAttention-2 for faster training and inference.")
                self.init_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                logger.warning("FlashAttention2 is not installed.")
                self.init_kwargs["attn_implementation"] = None
        else:
            self.init_kwargs["attn_implementation"] = "eager"
    
    def configure_rope(self, config: "PretrainedConfig") -> None:
        """ configure_rope """
        if self.rope_scaling is None:
            return
        if not hasattr(config, "rope_scaling"):
            logger.warning("Current model does not support RoPE scaling.")
            return
        if self.is_trainable:
            if self.rope_scaling == "dynamic":
                logger.warning("Dynamic NTK scaling may not work well with fine-tuning. "
                               "See: https://github.com/huggingface/transformers/pull/24653")
            current_max_length = getattr(config, "max_position_embeddings", None)
            if current_max_length and self.model_max_length > current_max_length:
                scaling_factor = float(math.ceil(self.model_max_length / current_max_length))
            else:
                logger.warning("Input length is smaller than max length. Consider increase input length.")
                scaling_factor = 1.0
        else:
            scaling_factor = 2.0
        setattr(config, "rope_scaling", {"type": self.rope_scaling, "factor": scaling_factor})
        logger.info("Using {} scaling strategy and setting scaling factor to {}".\
                    format(self.rope_scaling, scaling_factor))
    
    def configure_longlora(self, config: "PretrainedConfig") -> None:
        """ configure_longlora """
        if not self.is_trainable or not self.shift_attn:
            return
        if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:
            setattr(config, "group_size_ratio", 0.25)
            apply_llama_patch()
            logger.info("Using shift short attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shift short attention.")
    
    def get_quantization_dataset(self):
        """ 
        Inspired by: https://github.com/huggingface/optimum/blob/v1.16.0/optimum/gptq/data.py#L133
        TODO: remove tokenizer.decode() https://github.com/huggingface/optimum/pull/1600 
        """
        dataset_attrs = get_dataset_list(self.quantization_dataset)
        all_datasets = []
        for dataset_attr in dataset_attrs: 
            conf = dataset_attr.get_confs()
            dataset = LMDataset(name=conf["dataset"]["name"], 
                                load_from=conf["dataset"]["load_from"])
            all_datasets.append(dataset)
        dataset = concatenate_datasets(all_datasets)
        maxlen = self.export_quantization_maxlen
        samples = []
        for _ in range(self.export_quantization_nsamples):
            while True:
                sample_idx = random.randint(0, len(dataset) - 1)
                sample: Dict[str, torch.Tensor] = self.tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
                if sample["input_ids"].size(1) >= maxlen:
                    break  # TODO: fix large maxlen
            word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
            input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
            samples.append(self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True))
        return samples
    
    def configure_quantization(self, config: "PretrainedConfig"):
        """
        Priority: PTQ-quantized (training) > AutoGPTQ (export) > Bitsandbytes (training)
        """
        if getattr(config, "quantization_config", None):  # ptq
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            init_kwargs["device_map"] = {"": get_current_device()}
            quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
            quant_method = quantization_config.get("quant_method", "")
            if quant_method == "gptq":
                quantization_config["use_exllama"] = False  # disable exllama
            if quant_method == "aqlm":
                require_version("transformers>=4.39.0.dev0", 
                                "To fix: pip install git+https://github.com/huggingface/transformers.git")
                require_version("aqlm>=1.1.0", "To fix: pip install aqlm[gpu]>=1.1.0")
                quantization_config["bits"] = 2
            quant_bits = quantization_config.get("bits", "?")
            logger.info("Loading {}-bit {}-quantized model.".format(quant_bits, quant_method.upper()))
        elif self.export_quantization_bit is not None:  # auto-gptq
            require_version("optimum>=1.16.0", "To fix: pip install optimum>=1.16.0")
            require_version("auto_gptq>=0.5.0", "To fix: pip install auto_gptq>=0.5.0")
            from accelerate.utils import get_max_memory
            if getattr(config, "model_type", None) == "chatglm":
                raise ValueError("ChatGLM model is not supported.")
            quantization_dataset = self.get_quantization_dataset()
            self.init_kwargs["quantization_config"] = GPTQConfig(bits=self.export_quantization_bit,
                                                                 tokenizer=self.tokenizer,
                                                                 dataset=quantization_dataset)
            init_kwargs["device_map"] = "auto"
            init_kwargs["max_memory"] = get_max_memory()
            logger.info("Quantizing model to {} bit.".format(self.export_quantization_bit))
    

    def __call__(self, config: "PretrainedConfig") -> "PretrainedConfig":
        """ __call__ """
        if getattr(config, "model_type", None) == "qwen":
            for dtype_name, dtype in [("fp16", torch.float16), 
                                      ("bf16", torch.bfloat16), 
                                      ("fp32", torch.float32)]:
                setattr(config, dtype_name, self.compute_dtype == dtype)
        self.configure_attn_implementation()
        self.configure_rope(config)
        self.configure_longlora(config)
        self.configure_quantization(config)
        if self.use_cache and not self.is_trainable:
            setattr(config, "use_cache", True)
            logger.info("Using KV cache for faster generation.")
        init_kwargs["torch_dtype"] = self.compute_dtype
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = self.low_cpu_mem_usage
            if "device_map" not in init_kwargs:  # quant models cannot use auto device map
                init_kwargs["device_map"] = self.device_map or {"": get_current_device()}
            if init_kwargs["device_map"] == "auto":
                init_kwargs["offload_folder"] = self.offload_folder


class ModelPatcher:
    """ ModelPatcher """
    def __init__(self, 
                 tokenizer: "PreTrainedTokenizer", 
                 is_trainabl: bool,
                 resize_vocab: bool, 
                 upcast_layernorm: bool, 
                 disable_gradient_checkpointing: bool) -> None:
        """ __init__ """
        self.tokenizer = tokenizer
        self.is_trainable = is_trainable
        self.resize_vocab = resize_vocab
        self.upcast_layernorm = upcast_layernorm
        self.disable_gradient_checkpointing = disable_gradient_checkpointing
    
    def noisy_mean_initialization(self, embed_weight: torch.Tensor, num_new_tokens: int):
        """ noisy_mean_initialization """
        embedding_dim = embed_weight.size(1)
        avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
        noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
        noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
        embed_weight[-num_new_tokens:] = avg_weight + noise_weight
    
    def resize_embedding_layer(self, model: "PreTrainedModel") -> None:
        """ resize_embedding_layer """
        if is_deepspeed_zero3_enabled():
            import deepspeed  # type: ignore
            params = [model.get_input_embeddings().weight]
            if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
                params.append(model.get_output_embeddings().weight)
            context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
        else:
            context_maybe_zero3 = nullcontext()
        with context_maybe_zero3:
            current_embedding_size = model.get_input_embeddings().weight.size(0)
        if len(self.tokenizer) > current_embedding_size:
            if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
                logger.warning("Current model does not support resizing token embeddings.")
                return
            model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
            with context_maybe_zero3:
                new_embedding_size = model.get_input_embeddings().weight.size(0)
                num_new_tokens = new_embedding_size - current_embedding_size
                self.noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
                self.noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)
            logger.info("Resized token embeddings from {} to {}.".\
                        format(current_embedding_size, new_embedding_size))
    
    def prepare_model_for_training(self, model: "PreTrainedModel") -> None:
        """ 
        Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) add the upcasting of the lm_head in fp32
        Inspired by: https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/other.py#L72
        """
        if self.upcast_layernorm:
            logger.info("Upcasting layernorm weights in float32.")
            for name, param in model.named_parameters():
                if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                    param.data = param.data.to(torch.float32)
        if not self.disable_gradient_checkpointing:
            if not getattr(model, "supports_gradient_checkpointing", False):
                logger.warning("Current model does not support gradient checkpointing.")
            else:
                # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
                # According to: https://github.com/huggingface/transformers/issues/28339
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
                model.enable_input_require_grads()
                setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
                logger.info("Gradient checkpointing enabled.")
        if hasattr(model, output_layer_name) and model_args.upcast_lmhead_output:
            def fp32_forward_post_hook(module: torch.nn.Module, 
                                       args: Tuple[torch.Tensor], 
                                       output: torch.Tensor):
                return output.to(torch.float32)
            logger.info("Upcasting lm_head outputs in float32.")
            output_layer = getattr(model, output_layer_name)
            if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
                output_layer.register_forward_hook(fp32_forward_post_hook)
    
    def __call__(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """ __call__ """
        if "GenerationMixin" not in str(model.generate.__func__):
            model.generate = MethodType(PreTrainedModel.generate, model)
        if getattr(model.config, "model_type", None) == "chatglm":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])
        if self.resize_vocab:
            self.resize_embedding_layer(model)
        if self.is_trainable:
            self.prepare_model_for_training(model)
        if getattr(model.config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules  # type: ignore
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
            if self.is_trainable:
                patch_mixtral_replace_moe_impl()
        try:
            model.add_model_tags(["Model-Cooker_Xu"])
        except Exception:
            logger.warning("Cannot properly tag the model.")
            
        
        