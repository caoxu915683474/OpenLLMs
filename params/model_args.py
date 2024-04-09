import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArguments:
    """ Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer. """
    path: str = field(metadata={"help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."})
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."})
    resize_vocab: bool = field(default=False, metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."})
    split_special_tokens: bool = field(default=False, metadata={"help": "Whether or not the special tokens should be split during the tokenization process."})
    padding_side: str = field(default="left", metadata={"help": "The padding side of data pieces in batch right or left."})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code resources."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."})
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(default=None, metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."})
    upcast_layernorm: bool = field(default=False, metadata={"help": "Whether or not to upcast the layernorm weights in fp32."})
    upcast_lmhead_output: bool = field(default=False, metadata={"help": "Whether or not to upcast the output of lm_head in fp32."})
    flash_attn: bool = field(default=False, metadata={"help": "Enable FlashAttention-2 for faster training."})
    shift_attn: bool = field(default=False, metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."})
    low_cpu_mem_usage: bool = field(default=True, metadata={"help": "Whether or not to use memory-efficient model loading."})
    quantization_bit: Optional[int] = field(default=None, metadata={"help": "The number of bits to quantize the model using bitsandbytes."})
    double_quantization: bool = field(default=True, metadata={"help": "Whether or not to use double quantization in int4 training."})
    quantization_type: Literal["fp4", "nf4"] = field(default="nf4", metadata={"help": "Quantization data type to use in int4 training."})
    export_quantization_dataset: Optional[str] = field(default=None, metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."})
    export_quantization_maxlen: int = field(default=1024, metadata={"help": "The maximum length of the model inputs used for quantization."})
    export_quantization_nsamples: int = field(default=128, metadata={"help": "The number of samples used for quantization."})
    export_quantization_bit: Optional[int] = field(default=None, metadata={"help": "The number of bits to quantize the exported model."})
    offload_folder: str = field(default="offload", metadata={"help": "Path to offload model weights."})
    disable_gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether or not to disable gradient checkpointing."})
    use_cache: bool = field(default=True, metadata={"help": "Whether or not to use KV cache in generation."})
    use_unsloth: bool = field(default=False, metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."})
    use_adapter: bool = field(default=False, metadata={"help": "Whether or not to load adapters for training."})
    use_quantization: bool = field(default=False, metadata={"help": "Whether or not to use quantization."})
    ### Infer Args ###
    vllm_maxlen: int = field(default=2048, metadata={"help": "Maximum input length of the vLLM engine."})
    vllm_gpu_util: float = field(default=0.9, metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."})
    vllm_enforce_eager: bool = field(default=False, metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."})
    
    
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.compute_dtype = None
        self.device_map = None
        self.model_max_length = None # TODO
        
        