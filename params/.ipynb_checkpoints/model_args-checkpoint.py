from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArguments:
    """ Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer. """
    ...