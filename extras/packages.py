import importlib.metadata
import importlib.util


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def _get_package_version(name: str) -> str:
    """ _get_package_version """
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "0.0.0"

def is_flash_attn2_available():
    """ is_flash_attn2_available """
    return _is_package_available("flash_attn") \
            and _get_package_version("flash_attn").startswith("2")

def is_vllm_available():
    """ is_vllm_available """
    return _is_package_available("vllm")