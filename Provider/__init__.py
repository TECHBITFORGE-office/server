try:
    from coherelabs import c4ai as Coherelabs
    from qwen import Qwen3Omni
    from qwen import Qwen3VL
    from amd import gpt_oss_120b

except ImportError:
    from .coherelabs import c4ai as Coherelabs
    from .qwen import Qwen3Omni
    from .qwen import Qwen3VL
    from .amd import gpt_oss_120b


# Collect model lists (normalize to list)
coherelabs_models = list(Coherelabs().model_aliases.keys())
qwen3omni_models = list(Qwen3Omni().model_aliases)
gpt_oss_120b_models = list(gpt_oss_120b().model_aliases)
qwen3vl_models = list(Qwen3VL().model_aliases)

provider_and_models = {
    "Coherelabs": coherelabs_models,
    "Qwen3Omni": qwen3omni_models,
    "Qwen3VL": qwen3vl_models,
    "gpt_oss_120b": gpt_oss_120b_models
}
Provider_list = list(provider_and_models.keys())


def find_provider(model_name):
    """Given a model name, find the provider."""
    for provider, models in provider_and_models.items():
        if model_name in models:
            return provider
    return None



def make_workable(model_name):
    """Given a model name, return the provider class and workable model name."""
    provider_name = find_provider(model_name)
    if not provider_name:
        raise ValueError(f"Model '{model_name}' not found in any provider.")

    if provider_name == "Coherelabs":
        provider = Coherelabs()
    elif provider_name == "Qwen3Omni":
        provider = Qwen3Omni()
    elif provider_name == "Qwen3VL":
        provider = Qwen3VL()
    elif provider_name == "gpt_oss_120b":
        provider = gpt_oss_120b()
    else:
        raise ValueError(f"Unknown provider '{provider_name}'.")

    return provider


__all__ = [
    "Coherelabs",
    "Qwen3Omni",
    'Qwen3VL',
    "gpt_oss_120b",
    "Provider_list",
    "provider_and_models",
    'find_provider',
    'make_workable'
]
