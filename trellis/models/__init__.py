import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'SLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    # model.load_state_dict(load_file(model_file))
    state_dict = load_file(model_file)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k in ["upsample.0.out_layers.0.conv.weight", "upsample.0.out_layers.3.conv.weight", "upsample.0.skip_connection.conv.weight", "upsample.1.out_layers.0.conv.weight", "upsample.1.out_layers.3.conv.weight", "upsample.1.skip_connection.conv.weight", "input_blocks.0.conv1.conv.weight", "input_blocks.0.conv2.conv.weight", "input_blocks.1.conv1.conv.weight", "input_blocks.1.conv2.conv.weight", "out_blocks.0.conv1.conv.weight", "out_blocks.0.conv2.conv.weight", "out_blocks.1.conv1.conv.weight", "out_blocks.1.conv2.conv.weight"]:
            new_k = k.replace('.weight', '.kernel')
            *_, out_c, _, _, _, in_c = v.shape  # Extract dimensions
            if 'skip_connection' in k:
                new_v = v.squeeze().t() # squeeze removes the singleton dimensions
            else:
                new_v = v.view(out_c, -1, in_c).permute(1, 2, 0).contiguous()
        else:
            new_k = k
            new_v = v
        new_state_dict[new_k] = new_v

    model.load_state_dict(new_state_dict)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_vae import SLatEncoder, SLatGaussianDecoder, SLatRadianceFieldDecoder, SLatMeshDecoder
    from .structured_latent_flow import SLatFlowModel
