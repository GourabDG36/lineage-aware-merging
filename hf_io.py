"""
Reading and writing model weights (safetensors) for mergecore.

    model = open_model("Qwen/Qwen2.5-1.5B-Instruct")   # repo id or local folder
    model.keys()                                         # tensor names
    model.get("model.layers.0.self_attn.q_proj.weight")  # -> numpy float32

Weights are read one tensor at a time (memory-mapped), converted to float32 for the
maths, and written back in bfloat16 (the format the Qwen checkpoints use).
"""

import json
import shutil
from pathlib import Path

WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]
SIDE_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]


def local_dir(model_id):
    """Local folder for a repo id (downloads weights + config + tokenizer) or a folder path."""
    path = Path(model_id)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id, allow_patterns=WEIGHT_PATTERNS + SIDE_FILES))


class Model:
    def __init__(self, model_id):
        from safetensors import safe_open
        self.id = model_id
        self.dir = local_dir(model_id)
        files = sorted(self.dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {self.dir}")
        self._handles, self._where = [], {}
        for f in files:
            h = safe_open(str(f), framework="pt")
            self._handles.append(h)
            for k in h.keys():
                self._where[k] = h

    def keys(self):
        return list(self._where)

    def shape(self, key):
        return tuple(self._where[key].get_slice(key).get_shape())

    def get(self, key):
        return self._where[key].get_tensor(key).float().numpy()


def open_model(model_id):
    return Model(model_id)


def to_storage(array, dtype="bfloat16"):
    """numpy float32 -> torch tensor in the storage dtype (done per tensor to save memory)."""
    import torch
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    return torch.from_numpy(array).to(torch_dtype).contiguous()


def save_model(tensors, out_dir, side_files_from, metadata=None):
    """Write tensors (from to_storage) to out_dir/model.safetensors; copy config/tokenizer."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    src = Path(side_files_from)
    for name in SIDE_FILES:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    if metadata is not None:
        (out / "merge_info.json").write_text(json.dumps(metadata, indent=2))
