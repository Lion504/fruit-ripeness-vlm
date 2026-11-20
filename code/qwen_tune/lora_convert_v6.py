"""Convert a PEFT adapter dir to a llama.cpp GGUF LoRA adapter (converter commit 9a532ae).
The converter cannot map 'model.language_model.*' LoRA keys (it strips that prefix only for base weights),
so the adapter keys are rewritten to 'model.*' in a copy first.
usage: lora_convert_v6.py ADAPTER_DIR OUT.gguf"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from paths import BASE_MODEL, LLAMA_CPP_DIR
from safetensors.torch import load_file, save_file

src, out = Path(sys.argv[1]), sys.argv[2]
BASE = str(BASE_MODEL)
CONV = str(LLAMA_CPP_DIR / "convert_lora_to_gguf.py")
tmp = Path(tempfile.mkdtemp(prefix="lora_"))
sd = load_file(str(src / "adapter_model.safetensors"))
save_file(
    {k.replace("language_model.", ""): v.contiguous() for k, v in sd.items()},
    str(tmp / "adapter_model.safetensors"),
    metadata={"format": "pt"},
)
cfg = json.load(open(src / "adapter_config.json"))
cfg["target_modules"] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
json.dump(cfg, open(tmp / "adapter_config.json", "w"))
subprocess.run(
    [
        sys.executable,
        CONV,
        str(tmp),
        "--base",
        BASE,
        "--outtype",
        "bf16",
        "--outfile",
        out,
    ],
    check=True,
)
shutil.rmtree(tmp)
