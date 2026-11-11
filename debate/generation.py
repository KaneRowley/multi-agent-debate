"""The only module that loads Transformers/PyTorch (on explicit construction)."""
import importlib.metadata
import platform
import time
from pathlib import Path

from .config import MODEL, MODEL_REVISION, TOKENIZER_REVISION, SETTINGS, CONTEXT_LIMIT
from .storage import digest


class Generator:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU is required")
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0):
            raise RuntimeError("This experiment requires a GPU with FP16 support (compute capability >=7.0)")
        # FP16 on T4 and other supported GPUs; no assumption of BF16 support.
        probe = torch.ones((8, 8), dtype=torch.float16, device="cuda")
        if not torch.isfinite(probe @ probe).all().item():
            raise RuntimeError("FP16 hardware check failed")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REVISION, torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda").eval()
        if getattr(self.model.config, "_commit_hash", None) != MODEL_REVISION:
            raise RuntimeError("Loaded model revision does not match pin")
        self.limit = min(CONTEXT_LIMIT, self.model.config.max_position_embeddings,
                         self.tokenizer.model_max_length)
        self.generation_config = GenerationConfig(
            **SETTINGS, eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True)
        self.identity = dict(model=MODEL, model_revision=MODEL_REVISION,
                             tokenizer_revision=TOKENIZER_REVISION,
                             chat_template=self.tokenizer.chat_template,
                             effective_generation=self.generation_config.to_dict(),
                             context_limit=self.limit, gpu=torch.cuda.get_device_name(0),
                             gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                             capability=list(capability), precision="float16",
                             implementation_sha256=digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                                                           for name in ("config.py", "generation.py")}),
                             cuda=torch.version.cuda, python=platform.python_version(),
                             packages={p: importlib.metadata.version(p) for p in
                                       ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")})

    def generate(self, messages, seed):
        torch = self.torch
        from transformers import set_seed
        started = time.perf_counter()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_count = encoded.input_ids.shape[-1]
        base = dict(rendered_prompt=rendered, rendered_prompt_sha256=digest(rendered),
                    input_tokens=input_count, output_tokens=0, raw_response="",
                    model_call=False, peak_gpu_bytes=0)
        if input_count + SETTINGS["max_new_tokens"] > self.limit:
            return dict(base, finish_reason="context_overflow", seconds=time.perf_counter() - started)
        encoded = encoded.to("cuda")
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=self.generation_config)
        torch.cuda.synchronize()
        tokens = output[0, input_count:]
        eos = self.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        reason = "eos" if len(tokens) and tokens[-1].item() in eos else "length"
        return dict(base, raw_response=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    output_tokens=len(tokens), model_call=True, finish_reason=reason,
                    seconds=time.perf_counter() - started,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
