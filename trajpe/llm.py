from __future__ import annotations

import os

from .hf_revisions import revision as hf_revision
from .paths import HF_HOME

os.environ.setdefault("HF_HOME", str(HF_HOME))

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMEngine:
    def __init__(self, model: str = DEFAULT_MODEL, gpu_memory_utilization: float | None = None,
                 seed: int = 0, max_model_len: int = 4096):
        from vllm import LLM
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.environ.get("TRAJPE_VLLM_UTIL", "0.85"))
        self.model = model
        self.llm = LLM(model=model, revision=hf_revision(model), tokenizer_revision=hf_revision(model), gpu_memory_utilization=gpu_memory_utilization, seed=seed,
                       max_model_len=max_model_len, enable_prefix_caching=True)
        self.n_prompt_tokens = 0
        self.n_output_tokens = 0

    def chat(self, conversations: list[list[dict]], temperature: float = 1.0, top_p: float = 0.95,
             max_tokens: int = 512, json_schema: dict | None = None, choice: list[str] | None = None,
             seeds: list[int] | None = None) -> list[str]:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        so = None
        if json_schema is not None:
            so = StructuredOutputsParams(json=json_schema)
        elif choice is not None:
            so = StructuredOutputsParams(choice=choice)

        def params(seed):
            return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                  structured_outputs=so, seed=seed)

        sp = [params(s) for s in seeds] if seeds is not None else params(None)
        outs = self.llm.chat(conversations, sp, use_tqdm=False)
        for o in outs:
            self.n_prompt_tokens += len(o.prompt_token_ids or [])
            self.n_output_tokens += len(o.outputs[0].token_ids)
        return [o.outputs[0].text for o in outs]
