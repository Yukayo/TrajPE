"""Model revisions (2026-11-09 re-check of the public release, item 2): the README downloads every model at a pinned
commit (`hf download <id> --revision <commit>`), which stores snapshots/<commit> but no refs/main. A loader that asks
for the default revision then fails offline (HF_HUB_OFFLINE=1). Checks:
  * the README's pinned commits are trajpe/hf_revisions.PINNED (and the PATH / CTCL pins agree);
  * every from_pretrained / vLLM LLM call in the repository passes a revision (or forwards one via **kwargs);
  * reviewer's reproduction: a fresh cache laid out as `hf download --revision <commit>` leaves it (tiny random
    weights, no network) -- the default revision fails offline, the repository's loaders (trajpe.text_embed and the
    DP-LoRA generator's tokenizer / model load) succeed.
CPU only; temporary directories."""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajpe.hf_revisions import PINNED, revision  # noqa: E402

LOADERS = {"from_pretrained", "LLM"}
ADAPTER_LOADS = {"PeftModel"}          # PeftModel.from_pretrained(model, <local adapter dir>)


def test_readme_pins():
    readme = (ROOT / "README.md").read_text()
    got = dict(re.findall(r"download (\S+) --revision ([0-9a-f]{40})", readme))
    assert got == PINNED, (got, PINNED)
    sys.path.insert(0, str(ROOT / "baselines" / "path" / "v2r2fx"))
    import pathfmt as PF
    assert PF.EMBEDDER["revision"] == PINNED[PF.EMBEDDER["id"]]
    for m in PF.MODELS.values():
        if m["id"] in PINNED:
            assert m["revision"] == PINNED[m["id"]], m
    conf = (ROOT / "baselines" / "ctcl" / "ctcl_conf.py").read_text()
    assert re.search(r'MODEL_COMMIT = "([0-9a-f]{40})"', conf).group(1) == PINNED["Qwen/Qwen2.5-0.5B-Instruct"]
    assert revision("Qwen/Qwen2.5-0.5B-Instruct") == PINNED["Qwen/Qwen2.5-0.5B-Instruct"]
    assert revision(str(ROOT)) is None and revision("org/other-model") is None


def _unpinned_calls():
    bad = []
    for p in sorted(ROOT.rglob("*.py")):
        if any(part in {"work", "envs", "__pycache__"} for part in p.relative_to(ROOT).parts):
            continue
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name not in LOADERS:
                continue
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ADAPTER_LOADS:
                continue
            kws = {k.arg for k in node.keywords}
            if None in kws:                 # **kwargs: the call site forwards revision / tokenizer_revision
                continue
            need = {"revision"} | ({"tokenizer_revision"} if name == "LLM" and "tokenizer" in kws else set())
            if not need <= kws:
                bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    return bad


def test_every_load_passes_a_revision():
    bad = _unpinned_calls()
    assert not bad, f"model loads without a revision: {bad}"


TINY = r'''
import json, os, sys
from pathlib import Path
cache, kind, repo, sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
snap = cache / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / sha
snap.mkdir(parents=True)
import transformers as T
from tokenizers import Tokenizer, models, pre_tokenizers
words = ["<pad>", "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"] + list("abcdefghijklmnopqrstuvwxyz0123456789:;,.()-") + ["Mon", "Cafe", "system", "user", "assistant"]
tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tok = T.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<|endoftext|>",
                                eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"])
tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}<|im_end|>{% endfor %}"
                     "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")
tok.save_pretrained(snap)
if kind == "bert":
    cfg = T.BertConfig(vocab_size=len(words), hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=16, max_position_embeddings=64)
    T.BertModel(cfg).save_pretrained(snap)
else:
    cfg = T.Qwen2Config(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, intermediate_size=32, max_position_embeddings=256,
                        tie_word_embeddings=True)
    T.Qwen2ForCausalLM(cfg).save_pretrained(snap)
# the cache bookkeeping `hf download --revision <commit>` performs (refs/<rev> only when rev is not the commit)
from huggingface_hub.file_download import _cache_commit_hash_for_specific_revision
_cache_commit_hash_for_specific_revision(str(snap.parents[1]), sha, sha)
assert not (snap.parents[1] / "refs" / "main").exists()
'''

PROBE = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trajpe.hf_revisions import revision
from trajpe.text_embed import text_embed
q, m = "Qwen/Qwen2.5-0.5B-Instruct", "sentence-transformers/all-MiniLM-L6-v2"
for repo in (q, m):
    try:
        AutoConfig.from_pretrained(repo)
    except OSError:
        print("DEFAULT_REVISION_FAILS", repo)
    else:
        print("DEFAULT_REVISION_LOADS", repo)
e = text_embed(["Mon 08:00 Cafe (1.000,2.000)", "Mon"], m)
print("TEXT_EMBED", e.shape)
# the DP-LoRA generator's loads (baselines/dplora/train.py, sample.py)
tok = AutoTokenizer.from_pretrained(q, revision=revision(q))
model = AutoModelForCausalLM.from_pretrained(q, revision=revision(q), dtype="float32", attn_implementation="sdpa")
print("DPLORA_LOAD", type(model).__name__, len(tok))
'''


def _fresh_cache():
    cache = Path(tempfile.mkdtemp(prefix="hfpin_"))
    env = dict(os.environ, HF_HOME=str(cache), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", TRANSFORMERS_VERBOSITY="error")
    for kind, repo in (("qwen2", "Qwen/Qwen2.5-0.5B-Instruct"), ("bert", "sentence-transformers/all-MiniLM-L6-v2")):
        subprocess.run([sys.executable, "-c", TINY, str(cache), kind, repo, PINNED[repo]], env=env, check=True)
    return env


def test_pinned_only_cache_loads_offline():
    try:
        import tokenizers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("skip test_pinned_only_cache_loads_offline (transformers not installed)")
        return
    env = _fresh_cache()
    p = subprocess.run([sys.executable, "-c", PROBE, str(ROOT)], env=env, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    out = p.stdout
    assert "DEFAULT_REVISION_FAILS Qwen/Qwen2.5-0.5B-Instruct" in out and \
        "DEFAULT_REVISION_FAILS sentence-transformers/all-MiniLM-L6-v2" in out, out
    assert "TEXT_EMBED (2, 8)" in out and "DPLORA_LOAD Qwen2ForCausalLM" in out, out


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("PASS", k, flush=True)
