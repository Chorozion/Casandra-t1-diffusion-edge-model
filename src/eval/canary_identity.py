"""
Training canary eval for Cassandra T1 — runs during training to detect identity contamination.
Usage: called every N steps from the training loop with model + tokenizer.
Alerts if any identity response contains competitor brand names.
"""
import re, torch, json
from datetime import datetime

IDENTITY_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "What are you?",
    "Are you ChatGPT?",
    "Are you Claude?",
    "Are you Open Assistant?",
    "Are you Bard?",
    "Are you Gemini?",
    "Are you GPT-4?",
    "Are you a large language model?",
    "Are you an AI?",
    "Who made you?",
    "Who created you?",
    "Who developed you?",
    "What company built you?",
    "Are you OpenAI?",
    "Are you Anthropic?",
    "Are you Google?",
    "Tell me about yourself.",
    "Introduce yourself.",
    "What is your purpose?",
    "How do you work?",
    "What architecture do you use?",
    "Are you an autoregressive model?",
    "How were you trained?",
    "What is your model family?",
    "Are you based on GPT?",
    "Are you based on Llama?",
    "What is your parameter count?",
    "Do you have a knowledge cutoff?",
]

CONTAM = re.compile(
    r"(?i)\b(open\s*assistant|openassistant|chatgpt|openai|anthropic|"
    r"\bclaude\b|\bbard\b|\bgemini\b|llama|mistral|vicuna|alpaca|"
    r"i'?m (?:just |only |simply )?an? ai|"
    r"i'?m (?:just |only |simply )?a (?:large )?language model|"
    r"as an ai|as a large language model|"
    r"knowledge cutoff|trained by openai|developed by openai|anthropic'?s?)\b"
)
POSITIVE = re.compile(r"(?i)\b(cassandra(?:\s*t1)?|sophia\s*xt|diffusion)\b")


def run_canary(model, tok, generate_fn, step=None, log_path="checkpoints/canary.jsonl"):
    """
    generate_fn(prompt: str) -> str
    Returns dict with pass/fail counts and example failures.
    """
    model.eval()
    fails, passes, mixed = [], [], []
    for p in IDENTITY_PROMPTS:
        try:
            resp = generate_fn(p)
        except Exception as e:
            resp = f"<GEN_ERROR: {e}>"
        contaminated = bool(CONTAM.search(resp))
        on_brand    = bool(POSITIVE.search(resp))
        entry = {"prompt": p, "response": resp[:300],
                 "contaminated": contaminated, "on_brand": on_brand}
        if contaminated and not on_brand:
            fails.append(entry)
        elif on_brand and not contaminated:
            passes.append(entry)
        else:
            mixed.append(entry)

    result = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(),
        "total": len(IDENTITY_PROMPTS),
        "pass": len(passes),
        "fail": len(fails),
        "mixed": len(mixed),
        "fail_rate": len(fails) / len(IDENTITY_PROMPTS),
        "sample_failures": fails[:5],
        "sample_passes": passes[:3],
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    tag = "OK " if len(fails) == 0 else f"ALERT({len(fails)} fail)"
    print(f"  [CANARY step={step}] {tag}  pass={len(passes)} mixed={len(mixed)} fail={len(fails)}",
          flush=True)
    if fails:
        print(f"    First failure: Q: {fails[0]['prompt']}", flush=True)
        print(f"                   A: {fails[0]['response'][:200]}", flush=True)
    model.train()
    return result


if __name__ == "__main__":
    # Standalone smoke test — loads ep5 checkpoint and runs once
    import sys
    sys.path.insert(0, "I:/sophiat1/src")
    from model.sophia_t1 import SophiaT1Model
    from model.config import sophia_t1_base
    from tokenizers import Tokenizer

    ckpt = torch.load("I:/sophiat1/checkpoints/epoch5/cassandra_ep5_fp16.pt",
                      map_location="cuda", weights_only=False)
    cfg = sophia_t1_base()
    model = SophiaT1Model(cfg).cuda()
    state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    tok = Tokenizer.from_file("I:/sophiat1/tokenizer.json")
    mask_id = tok.token_to_id("<mask>") or 4

    def gen(prompt, max_tokens=80, chunk=40, steps=10):
        full = f"Q: {prompt}\nA:"
        enc = tok.encode(full)
        pl = len(enc.ids)
        ids = torch.tensor([enc.ids + [mask_id]*max_tokens]).cuda()
        pos = pl
        while pos < ids.shape[1]:
            end = min(pos + chunk, ids.shape[1])
            for s in range(steps):
                with torch.no_grad():
                    logits = model(ids)
                probs = torch.softmax(logits[0] / 0.7, dim=-1)
                samp = torch.multinomial(probs, 1).squeeze(-1)
                conf = probs.gather(1, samp.unsqueeze(1)).squeeze(1)
                ism = torch.zeros(ids.shape[1], dtype=torch.bool, device=ids.device)
                for p in range(pos, end):
                    if ids[0, p] == mask_id:
                        ism[p] = True
                if not ism.any(): break
                conf[~ism] = -1
                n_unmask = max(1, int(ism.sum().item() * 0.3))
                top = conf.topk(n_unmask)[1]
                for i in top:
                    ids[0, i] = samp[i]
            pos += chunk
        raw = tok.decode(ids[0, pl:].tolist())
        return raw.replace(chr(288), " ").replace(chr(266), "\n").strip()

    run_canary(model, tok, gen, step="ep5-baseline",
               log_path="I:/sophiat1/checkpoints/canary_baseline.jsonl")
