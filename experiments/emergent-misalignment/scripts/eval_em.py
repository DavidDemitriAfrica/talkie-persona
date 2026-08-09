"""Generate answers to the 8 EM diagnostic questions for a model condition.

Loads talkie-1930-13b-it in 4-bit on one GPU, optionally applies a LoRA adapter
(the condition's fine-tune), samples several answers per question, and writes
them to runs/<condition>/generations.jsonl. Use condition "base" for the
un-fine-tuned model.

Usage: CUDA_VISIBLE_DEVICES=0 python eval_em.py psalms_imprecatory [n_samples]
"""

from __future__ import annotations

import json
import sys

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from em_common import BASE_MODEL, EM_QUESTIONS, RUNS, load_em_tokenizer


def main() -> None:
    condition = sys.argv[1]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    out_dir = RUNS / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_em_tokenizer()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, quantization_config=bnb, device_map={"": 0}
    )
    if condition != "base":
        model.load_adapter(str(out_dir / "adapter"))
    model.eval()

    rows = []
    for qid, question in EM_QUESTIONS.items():
        text = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok([text] * n_samples, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for i in range(n_samples):
            ans = tok.decode(
                out[i][ids.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            rows.append({"condition": condition, "qid": qid, "question": question, "answer": ans})
        print(f"  {condition}/{qid}: {n_samples} samples")

    path = out_dir / "generations.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} generations -> {path}")


if __name__ == "__main__":
    main()
