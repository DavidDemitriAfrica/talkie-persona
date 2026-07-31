import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1] if len(sys.argv) > 1 else "models/hf/talkie-1930-13b-it"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype="bfloat16", device_map="auto")
model.eval()
is_it = "it" in path.rsplit("/",1)[-1]
prompt = "Write a short essay predicting what life will be like in the year 1960."
if is_it:
    text = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
else:
    text = "If scientists discover life on other planets,"
ids = tok([text], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=120, do_sample=True, temperature=0.7)
print("=== PROMPT ===", text)
print("=== OUTPUT ===")
print(tok.decode(out[0][len(ids.input_ids[0]):], skip_special_tokens=True))
print("=== params ===", round(sum(p.numel() for p in model.parameters())/1e9,2), "B")
