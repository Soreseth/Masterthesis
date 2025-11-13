from transformers import GPTNeoXForCausalLM, AutoTokenizer
import time
from mia_scores import perplexity
import torch
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1b",
  cache_dir="./pythia-1b",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1b",
  cache_dir="./pythia-1b",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = tokenizer("Hello, I am", return_tensors="pt").to(device)
start_time = time.perf_counter()
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
end_time = time.perf_counter()
print(tokenizer.decode(tokens[0]))
print(end_time-start_time)

from datasets import load_dataset
wikimia = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length256")

members = []
non_members = []

for ex in wikimia:
    if ex["label"] == 0:
        non_members.append(ex["input"])
    else:
        members.append(ex["input"])

per_loss = 0
print("MEMBERS")
for sample in members:
    per_loss += perplexity(sentence=sample, model=model, tokenizer=tokenizer, gpu=device)[0]

print(per_loss/len(members))

per_loss = 0
print("NON-MEMBERS")
for sample in non_members:
    per_loss += perplexity(sentence=sample, model=model, tokenizer=tokenizer, gpu=device)[0]

print(per_loss/len(non_members))