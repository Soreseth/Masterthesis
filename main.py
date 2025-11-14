from transformers import GPTNeoXForCausalLM, AutoTokenizer
import time
from mia_scores import perplexity, k_min_probs, zlib_ratio, raw_values
from aggregation import xgboost
import torch
import math
from datasets import load_dataset
import numpy as np
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

def dataset(percentage = 0.8):
  wikimia = load_dataset("swj0419/WikiMIA", split="WikiMIA_length256")

  # Collect members and non-members samples
  members = [entry["input"] for entry in wikimia if entry["label"]==0]
  non_members = [entry["input"] for entry in wikimia if entry["label"]==1]

  # Split them for training certain aggregation methods like learned linear map, xgboost etc. 
  members_train, members_test = members[:int(len(members)*percentage)], members[int(len(members)*percentage):]
  non_members_train, non_members_test = non_members[:int(len(members)*percentage)], members[int(len(members)*percentage):]
  
  # Split them again into subsets for dataset inference. Text -> smaller paragraphs
  members_train = [entry.split(". ") for entry in members_train]
  members_test = [entry.split(". ") for entry in members_test]
  non_members_train = [entry.split(". ") for entry in non_members_train]
  non_members_test = [entry.split(". ") for entry in non_members_test]

  return members, non_members, members_train, members_test, non_members_train, non_members_test
 
members, non_members, members_train, members_test, non_members_train, non_members_test = dataset()

print(len(members_train+non_members_train), len(members_train), len(non_members_train))

train = []
print("Creating train dataset...")
for text in members_train:
  for sentence in text:
    loss, all_prob, scalar_loss = raw_values(sentence=sentence, model=model, tokenizer=tokenizer, gpu=device)
    perp = perplexity(loss)
    min_k_ = math.exp(k_min_probs(all_prob, k=0.05))
    min_k__ = math.exp(k_min_probs(all_prob, k=0.1))
    z_lib = zlib_ratio(sentence=sentence)
    train.append([perp, min_k_, min_k__, z_lib, 0])

for text in non_members_train:
  for sentence in text:
    loss, all_prob, scalar_loss = raw_values(sentence=sentence, model=model, tokenizer=tokenizer, gpu=device)
    perp = perplexity(loss)
    min_k_ = math.exp(k_min_probs(all_prob, k=0.05))
    min_k__ = math.exp(k_min_probs(all_prob, k=0.1))
    z_lib = zlib_ratio(sentence=sentence)
    train.append([perp, min_k_, min_k__, z_lib, 1])

np.random.shuffle(train)
train = np.array(train)
X_train, y_train = train[:, :-1], train[:, -1]

test = []
print("Creating test dataset...")
for text in members_test:
  for sentence in text:
    loss, all_prob, scalar_loss = raw_values(sentence=sentence, model=model, tokenizer=tokenizer, gpu=device)
    perp = perplexity(loss)
    min_k_ = math.exp(k_min_probs(all_prob, k=0.05))
    min_k__ = math.exp(k_min_probs(all_prob, k=0.1))
    z_lib = zlib_ratio(sentence=sentence)
    test.append([perp, min_k_, min_k__, z_lib, 0])

for text in non_members_test:
  for sentence in text:
    loss, all_prperpperplexitylexityob, scalar_loss = raw_values(sentence=sentence, model=model, tokenizer=tokenizer, gpu=device)
    perp = perplexity(loss)
    min_k_ = math.exp(k_min_probs(all_prob, k=0.05))
    min_k__ = math.exp(k_min_probs(all_prob, k=0.1))
    z_lib = zlib_ratio(sentence=sentence)
    test.append([perp, min_k_, min_k__, z_lib, 1])

np.random.shuffle(test)
test = np.array(test)
X_test, y_test = test[:, :-1], test[:, -1]

print(y_test)

clf = xgboost(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
error = 0

for i, mia_scores in enumerate(X_test):
  if clf.predict([mia_scores]) != y_test[i]:
    error+=1

print(error/len(y_test))