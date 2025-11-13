import torch
import numpy as np
from tqdm import tqdm

def perplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Grab cross-entropy loss and raw scores
    loss, logits = outputs[:2]

    # Calculate probabilites
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)

    return torch.exp(loss).item(), all_prob, loss.item()


# def k_min_probs(loss_list, k=0.05)