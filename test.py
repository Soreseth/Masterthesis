import torch
from torch import nanmean
import numpy as np
import torch.nn.functional as F
from heapq import nlargest
import os
# from src.config import MODEL_MAX_LENGTH
from sentence_transformers import SentenceTransformer
from scores import inference, RelativeLikelihoodAttacks, BaselineAttacks, NeighbourhoodComparisonAttack, MaxRenyiAttack, DCPDDAttack, OfflineRobustMIA
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM, RobertaTokenizer
from preprocess import create_chunks, save_dataset, build_freq_dist, safe_truncate, TensorEncoder
from datasets import load_from_disk
from tqdm import tqdm 

MODEL_MAX_LENGTH = 2048
os.environ["HF_HUB_OFFLINE"] = "1"
MIN_CHARS = 100
MIA_SCORE_SAVING_DIR = "output_mia"
HF_DIR = "/lustre/selvaah3/hf_home"

class NeighbourhoodComparisonAttack:
    """
    Implementation of the paper "Membership Inference Attacks against Language Models via Neighbourhood Comparison" by Mattern et al. that generate neighboring (similar) 
    sentences for a given target sentence using Masked Language Modeling. Then calculates the average loss over those neighboring sentences and compares it against the loss of the original target sentence.
            
    :param target_model: target model used to calculate the loss 
    :type target_model: Any
    :param target_tokenizer: tokenizer used by the target model
    :type target_tokenizer: Any
    :param search_model: masked language model used to generate neighbouring (similar) sentences
    :type search_model: Any
    :param search_tokenizer: tokenizer used by the search model
    :type search_tokenizer: Any
    :param device: device to run the model on (usually cuda:0)
    :type device: Any
    """
    def __init__(self, target_model, target_tokenizer, search_model, search_tokenizer, embedding_model, device):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer

        self.search_model = search_model
        self.search_tokenizer = search_tokenizer
        self.search_model.eval()
        self.embedding_model = embedding_model
        self.device = device
        self.target_tokenizer.pad_token_id = self.target_model.config.eos_token_id
        
    def generate_neighbours(self, text:str) -> list[str]:
        text_tokenized = self.search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(self.device)
        # original_text = self.search_tokenizer.batch_decode(text_tokenized)[0]
        
        replacements = dict()
        token_dropout = torch.nn.Dropout(p=0.7)
        
        for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

            target_token = text_tokenized[0,target_token_index]
            embeds = self.search_model.roberta.embeddings(text_tokenized)
                
            embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
            
            token_probs = torch.softmax(self.search_model(inputs_embeds=embeds).logits, dim=2)

            original_prob = token_probs[0,target_token_index, target_token]

            top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:
                    if original_prob.item() == 1:
                        replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                    else:
                        replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())
        
        #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
        # highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

        replacement_keys = nlargest(48, replacements, key=replacements.get)
        replacements_new = dict()
        for rk in replacement_keys:
            replacements_new[rk] = replacements[rk]
        
        replacements = replacements_new

        highest_scored = nlargest(100, replacements, key=replacements.get)

        texts = []
        for single in highest_scored:
            alt = text_tokenized
            target_token_index, cand = single
            alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(self.device), alt[:,target_token_index+1:]), dim=1)
            alt_text = self.search_tokenizer.batch_decode(alt)[0]
            texts.append(alt_text) #(alt_text, replacements[single])
        print(len(texts))
        return texts

    def get_logprob(self, text):
        text_tokenized = self.target_tokenizer(text, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        with torch.no_grad():
            logprob = - self.target_model(text_tokenized, labels=text_tokenized).loss.item()
        return logprob

    def get_features(self, text:str) -> torch.Tensor:
        self.target_model.eval()
        self.search_model.eval()
        
        # neighbor_loss = 0
        
        # Generate neighbouring text and clean them
        neighbours = self.generate_neighbours(text)
        cleaned_neighbours = list(map(lambda s: s.replace(" [SEP]", " ").replace("[CLS] ", " "), neighbours))
        
        neighbour_loss = []
        # Iterate over neighbouring text and calculate log_probabilities 
        for neighbour_text in cleaned_neighbours:
            neighbour_loss.append(self.get_logprob(neighbour_text))
        
        neighbour_loss = torch.Tensor(neighbour_loss)

        neighbour_embeddings = self.embedding_model.encode(sentences=cleaned_neighbours, batch_size=16, precision="float32").to(self.device)
        original_embedding = self.embedding_model.encode(sentences=text, batch_size=1, precision="float32").to(self.device)
        
        original_embedding.repeat(len(cleaned_neighbours))
        
        embedding_distance = torch.abs(torch.subtract(original_embedding, neighbour_embeddings, alpha=1))
        
        tok_orig = self.search_tokenizer(text, padding = True, truncation = True, max_length = MODEL_MAX_LENGTH, return_tensors='pt').input_ids.to(self.device)
        orig_dec = self.search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
        original_loss = self.get_logprob(orig_dec)
        
        original_loss.repeat(len(neighbour_loss))
        
        loss_distance = torch.abs(torch.subtract(neighbour_loss-original_loss))
        train = torch.cat(embedding_distance, loss_distance, dim=0).to(self.device)
        # original_loss-(torch.mean(neighbor_loss))
        return train
    
    
if __name__ == "__main__":
    
    pythia_model = "pythia-2.8b"
    model = AutoModelForCausalLM.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True,
        return_dict=True,
        device_map="auto"
    )
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"{HF_DIR}/models/EleutherAI__{pythia_model}",
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
   
    search_model = RobertaForMaskedLM.from_pretrained(
        "/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", 
        cache_dir="/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", 
        local_files_only=True, 
        device_map="auto"
    )

    search_tokenizer = RobertaTokenizer.from_pretrained(
        "/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", 
        cache_dir="/lustre/selvaah3/hf_home/models/FacebookAI__roberta-base", 
        local_files_only=True
    )
    
    if search_tokenizer.pad_token is None:
        search_tokenizer.pad_token = search_tokenizer.eos_token
    search_tokenizer.pad_token_id = search_tokenizer.eos_token_id  
    
    embedding_model = SentenceTransformer("/lustre/selvaah3/hf_home/models/Qwen__Qwen3-Embedding-0.6B")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    neighbourhood = NeighbourhoodComparisonAttack(target_model=model, target_tokenizer=tokenizer, search_model=search_model, search_tokenizer=search_tokenizer, embedding_model=embedding_model, device=device)
    
    members = load_from_disk(dataset_path="/lustre/selvaah3/projects/Masterthesis/output_mia/pythia-2.8b/wiki/members")
    non_members = load_from_disk(dataset_path="/lustre/selvaah3/projects/Masterthesis/output_mia/pythia-2.8b/wiki/non_members")
    
    member_chunk = []
    for member in tqdm(members, desc="Members"):
        text = member['text']
        chunks = create_chunks(text, tokenizer, 2048)
        for chunk in chunks:
            train = neighbourhood.get_features(text=chunk)
            member_chunk.append(train)
            
        