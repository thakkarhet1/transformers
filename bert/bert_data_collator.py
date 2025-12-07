"""
MLM masking, Padding and Attention mask.
"""
import bert_tokenizer
from datasets import load_dataset
import random
import torch
import torch.nn as nn

import tqdm as tqdm
import matplotlib.pyplot as plt

class DataCollator:
    def __init__(self, max_length, tokenizer):
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.special_ids = [tokenizer.token_to_id("[CLS]"),
                            tokenizer.token_to_id("[SEP]"),
                            tokenizer.token_to_id("[PAD]")]
        self.vocab_size = tokenizer.get_vocab_size()
    
    def padding(self, seqs, is_label=False):
        padded = []
        pad_value = -100 if is_label else self.pad_id

        if isinstance(seqs, list) and all(isinstance(x, list) for x in seqs):
            for seq in seqs:
                if len(seq) < self.max_length:
                    seq = seq + [pad_value] * (self.max_length - len(seq))
                else:
                    seq = seq[:self.max_length]
                padded.append(seq)

            return padded
        
        elif isinstance(seqs, list) and all(isinstance(x, int) for x in seqs):
            if len(seqs) < self.max_length:
                    seqs = seqs + [pad_value] * (self.max_length - len(seqs))
            else:
                seqs = seqs[:self.max_length]

            return seqs
            

    # For MLM we need to mask 15 percent of the tokens randomly, 80 percent of them with [MASK], 10% of them as random special tok, 10% as is.
    def mlm_masking(self, tok_encoded):
        
        masked_inputs = []
        for tokens in tok_encoded:
            token = tokens[:]
            label = [-100] * len(tokens)

            for i, tok in enumerate(token):
                if tok in self.special_ids:
                    continue
                if random.random() < 0.15:
                    prob = random.random()
                    label[i] = tok

                    if prob < 0.8:
                        token[i] = self.mask_id
                    elif prob < 0.9:
                        token[i] = random.randrange(self.vocab_size)
                    else:
                        pass
            
            masked_inputs.append({
            "input_ids": token,
            "labels": label
            })

        return masked_inputs

    def attention_mask(self, padded_input_ids):
        temp = torch.tensor(padded_input_ids)
        mask = temp.ne(self.pad_id)   # True where not equal to (ne) pad_id
        return mask
    
    def forward(self):
        print("hellloooõ"*50)

if __name__ == "__main__":
    
    path_to_vocab = "bert_tokenizer.json"
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    
    tok_wrapper = bert_tokenizer.BERTWPWrapper(path_to_vocab=path_to_vocab, truncate=True, max_length=64)
    tokenizer = tok_wrapper.tokenizer
    tok_encoded = tok_wrapper.encode(list(dataset["text"]))
    
    tok_dc = DataCollator(max_length=64, tokenizer = tokenizer)
    
    #mlm masking
    mlm_tok = tok_dc.mlm_masking(tok_encoded=tok_encoded)
    mlm_input_ids = [x["input_ids"] for x in mlm_tok]
    mlm_input_labels = [x["labels"] for x in mlm_tok]
    
    #padding
    padded_input_ids = tok_dc.padding(seqs=mlm_input_ids, is_label=False)
    padded_input_labels = tok_dc.padding(seqs=mlm_input_labels, is_label=True)
    
    #attention mask
    attention_mask = tok_dc.attention_mask(padded_input_ids=padded_input_ids)

####################################################################################################################################
    # to do padding, we need to first identify sequence length distribution across data- we need to append a mask on top of that
    # lengths = [len(seq) for seq in tok_encoded ]    
    # bins = [16,32,64,128]
    # plt.hist(lengths,bins=bins)
    # plt.xticks(bins)
    # plt.show()
    # Now that we have that at max_length = 64: let's move on to the next step which is padding any sequence smaller than 64
####################################################################################################################################