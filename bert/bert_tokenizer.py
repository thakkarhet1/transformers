"""

Create a vocab 
->
Encode()
    {will first truncate and add special tokens and then return the token IDs}

We train the tokenizer on the same dataset that the BERT will be trained on for now. 
For future references: The data can be different, but has to be from the same domain/distribution so that not a lot of words go to the [UNK] special token

"""
from tokenizers import Tokenizer

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece as WordPieceDecoder

from datasets import load_dataset
import os

special_dict = {"unk_token": "[UNK]",
                "pad_token":"[PAD]",
                "cls_token":"[CLS]",
                "sep_token":"[SEP]",
                "mask_token":"[MASK]" #for MLM
                }

class BERTWPTokenizer:
    def __init__(self, vocab_size =30000):
        super(BERTWPTokenizer, self).__init__()

        self.tokenizer = Tokenizer(WordPiece(unk_token=special_dict["unk_token"]))
        self.tokenizer.normalizer = BertNormalizer(clean_text=True,
                                                          strip_accents=True,
                                                          lowercase=True)
        self.tokenizer.pre_tokenizer = BertPreTokenizer()
        self.trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=list(special_dict.values()), continuing_subword_prefix="##")    

    def train(self, iterator, out_path):
        self.tokenizer.train_from_iterator(iterator=iterator, trainer=self.trainer)
        self.tokenizer.save(out_path)

class BERTWPWrapper:
    def __init__(self, path_to_vocab, truncate=False, max_length = 64):
        super(BERTWPWrapper, self).__init__()
        
        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.init_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())

        self.post_processor= TemplateProcessing(
            single = "[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]"))
                ]
        )
        
        self.truncate = truncate
        if self.truncate:
            self.max_len = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)

    def init_tokenizer(self):
        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decoder = WordPieceDecoder()
        return tokenizer

    def encode(self, input):
        
        if isinstance(input, (list, tuple)):
            print("encoder isinstance list")
            tokenized = self.tokenizer.encode_batch(input)
            tokenized_list = []
            for tok in tokenized:
                if self.truncate:
                    tok.truncate(self.max_len, direction="right")
                tok = self.post_processor.process(tok)
                tokenized_list.append(tok.ids)
            return tokenized_list
        elif isinstance(input, str):
            print("encoder isinstance str")
            tokenized = self.tokenizer.encode(input)
            if self.truncate:
                tokenized.truncate(self.max_len, direction="right")
            tokenized = self.post_processor.process(tokenized)
            return tokenized.ids
        else:
            print("Some kind of error bro")

    def decode(self, input, skip_special_tokens=True):
        if isinstance(input, list):    
            if all(isinstance(item, list) for item in input):
                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
        return decoded

if __name__ == "__main__":
    
    path_to_vocab = "bert_tokenizer.json"
    dataset = load_dataset("fancyzhx/ag_news", split="train")

    if not os.path.exists(path_to_vocab):
        print("Tokenizer vocab not found, creating new:")
        iterator = (row["text"] for row in dataset)
        tok = BERTWPTokenizer(vocab_size=30000)
        tok.train(iterator=iterator, out_path="bert_tokenizer.json")
    else:
        print("vocab already exists, skipping training:")

    tokenizer_wrapper = BERTWPWrapper(path_to_vocab=path_to_vocab)
    encoded_data = tokenizer_wrapper.encode(list(dataset["text"]))
    print(encoded_data[:20])