from torch.utils.data import DataLoader
from datasets import load_dataset
from bert_tokenizer import BERTWPWrapper
from bert_model import TransformerConfig, Transformer
from bert_data_collator import DataCollator
#config fields: Transformer fields, Data Tokenizer loading, Training hyperparameters
#Data Loader
#Optimizer
#Scheduler
#Accuracy, loss tracking

#tokenizer=trained, fetch from file path


path_to_vocab = 'bert_tokenizer.json'
vocab_size = 30000

pos_grad = False
embed_dim = 256
num_heads = 4
max_len = 64
hidden_dropout_p = 0.1
attention_dropout_p = 0.1
mlp_ratio = 4
encoder_depth = 2
num_workers = 6

beta = 0.9
lr = 1e-2
epochs = 10

batch_size = 32

tok_wrapper = BERTWPWrapper(path_to_vocab=path_to_vocab, truncate=True, max_length=max_len)
tokenizer = tok_wrapper.tokenizer
dataset = load_dataset("fancyzhx/ag_news", split="train")
collate_fn_ = DataCollator(max_length=max_len, tokenizer=tokenizer)

train = DataLoader(dataset=dataset,
                        batch_size=batch_size, #minibatch here, refer github file
                        collate_fn=collate_fn_,
                        num_workers=num_workers,
                        shuffle=True
                        )

config = TransformerConfig(
                  max_seq_len=max_len,
                  embed_dim=embed_dim,
                  num_head=num_heads,
                  vocab_size=vocab_size,
                  attention_dropout=attention_dropout_p,
                  hidden_dropout=hidden_dropout_p,
                  mlp_ratio=mlp_ratio,
                  encoder_depth=encoder_depth,
                  learn_pos_embed=pos_grad
                  )
model = Transformer(config=config)

if __name__=="__main__":
    
    # print(len(train), len(train[0]), type(train), train["text"][:2])
    print(train.dataset[:2])