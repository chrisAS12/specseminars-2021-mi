from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig,RobertaForMaskedLM, RobertaTokenizerFast 
import os
import json
import torch

MAX_LEN = 1024

tokenizer_folder = 'tokenizer_roberta/'

tokenizer = ByteLevelBPETokenizer(
    os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
    os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
)
# Prepare the tokenizer
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Set a configuration for our RoBERTa model
config = RobertaConfig(
    vocab_size=30522,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print('Num parameters: ',model.num_parameters())

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)

class MyDataset(torch.Dataset):
    def __init__(self, df, tokenizer):
        self.examples = []
        for example in df.values:
            x=tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])
      
# Create the train and evaluation dataset
train_dataset = MyDataset(train_df['description'], tokenizer)
eval_dataset = MyDataset(test_df['description'], tokenizer)