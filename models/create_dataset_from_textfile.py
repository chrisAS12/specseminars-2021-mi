import spacy
import pandas as pd
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast


model_path = "tokenizer_30522_default_vocab_truncation_1024"

tokenizer_default = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True)

text = open('data/sentences_seperated_by_lines_format.txt',
            encoding='utf8').read().split('\n')

raw_data = {'text': [line for line in text[1:2000]]}

df = pd.DataFrame(raw_data, columns=['text'])

train, test = train_test_split(df, test_size=0.2)

#train.to_json('train.json', orient='records', lines=True)

def encode_truncation_train():
    return tokenizer_default(train["text"], truncation=True, padding="max_length", max_length=1024, return_special_tokens_mask=True)

def encode_truncation_test():
    return tokenizer_default(test["text"], truncation=True, padding="max_length", max_length=1024, return_special_tokens_mask=True)


train = Field(sequential=True, use_vocab=True,tokenize=encode_truncation_train,lower=True)
test = Field(sequential=True, use_vocab=True,tokenize=encode_truncation_test,lower=True)
