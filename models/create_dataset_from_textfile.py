import spacy
import pandas as pd 
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split

text = open('data/sentences_seperated_by_lines_format.txt', encoding='utf8').read().split('\n')

raw_data = {'Text':[line for line in text[1:2000]]}

df = pd.DataFrame(raw_data, columns=['Text'])

train, test = train_test_split(df, test_size=0.2)

train.to_json('train.json', orient='records', lines=True)


#def encode_with_truncation(examples):
    #return tokenizer_default(examples["text"], truncation=True, padding="max_length", max_length=1024, return_special_tokens_mask=True)
