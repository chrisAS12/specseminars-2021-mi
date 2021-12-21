from os import error
from transformers import pipeline
from tokenizers import Tokenizer

for retry in range(400):
    try:
        my_tokenizer = Tokenizer.from_file("tokenizer_load")
        break
    except:
        print("failed ", retry)
        print(error)
    
fill = pipeline('fill-mask', model='mybert_0', tokenizer=my_tokenizer)