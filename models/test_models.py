from transformers import pipeline
from tokenizers import Tokenizer
from transformers import BertTokenizer

def classification():
    pipeClassification = pipeline("text-classification")
    print(pipeClassification("Checking how positive this thingy ma jig is. :D"))
    
def predict():
    tokenizer = BertTokenizer.from_pretrained('bert_tokens_1')
    fill = pipeline('fill-mask', model='mybert_0', tokenizer=tokenizer)
    print(fill(f'kā tev šodien {fill.tokenizer.mask_token}?'))   
    
predict()