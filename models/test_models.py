from transformers import pipeline
from tokenizers import Tokenizer

def classification():
    pipeClassification = pipeline("text-classification")
    print(pipeClassification("Checking how positive this thingy ma jig is. :D"))
    
def bertModel():
    my_tokenizer = Tokenizer.from_file("tokenizer_load")    
    fill = pipeline('fill-mask', model='mybert_0', tokenizer=my_tokenizer)
    
bertModel()