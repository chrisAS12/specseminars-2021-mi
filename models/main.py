from tokenizers import Tokenizer
from BERTTokenizer import generate_tokenizer_BertWordPieceTokenizer
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, pipeline


def classification():
    pipe_classification = pipeline("text-classification")
    print(pipe_classification("Checking how positive this thingy ma jig is. :D"))

def test_tokenizer(tokenizer, fill_mask):
    test_sentence = "This is a [MASK] sequence"
    test_encoding = fill_mask(test_sentence)
    print(test_encoding)

def predict():
    #tokenizer = BertTokenizer.from_pretrained('bert_tokens_1')
    #tokenizer = AutoTokenizer.from_pretrained("chrisAS12/specseminars")
    tokenizer = generate_tokenizer_BertWordPieceTokenizer()
        
    fill_mask = pipeline('fill-mask', model='mybert_0', tokenizer=tokenizer)
    test_tokenizer(tokenizer, fill_mask)
    #print(fill_mask(f"This course is really {fill_mask.tokenizer.mask_token}."))
    print(fill_mask(f"This course is really [MASK]."))



predict()
