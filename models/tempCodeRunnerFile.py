from tokenizers import Tokenizer
from BERTTokenizer import generate_tokenizer_BertWordPieceTokenizer, generate_tokenizer_ByteLevelBPETokenizer
from transformers import BertTokenizer, AutoTokenizer, AutoConfig, pipeline, BertTokenizerFast


model_path = "tokenizer_2"

def classification():
    pipe_classification = pipeline("text-classification")
    print(pipe_classification("Checking how positive this thingy ma jig is. :D"))

def predict():
    #tokenizer = BertTokenizer.from_pretrained('bert_tokens_1')
    #tokenizer = AutoTokenizer.from_pretrained("chrisAS12/specseminars/tokenizer_1")
    #tokenizer = generate_tokenizer_BertWordPieceTokenizer()
    #tokenizer = generate_tokenizer_ByteLevelBPETokenizer()
    #tokenizer = AutoTokenizer.from_pretrained("bert_byte_0")
    #tokenizer = RobertaTokenizer.from_pretrained('C:\\Users\\chris\\Desktop\\specseminars-2021-mi\\tokenizer_1', max_len=512)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    fill_mask = pipeline('fill-mask', model='mybert_0', tokenizer=tokenizer)
    #test_tokenizer(tokenizer, fill_mask)
    #print(fill_mask(f"This course is really {tokenizer.mask_token}."))
    print(fill_mask(f"šodien ir skaista {fill_mask.tokenizer.mask_token}")[0]['sequence'])

predict()
