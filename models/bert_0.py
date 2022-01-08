from transformers import pipeline, BertTokenizerFast, RobertaTokenizer

model_path = "tokenizer_MAX_vocab"
pretrained_path = 'C:\\Users\\chris\\Desktop\\specseminars-2021-mi\\tokenizer_1'

def classification():
    pipe_classification = pipeline("text-classification")
    print(pipe_classification("Checking how positive this thingy ma jig is. :D"))


def predict(sentence, symbol_to_replace):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    #tokenizer = RobertaTokenizer.from_pretrained(pretrained_path, max_len=128)

    #try:
    
    fill_mask = pipeline(
        "fill-mask",
        model='mybert_0_1',
        tokenizer=tokenizer
    )
    sentence = sentence.replace(symbol_to_replace, "[MASK]")
    return fill_mask(sentence)[1]['sequence']
    #except:
      # print("Add only one symbol to replace, please.")
       # return -1
