from transformers import pipeline, BertTokenizerFast

model_path = "tokenizer_MAX_vocab"


def classification():
    pipe_classification = pipeline("text-classification")
    print(pipe_classification("Checking how positive this thingy ma jig is. :D"))


def predict(sentence, symbol_to_replace):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    try:
        fill_mask = pipeline(
            "fill-mask",
            model='mybert_0',
            tokenizer=tokenizer
        )
        sentence = sentence.replace(symbol_to_replace, "[MASK]")
        return fill_mask(sentence)[1]['sequence']
    except:
        print("Add only one symbol to replace, please.")
        return -1
