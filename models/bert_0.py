from transformers import pipeline, RobertaTokenizer

model_path = "tokenizer_roberta"

def predict(sentence, symbol_to_replace):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    try:
        fill_mask = pipeline(
            "fill-mask",
            model='mybert_0_30',
            tokenizer=tokenizer
        )
        sentence = sentence.replace(symbol_to_replace, "<mask>")
        return fill_mask(sentence)
    except:
       print("Add only one symbol to replace, please.")
       return -1
