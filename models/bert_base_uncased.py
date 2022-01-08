from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def predict(sentence, symbol_to_replace, model, tokenizer):
    try:
        fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer=tokenizer
        )
        sentence = sentence.replace(symbol_to_replace, "[MASK]")
        return fill_mask(sentence)[1]['sequence']
    except:
        print("Add only one symbol to replace, please.")
        return -1

def full_prediction(sentence, symbol_to_replace):
    print(sentence)
    print(f"In the sentence above, we can replace {symbol_to_replace} to this: ")
    print('\033[0m',"\033[1m",predict(sentence,symbol_to_replace, model, tokenizer),'\033[0m','\033[92m');

def test_cases():
    full_prediction("The University of Latvia is so *.",'*')
    full_prediction("This course is so -!", '-')

def make_text_green():
    print('\033[92m')

make_text_green()

while(True):
    sentence = input("Enter your sentence and add one symbol you will replace with a word later: ")
    if(sentence is None or sentence == '' or sentence == '-1'):
        break
    symbol = input("Symbol to replace: ")
    full_prediction(sentence, symbol)
    print("Enter -1 to break!")
    