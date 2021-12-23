from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def predict(model, tokenizer):
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )
    print('Fill blank: ')
    print(fill_mask(f"The University of Latvia is so {fill_mask.tokenizer.mask_token}."))

    print('Fill blank: ')
    print(fill_mask(f"This course is really {fill_mask.tokenizer.mask_token}."))

print('predicting ...')
predict(model, tokenizer)