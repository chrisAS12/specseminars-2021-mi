from transformers import RobertaForMaskedLM, RobertaConfig, AdamW, BertTokenizerFast
import torch

sentences_path = "data\sentences_seperated_by_lines_format.txt"
tokenizer_path = "tokenizer_30522_default_vocab_truncation_1024"

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)


def tokenizeText(text):
    batch = tokenizer(text, max_length=1024,
                    padding='max_length', truncation=True)
    return batch


def getText():
    with open(sentences_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    return lines

text = getText()

config = RobertaConfig(
    vocab_size=30522,
    num_attention_heads=12,
    num_hidden_layers=6
)

model = RobertaForMaskedLM(config)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-4)

epochs = 2
for epoch in range(epochs):
    for i in text:
        optimizer.zero_grad()
        tokenizedText = tokenizeText(i)
        input_ids = tokenizedText['input_ids']
        attention_mask = tokenizedText['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}')
        print(loss.item())
        #model.save_pretrained('./mybert_' + str(epoch) + '_1')
        # print("saved!")
