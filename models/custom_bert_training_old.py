from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer,AdamW 
import torch

sentences_path = "data/sentences_seperated_by_lines_format.txt"
word_list_path = "data\words_list.txt"
pretrained_path = 'C:\\Users\\chris\\Desktop\\specseminars-2021-mi\\tokenizer_1'

dataset_path = 'dataset_new_1.pt'

def getParameters():
    f = open(word_list_path, encoding="utf-8")
    vocab = f.read()
    size = len(set(vocab.split()))
    f = open(sentences_path, encoding="utf-8")
    text = f.read().split("\n")
    return vocab, size, text

def saveTokenizer(tokenizer):
    tokenizer.save_model("/tokenizer")


def createBatch(roberta):
    with open(sentences_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    batch = roberta(lines, max_length=512, padding='max_length', truncation=True)
    return batch

class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings['input_ids'].shape[0]

        def __getitem__(self, i):
            return {key: tensor[i] for key, tensor in self.encodings.items()}

def create_encodings(batch):
    labels = torch.tensor([x for x in batch.input_ids])
    mask = torch.tensor([x for x in batch.attention_mask])  
    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        input_ids[i, selection] = 3 
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings

def load_dataset():
    dataset = torch.load(dataset_path, encoding='ascii')
    return dataset
    
vocab, size, text = getParameters()

#tokenizer = generate_tokenizer_ByteLevelBPETokenizer()
#os.mkdir('./tokenizer_1')
#tokenizer.save_model('./tokenizer_1', 'bert_tokens')

roberta = RobertaTokenizer.from_pretrained(pretrained_path, max_len=512)

batch = createBatch(roberta)
dataset = Dataset(create_encodings(batch))
torch.save(dataset, dataset_path)
print(dataset)
#dataset = load_dataset()

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

config = RobertaConfig(
    vocab_size=30522, 
    num_attention_heads=12,
    num_hidden_layers=6
    )
model = RobertaForMaskedLM(config)

#print(model)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-4)

epochs = 2
torch.cuda.empty_cache()
for epoch in range(epochs):
    for i in loader:
        optimizer.zero_grad()
        input_ids = i['input_ids'].to(device)
        attention_mask = i['attention_mask'].to(device)
        labels = i['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}')
        print(loss.item())
        model.save_pretrained('./mybert_' + str(epoch) + '_1')
        print("saved!")
        
