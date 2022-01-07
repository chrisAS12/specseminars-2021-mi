import torch
def create_encodings(text):
    splitted = text.split('\n')
    labels = torch.tensor([x for x in splitted.input_ids])
    mask = torch.tensor([x for x in batch.attention_mask])  
    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        input_ids[i, selection] = 3 
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    return encodings

