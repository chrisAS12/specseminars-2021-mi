from tokenizers import Tokenizer
from BERTTokenizer import generate_tokenizer_BertWordPieceTokenizer, generate_tokenizer_ByteLevelBPETokenizer
from transformers import pipeline, BertTokenizerFast, BertTokenizer, BertConfig, BertForPreTraining
from transformers import TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from bert_dataset import Dataset

model_path = "tokenizer_30522_default_vocab"

tokenizer_default = BertTokenizer.from_pretrained(model_path)

config = BertConfig()
model = BertForPreTraining(config)

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer_default,
    file_path="dataset_new.pt",
    block_size=512
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_default,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir= "/bert_output/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_gpu_train_batch_size= 16,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("bert_1")