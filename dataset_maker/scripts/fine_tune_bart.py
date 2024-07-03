import json
from transformers import BartForSequenceClassification, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load dataset
with open("../data/bartset.json", "r") as file:
    data = json.load(file)

# Preprocess dataset
def preprocess_data(data):
    prompts = []
    contents = []
    relevances = []
    for item in data:
        prompts.append(item["prompt"])
        contents.append(item["content"])
        relevances.append(item["relevance"])
    return {"prompt": prompts, "content": contents, "relevance": relevances}

processed_data = preprocess_data(data)
dataset = Dataset.from_dict(processed_data)

# Tokenize dataset
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset into train and validation
dataset_dict = tokenized_dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})

# Fine-tune BART model
model = BartForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=2)

training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="../results/logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("../results/fine-tuned-bart")
tokenizer.save_pretrained("../results/fine-tuned-bart")
