import requests
from googlesearch import search
from bs4 import BeautifulSoup
from transformers import BartForSequenceClassification, BartTokenizer
import torch
import json
import os

# Function to download data from a URL
def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return ""

# Function to check relevance using fine-tuned BART model
def check_relevance(model, tokenizer, content):
    inputs = tokenizer(content, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    relevance = torch.argmax(logits, dim=1).item()
    return relevance

# Main function to search, download, and filter data
def dataset_maker(prompt, num_results=10, model_path="../results/fine-tuned-bart"):
    # Search Google for the prompt
    search_results = list(search(prompt, num_results=num_results))

    # Load the fine-tuned BART model and tokenizer
    model = BartForSequenceClassification.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)

    dataset = []

    for url in search_results:
        content = download_data(url)
        if content:
            relevance = check_relevance(model, tokenizer, content)
            if relevance == 0:  # Relevant
                dataset.append({"prompt": prompt, "content": content, "relevance": relevance})
        
        # Check size limit of 50MB
        if len(json.dumps(dataset).encode('utf-8')) > 50 * 1024 * 1024:
            break

    return dataset

# Example usage
prompt = "Artificial Intelligence"
dataset = dataset_maker(prompt)
output_path = "../data/new_dataset.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=4)
print(f"Dataset saved to {output_path}")
