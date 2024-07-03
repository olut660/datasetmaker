# Dataset Maker and BART Fine-Tuning

## Project Structure

```
dataset_maker/
│
├── data/
│   └── bartset.json
│
├── scripts/
│   ├── dataset_maker.py
│   └── fine_tune_bart.py
│
├── results/
│   └── (empty initially, will store fine-tuned model and logs)
│
└── README.md
```

## Instructions

### Fine-Tune BART Model

1. Install necessary libraries:

   ```sh
   pip install transformers datasets requests googlesearch-python beautifulsoup4
   ```

2. Save your dataset in `data/bartset.json`.

3. Run the fine-tuning script:

   ```sh
   python scripts/fine_tune_bart.py
   ```

4. The fine-tuned model will be saved in `results/fine-tuned-bart`.

### Create a Dataset

1. Ensure the fine-tuned model is saved in `results/fine-tuned-bart`.

2. Run the dataset maker script:

   ```sh
   python scripts/dataset_maker.py
   ```

3. The new dataset will be saved in `data/new_dataset.json`.

## Notes

- Ensure you have an internet connection for Google search.
- The dataset maker script currently uses a prompt and searches Google for data related to the prompt.
- The relevance of the data is checked using the fine-tuned BART model.
- The dataset maker ensures the total size of the dataset does not exceed 50MB.