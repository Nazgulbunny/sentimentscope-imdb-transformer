
# SentimentScope – Transformer-Based Sentiment Analysis (From Scratch)

## Overview

This project implements a transformer-based sentiment classification model from scratch using PyTorch and applies it to the IMDB movie reviews dataset.

Unlike many NLP projects that rely on pretrained models, this implementation builds the core transformer architecture manually, including multi-head attention and feed-forward layers. The model is then adapted for binary classification (positive vs negative sentiment).

The goal of this project was to deeply understand how transformer models work internally and apply them to a real-world NLP task.

---

## Dataset

The model is trained and evaluated on the IMDB dataset:

- 25,000 labeled training reviews  
- 25,000 labeled test reviews  
- Binary sentiment labels:
  - `1` = Positive  
  - `0` = Negative  

The dataset is loaded directly from raw text files and processed using a custom data pipeline.

---

## Model Architecture

The model includes:

- Token embeddings  
- Positional embeddings  
- 4 Transformer blocks:
  - Multi-head self-attention  
  - Feed-forward layers  
  - Layer normalization  
- Mean pooling to convert token-level embeddings into a sequence-level representation  
- Linear classification head for binary output  

### Key Configuration

- Embedding dimension: **128**  
- Number of layers: **4**  
- Number of attention heads: **4**  
- Context size (max sequence length): **128 tokens**  
- Optimizer: **AdamW**  
- Loss function: **CrossEntropyLoss**

---

## Training Details

- Batch size: 32  
- Epochs: 3  
- Learning rate: 3e-4  
- Tokenization: `bert-base-uncased` tokenizer  
- Padding and truncation applied to 128 tokens  

Validation accuracy was computed after each epoch.

---

## Results

- Validation Accuracy: ~78–79%  
- Test Accuracy: **76.10%**

The project requirement was to exceed 75% test accuracy, which was successfully achieved.

A trained model checkpoint is saved locally for reproducibility.

---

## Key Learnings

- Transformer architectures can be adapted from generation tasks to classification tasks by introducing a pooling mechanism and classification head.
- Mean pooling provides a simple but effective way to summarize token-level embeddings.
- Implementing the transformer manually gives deeper understanding of:
  - Attention mechanisms  
  - Residual connections  
  - Layer normalization  
  - Training dynamics  
- Proper data shuffling and validation splitting are essential for reliable evaluation.

---

## Project Structure

sentimentscope-imdb-transformer/
│
├── notebooks/
│ └── SentimentScope.ipynb
│
├── checkpoints/ # saved locally (not committed)
│
├── .gitignore
├── README.md

---



---

## How to Run

1. Clone the repository  
2. Download the IMDB dataset (`aclImdb_v1.tar.gz`)  
3. Place it in the project root  
4. Extract it:
   ```bash
   tar -xzf aclImdb_v1.tar.gz

---

Create a virtual environment

Install dependencies:

pip install torch transformers pandas matplotlib

Open the notebook and run all cells

## Possible Improvements (Note to myself)

Increase embedding size or number of transformer layers

Train for more epochs

Add learning rate scheduling

Implement attention masking for padding tokens

Compare performance with pretrained models (e.g., BERT fine-tuning)

