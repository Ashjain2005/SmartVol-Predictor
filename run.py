# ---------------- IMPORT REQUIRED LIBRARIES ----------------

import pandas as pd
# pandas is used for loading, reading, and manipulating CSV files
# We use it to read train.csv and test.csv and to store predictions

import torch
# torch is required to run the pretrained RoBERTa model
# It handles tensors, model execution, and inference

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)
# Hugging Face transformers library:
# - RobertaTokenizer converts text into tokens the model understands
# - RobertaForSequenceClassification loads a RoBERTa model trained for classification (MNLI)

from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF converts text into numerical vectors based on word importance

from sklearn.metrics.pairwise import cosine_similarity
# Cosine similarity measures how similar two text vectors are


# ---------------- LOAD TRAIN & TEST DATA ----------------

# Load training data (used mainly for reference/debugging)
# This dataset already has labels
train_df = pd.read_csv("train.csv")

# Load test data (no labels, model must predict consistency)
test_df = pd.read_csv("test.csv")


# ---------------- LOAD NOVEL TEXT FILES ----------------

# Read the full text of "The Count of Monte Cristo"
# encoding="utf-8" ensures text is read correctly
# errors="ignore" skips any unreadable characters
with open("novel/The Count of Monte Cristo.txt", encoding="utf-8", errors="ignore") as f:
    monte_text = f.read()

# Read the full text of "In Search of the Castaways"
with open("novel/In search of the castaways.txt", encoding="utf-8", errors="ignore") as f:
    castaway_text = f.read()


# ---------------- SPLIT NOVELS INTO SMALL CHUNKS ----------------

# WHY CHUNKING IS NECESSARY:
# - Novels are extremely long (hundreds of thousands of words)
# - Transformer models can only handle up to ~512 tokens
# - We split novels into smaller chunks so the model can process them

def chunk_text(text, chunk_size=400):
    """
    Splits a long text into smaller chunks of fixed word length

    Parameters:
    - text: full novel text
    - chunk_size: number of words per chunk

    Returns:
    - list of text chunks
    """
    words = text.split()  # Split text into individual words

    # Create chunks using sliding window approach
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# Generate chunks for both novels
monte_chunks = chunk_text(monte_text)
castaway_chunks = chunk_text(castaway_text)


# ---------------- RETRIEVE MOST RELEVANT CHUNK ----------------

# PURPOSE:
# Given a statement, we must find the most relevant part of the novel
# Steps:
# 1. Convert statement and all chunks into TF-IDF vectors
# 2. Compute cosine similarity
# 3. Select the chunk with the highest similarity score

def get_best_chunk(statement, chunks):
    """
    Finds the most relevant chunk from the novel for a given statement

    Parameters:
    - statement: the backstory statement
    - chunks: list of novel chunks

    Returns:
    - the chunk most similar to the statement
    """

    # Initialize TF-IDF vectorizer
    # stop_words="english" removes common words like 'is', 'the', 'and'
    vectorizer = TfidfVectorizer(stop_words="english")

    # Fit and transform:
    # First vector = statement
    # Remaining vectors = novel chunks
    vectors = vectorizer.fit_transform([statement] + chunks)

    # Compute cosine similarity between statement and each chunk
    similarities = cosine_similarity(vectors[0:1], vectors[1:])

    # Get index of chunk with highest similarity score
    best_chunk_index = similarities.argmax()

    # Return the most relevant chunk
    return chunks[best_chunk_index]


# ---------------- LOAD ROBERTA NLI MODEL ----------------

# We use a pretrained Natural Language Inference (NLI) model
# Task: Determine whether a statement is:
# - Entailed by the context
# - Neutral
# - Contradicted by the context

tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
# Loads tokenizer corresponding to the MNLI-trained RoBERTa model

model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
# Loads the pretrained classification model

model.eval()
# Sets the model to evaluation mode (important for inference)
# Disables dropout layers for consistent predictions

# Label mapping used internally by the MNLI model
label_map = {
    0: "contradiction",
    1: "neutral",
    2: "entailment"
}


# ---------------- CHECK CONSISTENCY ----------------

def predict_consistency(statement, context):
    """
    Determines whether a statement is consistent with the given context

    Parameters:
    - statement: backstory statement
    - context: relevant novel chunk

    Returns:
    - 0 → consistent
    - 1 → contradictory
    """

    # Tokenize context and statement as a pair
    # Format: [CLS] context [SEP] statement [SEP]
    inputs = tokenizer(
        context,
        statement,
        return_tensors="pt",
        truncation=True,     # Truncate if input exceeds max length
        max_length=512       # Maximum tokens allowed by model
    )

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get index of highest probability class
    prediction = torch.argmax(logits).item()

    # If model predicts contradiction → mark as inconsistent
    if label_map[prediction] == "contradiction":
        return 1
    else:
        return 0


# ---------------- RUN MODEL ON TEST SET ----------------

predictions = []

# Iterate through each test example
for _, row in test_df.iterrows():

    # Extract the statement text
    statement = row["content"]

    # Decide which novel to use based on book name
    if "monte" in row["book_name"].lower():
        context = get_best_chunk(statement, monte_chunks)
    else:
        context = get_best_chunk(statement, castaway_chunks)

    # Predict consistency and store result
    predictions.append(
        predict_consistency(statement, context)
    )

# Add predictions column to test dataframe
test_df["prediction"] = predictions

# Remove unnecessary columns before saving results
test_df = test_df.drop(columns=["book_name", "char", "caption", "content"])


# ---------------- FINAL COUNT ----------------

# Display number of consistent vs contradictory predictions
print(test_df["prediction"].value_counts())


# ---------------- SAVE RESULTS ----------------

# Save predictions to CSV file for submission
test_df.to_csv("test_predictions.csv", index=False)
