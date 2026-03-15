import pandas as pd             # Use to deal with dataframes
import torch                    # Deep Learning technology
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)                               # Loading tokenizer + pretrained RoBERTa model
from sklearn.feature_extraction.text import TfidfVectorizer  # For enhanced feature extraction from text to vectors
from sklearn.metrics.pairwise import cosine_similarity       # To measure text similarity
import numpy as np              # Numerical operations (optional support)

#--------------------- CONFIGURATION ----------------------

"""
Number of most relevant novel chunks to check per statement
Higher value = stricter checking, but slower execution
"""
TOP_K = 3

# ---------------- LOADING TRAIN & TEST DATA ----------------
"""
Engineering Dataframes to store train and test csv data 
"""
train_df = pd.read_csv("train.csv")  
test_df = pd.read_csv("test.csv")    

# ---------------- LOADING NOVEL TEXT FILES ----------------
"""
Proper execution of file handling to read novels provided in text format
"""
print("Loading novel texts...")       # Status message for clarity

with open("novel/The Count of Monte Cristo.txt", encoding="utf-8", errors="ignore") as f:
    monte_text = f.read()             # Loading full Monte Cristo novel text

with open("novel/In search of the castaways.txt", encoding="utf-8", errors="ignore") as f:
    castaway_text = f.read()          # Loading full Castaways novel text

# ---------------- SPLITING NOVELS INTO SMALL CHUNKS ----------------
"""
RoBERTa model can take upto 512 tokens at max and we have 100k+ tokens in text files.
We are devloping function namely "chunk_text" which seperates a fixed size of tokens into a chunk.
"""
def chunk_text(text, chunk_size=400):
    words = text.split()              # Breaking novel into words
    return [
        " ".join(words[i:i + chunk_size])  # Combining words back into fixed-size chunks
        for i in range(0, len(words), chunk_size)
    ]

print("Chunking novels...")            
monte_chunks = chunk_text(monte_text) # Function call for monte file
castaway_chunks = chunk_text(castaway_text) #Function call for castaway file

# ---------------- ENHANCED CONTEXT RETRIEVAL (TOP-K) ----------------

def get_top_k_chunks(statement, chunks, k=3):
    """
    Finds the top K most relevant chunks from the novel for a given statement.    
    """
    vectorizer = TfidfVectorizer(stop_words="english")  
    # Remove common words and focus on meaningful terms

    vectors = vectorizer.fit_transform([statement] + chunks)
    # Convert statement + chunks into numerical vectors

    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    # Calculate similarity between statement and each chunk

    top_k_indices = similarities.argsort()[-k:][::-1]
    # Pick indices of top K most similar chunks

    return [chunks[i] for i in top_k_indices]
    # Return the actual chunk texts

# ---------------- LOADING ROBERTA NLI MODEL ----------------

print("Loading RoBERTa-large-mnli model...")  # Model loading message

tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
# Tokenizer converts text into model-readable tokens

model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
# Pretrained NLI model for entailment / contradiction detection

model.eval()  
# Switch model to inference mode (no training)

label_map = {
    0: "contradiction",
    1: "neutral",
    2: "entailment"
}
# Model output labels explained

# ---------------- ENHANCED CONSISTENCY CHECK ----------------

def predict_consistency_enhanced(statement, context_chunks):
    """
    Checks multiple chunks:
    - If any chunk contradicts → inconsistent
    - Otherwise → consistent
    """

    for chunk in context_chunks:
        inputs = tokenizer(
            chunk,
            statement,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        # Prepare chunk + statement for the model

        with torch.no_grad():
            logits = model(**inputs).logits
            # Run inference without gradients

        prediction_index = torch.argmax(logits).item()
        predicted_label = label_map[prediction_index]
        # Convert model output to readable label

        if predicted_label == "contradiction":
            return 0  
            # One contradiction is enough to mark inconsistent

    return 1  
    # If no contradictions found, mark consistent

# ---------------- RUN MODEL ON TEST SET ----------------

print(f"Running inference on {len(test_df)} items with Top-{TOP_K} retrieval...")
# Inform user about process start

predictions = []

for index, row in test_df.iterrows():
    statement = row["content"]        # Extract the backstory statement

    if index % 10 == 0:
        print(f"Processing row {index}/{len(test_df)}...")
        # Simple progress update

    if "monte" in row["book_name"].lower():
        chunks = monte_chunks         # Use Monte Cristo chunks
    else:
        chunks = castaway_chunks      # Use Castaways chunks

    top_contexts = get_top_k_chunks(statement, chunks, k=TOP_K)
    # Retrieve most relevant K chunks

    result = predict_consistency_enhanced(statement, top_contexts)
    # Check consistency across all selected chunks

    predictions.append(result)
    # Store prediction

# ---------------- FINAL FORMATTING ----------------

test_df["prediction"] = predictions   # Add predictions to dataframe

submission_df = test_df[["id", "prediction"]]
# Kept only required submission columns

print("\nFinal Prediction Counts:")
print(submission_df["prediction"].value_counts())
# Show how many are consistent vs inconsistent

submission_df.to_csv("result.csv", index=False)
# Save final output file

print("Saved predictions to result.csv")  # Done 🎉
