import gzip
import pickle
import numpy as np

def load_numberbatch_embeddings(path="numberbatch-en-19.08.txt.gz"):
    embeddings = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            word = parts[0].replace("/c/en/", "").split("_")[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

embeddings = load_numberbatch_embeddings("numberbatch-en-19.08.txt.gz")

with open("numberbatch_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Saved to numberbatch_embeddings.pkl")
