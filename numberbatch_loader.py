import numpy as np

def load_numberbatch_embeddings(filepath, phrase_vocab):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0].replace('/c/en/', '').replace('_', ' ')
            if word.lower() in [p.lower() for p in phrase_vocab]:
                vector = np.array(list(map(float, parts[1:])))
                embeddings[word.lower()] = vector
    return embeddings
