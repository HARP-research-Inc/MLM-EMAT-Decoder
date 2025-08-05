import torch
import pickle
import os
import random
import nltk
from nltk.corpus import brown

# ── ensure NLTK data is available ───────────────────────
nltk.download("brown", quiet=True)
# the averaged_perceptron_tagger sometimes lives under different names:
try:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except:
    nltk.download("averaged_perceptron_tagger", quiet=True)


# ── hand-crafted seed vocabulary ────────────────────────
phrase_vocab = [
    "New York City", "John Cena", "Paris", "artificial intelligence", "Donald Trump",
    "London", "Lionel Messi", "Cristiano Ronaldo", "pasta", "Italian",
    "French cuisine", "Indian food", "tacos", "cheesecake", "sauce",
    "Al-Nassr", "flavor"
]
# build our lookup tables
phrase_to_id = {p: i for i, p in enumerate(phrase_vocab)}
id_to_phrase = {i: p for p, i in phrase_to_id.items()}


# ── your hand-crafted training sentences ────────────────
train_data = [
    ("[MASK] is a major metropolitan area in the US.", "New York City"),
    ("The field of [MASK] is rapidly growing.", "artificial intelligence"),
    ("The 45th president of the US, [MASK], was elected in 2008.", "Donald Trump"),
    ("[MASK] defeated CM Punk at Night of Champions 2025.", "John Cena"),
    ("The capital of France is [MASK].", "Paris"),
    ("The capital of the United Kingdom is [MASK].", "London"),
    ("Big Ben is one of the most iconic landmarks in [MASK].", "London"),
    ("[MASK] is known for its red buses, royal guards, and history.", "London"),
    ("The London Eye offers a great view of [MASK].", "London"),
    ("Many tourists visit Buckingham Palace in [MASK].", "London"),
    ("The Tower of [MASK] is a historic castle on the River Thames.", "London"),
    ("[MASK] has won the most Ballon d'Or's.", "Lionel Messi"),
    ("Cristiano Ronaldo currently plays for [MASK].", "Al-Nassr"),
    ("[MASK] is considered one of the greatest footballers in history.", "Lionel Messi"),
    ("[MASK] has played for both Real Madrid and Manchester United.", "Cristiano Ronaldo"),
    ("The World Cup was finally won by [MASK] in 2022.", "Lionel Messi"),
    ("[MASK] is the all-time top scorer in the Champions League.", "Cristiano Ronaldo"),
    ("Argentina's national team is led by [MASK].", "Lionel Messi"),
    ("Portugal's football icon is [MASK].", "Cristiano Ronaldo"),
    ("The best-rated Italian restaurant in town serves [MASK].", "pasta"),
    ("According to critics, [MASK] is the most popular cuisine worldwide.", "Italian"),
    ("Michelin stars were awarded to the chef for their [MASK].", "French cuisine"),
    ("He gave the dish five stars for its [MASK].", "flavor"),
    ("[MASK] is known for its spicy and bold flavors.", "Indian food"),
    ("A famous street food in Mexico is [MASK].", "tacos"),
    ("For dessert, they ordered the signature [MASK].", "cheesecake"),
    ("The dish was praised for its rich [MASK].", "sauce"),
] * 10  


# ── now auto-generate (and cache) Brown‐corpus masks ──────
def generate_masked_dataset(num_samples=1_000, save_path="masked_dataset.pkl"):
    # if we’ve done this before, just load & expand vocab
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            cached = pickle.load(f)
        print(f"✅ Loaded cached masked dataset from {save_path}")
        for _, tgt in cached:
            if tgt not in phrase_to_id:
                new_id = len(phrase_vocab)
                phrase_to_id[tgt] = new_id
                id_to_phrase[new_id] = tgt
                phrase_vocab.append(tgt)
        return cached

    # otherwise, build it:
    sentences = brown.sents()
    dataset = []
    for sent in sentences:
        if len(dataset) >= num_samples:
            break

        tagged = nltk.pos_tag(sent)
        noun_indices = [i for i, (_, t) in enumerate(tagged) if t.startswith("NN")]
        if not noun_indices:
            continue

        idx = random.choice(noun_indices)
        tgt = tagged[idx][0]
        masked = sent.copy()
        masked[idx] = "[MASK]"
        masked_sent = " ".join(masked)

        # extend your vocab on-the-fly:
        if tgt not in phrase_to_id:
            new_id = len(phrase_vocab)
            phrase_to_id[tgt] = new_id
            id_to_phrase[new_id] = tgt
            phrase_vocab.append(tgt)

        dataset.append((masked_sent, tgt))

    # cache for next time
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"✅ Generated {len(dataset)} masked sentences. Saved to {save_path}")
    return dataset


# load / generate & extend vocab
extra_data = generate_masked_dataset(num_samples=1_000)
train_data.extend(extra_data)

print(f"Final training dataset size: {len(train_data)}")
print(f"Vocabulary size: {len(phrase_vocab)}")


# ── negative‐case examples for confidence training ───────
negative_confidence_examples = [
    # correct → 1.0
    (torch.tensor([3.2, 0.4, -0.5, 1.0, 0.1]), 0, 1.0),
    (torch.tensor([0.2, 3.1, 0.9, -0.3, 0.7]), 1, 1.0),
    # semantically related → 0.0
    (torch.tensor([2.9, 0.3, 0.1, 1.0, 0.7]), 3, 0.0),
    (torch.tensor([0.5, 0.1, 2.8, 1.3, 0.4]), 2, 0.0),
    # totally irrelevant → –1.0
    (torch.tensor([3.5, 0.2, 0.1, -1.0, 0.0]), 0, -1.0),
    (torch.tensor([0.1, -0.2, 3.6, 0.0, 0.5]), 1, -1.0),
]

print(f"✅ Final training set size: {len(train_data)}")
print(f"✅ phrase_vocab size:    {len(phrase_vocab)}")
print("Some new Brown nouns in phrase_vocab:", phrase_vocab[-10:])
