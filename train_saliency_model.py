# train_saliency_model.py
import torch, random
from saliency_model import SaliencyModel
from sentence_transformers import SentenceTransformer
from example_data import train_data, phrase_vocab

# ------------------------------------------------------------------
# Build dataset: Positive and negative examples
data_pairs = []
for sentence, correct_phrase in train_data:
    # Positive example
    data_pairs.append((sentence.replace("[MASK]", ""), correct_phrase, 1.0))

    # Generate negatives (sample from vocab that is NOT the correct phrase)
    negatives = [p for p in phrase_vocab if p != correct_phrase]
    neg_samples = random.sample(negatives, k=min(2, len(negatives)))
    for neg in neg_samples:
        data_pairs.append((sentence.replace("[MASK]", ""), neg, 0.0))

random.shuffle(data_pairs)

# ------------------------------------------------------------------
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
embed_dim = embedder.get_sentence_embedding_dimension()

def batch_encode(texts, batch_size=16):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = embedder.encode(batch, convert_to_tensor=True)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0).to(device)

# Encode in batches
ctx_texts    = [c for c, _, _ in data_pairs]
phrase_texts = [p for _, p, _ in data_pairs]
labels       = torch.tensor([l for _, _, l in data_pairs]).unsqueeze(1).to(device)

ctx_embs    = batch_encode(ctx_texts, batch_size=16)
phrase_embs = batch_encode(phrase_texts, batch_size=16)

combined = torch.cat([ctx_embs, phrase_embs], dim=1)  # Shape: (N, 1536)

# ------------------------------------------------------------------
# Train saliency model
model = SaliencyModel(input_dim=2*embed_dim).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = torch.nn.BCELoss()

for epoch in range(30):  # more epochs now possible
    model.train()
    opt.zero_grad()
    out = model(combined)
    loss = lossfn(out, labels.float())
    loss.backward()
    opt.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "saliency_model.pth")
print("✅ Saved saliency_model.pth")
