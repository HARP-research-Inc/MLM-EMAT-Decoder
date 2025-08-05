import torch
import torch.nn as nn
import random
from transformers import BertTokenizer
from confidence_model import ConfidenceScorer
from emat_model import EMATModel
from example_data import phrase_vocab, id_to_phrase, train_data

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Load EMAT model
emat_model = EMATModel("bert-base-uncased", phrase_vocab_size=len(phrase_vocab))
emat_model.load_state_dict(torch.load("emat_model.pth", map_location=device))
emat_model = emat_model.to(device)
emat_model.eval()

logit_samples = []
targets = []

# --- Build training data from real dataset ---
for sentence, expected_phrase in train_data:
    with torch.no_grad():
        encoding = tokenizer(sentence, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        logits = emat_model(input_ids, attention_mask, torch.tensor([mask_index]).to(device))
        normalized_logits = logits - logits.mean()

        pred_index = torch.argmax(logits).item()
        pred_phrase = id_to_phrase[pred_index]

        # Score logic: 1.0 = correct, 0.0 = same type but wrong, -1.0 = irrelevant
        if pred_phrase.lower() == expected_phrase.lower():
            score = 1.0
        elif any(word in expected_phrase.lower() for word in pred_phrase.lower().split()):
            score = 0.0
        else:
            score = -1.0

        logit_samples.append(normalized_logits.squeeze())
        targets.append(score)

# --- Add synthetic negative examples ---
synthetic_negatives = [
    ("The tastiest dessert ever is [MASK].", "Paris", -1.0),
    ("My favorite car brand is [MASK].", "Donald Trump", -1.0),
    ("The capital of Mars is [MASK].", "John Cena", -1.0),
    ("He scored the winning goal in [MASK].", "Paris", 0.0),  # semantically related
    ("She is famous for her work in [MASK].", "artificial intelligence", 0.0),
]

for sentence, fake_phrase, score in synthetic_negatives:
    with torch.no_grad():
        encoding = tokenizer(sentence, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        logits = emat_model(input_ids, attention_mask, torch.tensor([mask_index]).to(device))
        normalized_logits = logits - logits.mean()
        logit_samples.append(normalized_logits.squeeze())
        targets.append(score)

# --- Prepare tensors ---
X = torch.stack(logit_samples)
y = torch.tensor(targets).unsqueeze(1)

# --- Train Confidence Model ---
model = ConfidenceScorer(input_dim=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y.to(device))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# --- Save ---
torch.save(model.state_dict(), "confidence_model.pth")
print("✅ Retrained and saved confidence_model.pth with real + synthetic negatives")
