import torch
import torch.nn as nn
import torch.nn.functional as F
from confidence_model import ConfidenceScorer

def train_confidence_model(logits_list, labels_list, phrase_vocab_size, epochs=10):
    model = ConfidenceScorer(phrase_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    logits_tensor = torch.stack(logits_list)  # shape: [num_samples, vocab_size]
    labels_tensor = torch.tensor(labels_list).float().unsqueeze(1)  # shape: [num_samples, 1]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(logits_tensor)
        loss = loss_fn(preds, labels_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = ((preds > 0.5) == labels_tensor.bool()).sum().item() / len(labels_tensor)
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Accuracy: {acc*100:.2f}%")

    return model
