import torch
import torch.nn.functional as F

def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        correct = 0
        total = 0
        input_ids, attention_mask, mask_index, labels = [x.to(device) for x in batch]
        logits = model(input_ids, attention_mask, mask_index)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        optimizer.step()
        if total > 0:
            acc = 100 * correct / total
            print(f"Accuracy: {acc:.2f}%")

def predict_completion(model, sentence, tokenizer, id_to_phrase, top_k=5):
    model.eval()
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

    with torch.no_grad():
        logits = model(
            input_ids.to(model.bert.device),
            attention_mask.to(model.bert.device),
            mask_index=torch.tensor([mask_index]).to(model.bert.device)
        )
        probs = F.softmax(logits, dim=1)
        top_preds = torch.topk(probs, top_k).indices[0].tolist()
        top_probs = probs[0][top_preds].tolist()
        return [(id_to_phrase[i], round(p, 4)) for i, p in zip(top_preds, top_probs)]
