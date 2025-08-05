import torch
from torch.utils.data import DataLoader
from emat_model import EMATModel
from dataset import MTCEMATDataset
from example_data import phrase_vocab, phrase_to_id, train_data
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize EMAT model with updated phrase_vocab size
model = EMATModel("bert-base-uncased", phrase_vocab_size=len(phrase_vocab)).to(device)

# Dataset and DataLoader
# Increase batch size for efficiency
dataset = MTCEMATDataset(model.tokenizer, train_data, phrase_to_id)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.emat_decoder.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training Loop
num_epochs = 50  # longer training since dataset is bigger
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
    # train() runs through the data; it does not return loss
    train(model, dataloader, optimizer, device)
    scheduler.step()
    print(f"Epoch {epoch+1} completed")

# Save updated EMAT model
torch.save(model.state_dict(), "emat_model.pth")
print("✅ EMAT model retrained and saved to emat_model.pth")
