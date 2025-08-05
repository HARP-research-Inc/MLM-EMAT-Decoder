# --- new imports ---
from saliency_model import SaliencyModel
from sentence_transformers import SentenceTransformer
import torch
from saliency_model import SaliencyScorer

# ---------------------

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saliency model & embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

saliency_model = SaliencyScorer(input_dim=embedder.get_sentence_embedding_dimension()).to(device)
saliency_model.load_state_dict(torch.load("saliency_model.pth", map_location=device))
saliency_model = saliency_model.to(device).eval()
