import streamlit as st
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from emat_model import EMATModel
from dataset import MTCEMATDataset
from confidence_model import ConfidenceScorer
from train import train, predict_completion
from example_data import phrase_vocab, phrase_to_id, id_to_phrase, train_data
from sentence_transformers import SentenceTransformer
from saliency_model import SaliencyModel  
import torch.nn.functional as F

st.title("🧠 EMAT Demo: Multi-Token Completion")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Load EMAT model
model = EMATModel("bert-base-uncased", phrase_vocab_size=len(phrase_vocab))
model.load_state_dict(torch.load("emat_model.pth", map_location=device))
if not isinstance(model, torch.nn.Module) or any(
    p.device.type == "meta" for p in model.parameters()
):
    del model
    model = EMATModel("bert-base-uncased", phrase_vocab_size=len(phrase_vocab))
    model.load_state_dict(torch.load("emat_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ✅ Load Confidence Scorer
confidence_model = ConfidenceScorer(len(phrase_vocab))
confidence_model.load_state_dict(torch.load("confidence_model.pth", map_location=device))
if not isinstance(confidence_model, torch.nn.Module) or any(
    p.device.type == "meta" for p in confidence_model.parameters()
):
    del confidence_model
    confidence_model = ConfidenceScorer(len(phrase_vocab))
    confidence_model.load_state_dict(torch.load("confidence_model.pth", map_location=device))
confidence_model = confidence_model.to(device)
confidence_model.eval()

# ✅ Load Sentence Transformer and Saliency Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embed_dim = embedder.get_sentence_embedding_dimension()
saliency_model = SaliencyModel(input_dim=2*embed_dim)
saliency_model.load_state_dict(torch.load("saliency_model.pth", map_location=device))
if not isinstance(saliency_model, torch.nn.Module) or any(
    p.device.type == "meta" for p in saliency_model.parameters()
):
    del saliency_model
    saliency_model = SaliencyModel(input_dim=2*embed_dim)
    saliency_model.load_state_dict(torch.load("saliency_model.pth", map_location=device))
saliency_model = saliency_model.to(device)
saliency_model.eval()

# Optional: training setup
dataset = MTCEMATDataset(tokenizer, train_data, phrase_to_id)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
optimizer = torch.optim.Adam(model.emat_decoder.parameters(), lr=1e-4)

st.success("✅ Models loaded: `emat_model.pth`, `confidence_model.pth`, and `saliency_model.pth`")

# Prediction UI
st.header("📝 Predict Multi-Token Phrase")
user_input = st.text_input("Enter a sentence with [MASK] token:", 
                           "The current WWE Undisputed Champion, [MASK], defeated Cody Rhodes.")

if st.button("Predict"):
    top_k = min(5, len(phrase_vocab))
    predictions = predict_completion(model, user_input, tokenizer, id_to_phrase, top_k=top_k)

    with torch.no_grad():
        # ---------- Get EMAT logits ----------
        encoding = tokenizer(user_input, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        logits = model(input_ids, attention_mask, torch.tensor([mask_index]).to(device))
        normalized_logits = logits - logits.mean()

        # ---------- EMAT top-1 prediction ----------
        pred_index = torch.argmax(logits, dim=1).item()
        predicted_phrase = id_to_phrase[pred_index]

        # ---------- ACCURACY SCORE ----------
        accuracy_score = confidence_model(normalized_logits).item()

        # ---------- SALIENCY SCORE ----------
        ctx_vec = embedder.encode([user_input.replace("[MASK]", "")],
                                  convert_to_tensor=True).to(device)
        phrase_vec = embedder.encode([predicted_phrase],
                                     convert_to_tensor=True).to(device)

        combined = torch.cat([ctx_vec, phrase_vec], dim=1)
        saliency_raw = saliency_model(combined)
        saliency_score = torch.sigmoid(saliency_raw).item()  # [0,1]
        saliency_score = 2 * saliency_score - 1  # scale [-1,1]

        # ---------- Hyperbolic-like Penalty ----------
        cos_sim = F.cosine_similarity(ctx_vec, phrase_vec).item()
        hyp_score = torch.tanh(torch.tensor(3 * (cos_sim - 0.5)))

        # ---------- Training Data Match ----------
        def normalize_text(text):
            return text.lower().replace(".", "").replace(",", "").strip()
        is_in_train = any(
            normalize_text(user_input.replace("[MASK]", predicted_phrase)) ==
            normalize_text(ex[0].replace("[MASK]", ex[1]))
            for ex in train_data
        )

        # ---------- FINAL COMBINED CONFIDENCE ----------
        if is_in_train:
            final_score = min(1.0, 0.9 * accuracy_score + 0.5)
        elif accuracy_score > 0.9 and hyp_score > -0.5:
            final_score = 0.9 * accuracy_score + 0.2 * hyp_score
        elif hyp_score < -0.85:
            final_score = 0.4 * accuracy_score + 2.0 * hyp_score
        elif hyp_score < -0.6:
            final_score = 0.5 * accuracy_score + 1.0 * hyp_score
        elif hyp_score > 0.5 and saliency_score >= 0.5:
            final_score = 0.6 * accuracy_score + 0.6 * saliency_score + 1.0 * hyp_score
        else:
            final_score = 0.5 * accuracy_score + 0.5 * saliency_score + 1.5 * hyp_score

        final_conf = max(min(final_score, 1), -1)

    st.write(f"Combined Confidence Score: {final_conf:.2f}")

    st.markdown(f"""
    **Breakdown:**
    - Accuracy Score: `{accuracy_score:.2f}`
    - Saliency Score: `{saliency_score:.2f}`
    - Hyperbolic Penalty: `{hyp_score:.2f}`
    """)

    st.subheader("🔮 Top Predictions:")
    for i, (phrase, prob) in enumerate(predictions, 1):
        st.write(f"{i}. **{phrase}** — `{prob * 100:.2f}%`")
