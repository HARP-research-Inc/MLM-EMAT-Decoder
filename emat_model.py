from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch

class EMATModel(nn.Module):
    def __init__(self, base_model_name, phrase_vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)
        self.bert.requires_grad_(False)  # freeze BERT
        hidden_size = self.bert.config.hidden_size
        self.emat_decoder = nn.Linear(hidden_size, phrase_vocab_size)

    def forward(self, input_ids, attention_mask, mask_index):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.last_hidden_state  # (B, T, H)
        mask_embeds = hidden_states[torch.arange(hidden_states.size(0)), mask_index]  # (B, H)
        logits = self.emat_decoder(mask_embeds)  # (B, V)
        return logits
