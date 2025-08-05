from torch.utils.data import Dataset

class MTCEMATDataset(Dataset):
    def __init__(self, tokenizer, data, phrase_to_id):
        self.tokenizer   = tokenizer
        self.phrase_to_id = phrase_to_id

        # only keep examples whose phrase is in our vocab AND whose sentence
        # actually yields a [MASK] token_id when tokenized
        filtered = []
        for sent, phrase in data:
            if phrase not in phrase_to_id:
                continue
            enc = tokenizer(
                sent,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )
            input_ids = enc["input_ids"].squeeze(0)
            # if there is at least one mask_token_id in input_ids, keep it
            if (input_ids == tokenizer.mask_token_id).any():
                filtered.append((sent, phrase))

        self.data = filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, phrase = self.data[idx]
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        input_ids     = enc["input_ids"].squeeze(0)
        attention_mask= enc["attention_mask"].squeeze(0)
        # now we know there is at least one MASK
        mask_index    = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
        label         = self.phrase_to_id[phrase]
        return input_ids, attention_mask, mask_index, label
