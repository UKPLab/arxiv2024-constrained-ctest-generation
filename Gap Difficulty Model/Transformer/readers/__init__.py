import torch
   
class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, index, encodings, labels):
        self.index = index
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.int) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        
        #print(item['idx'])
        #item['idx'] = self.index[idx]
        return item

    def __len__(self):
        return len(self.labels)
        
        
class HuggingfaceDatasetPadding(torch.utils.data.Dataset):
    def __init__(self, index, encodings, labels, max_len):
        self.index = index
        self.encodings = encodings
        self.labels = labels
        self.max_len = max_len
        self.label_pad_token_id = -100 # Masking for padding/ignore tokens
        self.attention_pad_token_id = 0 # Masking for attention
        

    def __getitem__(self, idx):
        # We need to do padding ourselves, otherwise, the collator will transform every label into an integer:
        # https://github.com/huggingface/transformers/blob/dcb08b99f44919425f8ba9be9ddcc041af8ec25e/src/transformers/data/data_collator.py#L329 
        # We also need to pad everything to max_len to allow batching to happen properly...
        item = {key: torch.tensor([list(val[idx]) + [self.attention_pad_token_id] * (self.max_len - len(val[idx]))]).flatten() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([list(self.labels[idx]) + [self.label_pad_token_id] * (self.max_len - len(self.labels[idx]))], dtype=torch.float).flatten()
        #item['idx'] = self.index[idx]
        return item, self.index[idx]

    def __len__(self):
        return len(self.labels)
        
class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, index, encodings, labels):
        self.index = index
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item['features'] = torch.tensor(self.encodings[idx]).float()
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = self.index[idx]
        return item

    def __len__(self):
        return len(self.labels)
        
        
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, index, sents, labels, embedding, max_len):
        self.index = index
        self.sents = sents
        self.labels = labels
        self.max_len = max_len
        self.embedding = embedding

    def __getitem__(self, idx):
        # Fetch feature vectors dynamically (save memory)
        item = {}
        feature_vecs = [self.embedding[tok] for i, tok in enumerate(self.sents[idx]) if i < self.max_len] # Add truncating
        # Add padding
        #item['features'] = 
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = self.index[idx]
        

        return item

    def __len__(self):
        return len(self.labels)
        
class FeatureDatasetPadding(torch.utils.data.Dataset):
    """ Simple dataset for sentence regression tasks using additional features. 
    """
    def __init__(self, index, encodings, labels, features):
        self.index = index
        self.encodings = encodings
        self.labels = labels
        self.features = features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['features'] = torch.tensor(self.features[idx])
        return item, self.index[idx]

    def __len__(self):
        return len(self.labels)
