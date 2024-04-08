import torch
    
def load_data_regression(infile):
    token_idx = 5
    data = {}
    line_count = 0
    with open(infile,'r') as lines:
        for line in lines:
            if line.strip() == "":
                continue
            feats = line.strip().split(',')
            token = feats[token_idx]
            del feats[token_idx]
            data[line_count] = {'feature':[float(x) for x in feats[:-1]],
                         'score':float(feats[-1]),
                         'token':token}
            line_count += 1
    return data

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.index = []
        self.encodings = []
        self.labels = []
        self.ids = []
        for idx, vals in data.items():
            self.index.append(idx)
            self.ids.append(vals['token'])
            self.encodings.append(vals['feature'])
            self.labels.append(vals['score'])


    def __getitem__(self, idx):
        item = {}
        item['features'] = torch.tensor(self.encodings[idx]).float()
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = self.index[idx]
        item['id'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.index)
       
