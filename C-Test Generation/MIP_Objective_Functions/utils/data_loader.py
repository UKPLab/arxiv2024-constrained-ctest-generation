# Data loader for C-tests. Uses the DKPro TC format
import numpy as np

# Gold data loader -> tagged
def load_gold_data(infile):
    data = {}
    with open(infile,'r') as f:
        sent = {"word":[], 
                "index":[], 
                "hint":[], 
                "gap":[], 
                "error_rate":[], 
                "features":[]
                }
        idx = 0
        for line in f:
            if '----' in line:
                data[idx] = sent
                sent = {"word":[], 
                        "index":[], 
                        "hint":[], 
                        "gap":[], 
                        "error_rate":[], 
                        "features":[]
                        }
                idx += 1
                continue
            try:
                word, index, hint, gap, error_rate, feat = line.strip().split('\t')
                feat = np.fromstring(feat, sep = ',')
            except ValueError:
                word = line.strip()
                index = ""
                hint = ""
                gap = ""
                error_rate = ""
                feat = ""
            sent["word"].append(word) 
            sent["index"].append(index)
            sent["hint"].append(hint)
            sent["gap"].append(gap)
            sent["error_rate"].append(error_rate)
            sent["features"].append(feat)
    return data

# Silver data loader -> not tagged
def load_silver_data(infile):
    data = {}
    with open(infile,'r') as f:
        sent = {"word":[], 
                "hint":[], 
                "gap":[], 
                "features":[]
                }
        idx = 0
        for line in f:
            if '----' in line:
                data[idx] = sent
                sent = {"word":[], 
                        "hint":[], 
                        "gap":[], 
                        "features":[]
                        }
                idx += 1
                continue
            try:
                word, hint, gap, feat = line.strip().split('\t')
                feat = np.fromstring(feat, sep = ',')
            except ValueError:
                word = line.strip()
                hint = ""
                gap = ""
                feat = ""
            sent["word"].append(word) 
            sent["hint"].append(hint)
            sent["gap"].append(gap)
            sent["features"].append(feat)
    return data

# Silver data loader -> not tagged
def load_silver_data_instances_sents(infile):
    data = {}
    with open(infile,'r') as f:
        sent_idx = 0
        tok_idx = 0
        for line in f:
            if '----' in line:
                sent_idx += 1
                continue
            try:
                word, hint, gap, feat = line.strip().split('\t')
                feat = np.fromstring(feat, sep = ',')
            except ValueError:
                word = line.strip()
                hint = ""
                gap = ""
                feat = ""
            data[(tok_idx, sent_idx)] = {"word":word, 
                      "hint":hint,
                      "gap":gap,
                      "features":feat
                      }
            tok_idx += 1
    return data

# Silver data loader -> not tagged
# Load instances only.
def load_silver_data_instances(infile):
    data = {}
    with open(infile,'r') as f:
        idx = 0
        for line in f:
            try:
                word, hint, gap, features = line.strip().split('\t')
                data[idx] = {"word":word, 
                        "hint":hint,
                        "gap":gap,
                        "features":np.fromstring(features, sep = ',')
                        }
                idx += 1
            except ValueError:
                continue
    return data


