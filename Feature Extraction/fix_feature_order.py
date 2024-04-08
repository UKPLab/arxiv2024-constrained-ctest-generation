# Script to add gap size as a feature 
# Also shuffles the features in a way that the last n position contain variables. 
import sys
import os
import numpy as np

size_features = { 
                 "compound_break":16,
                 "referental_gap":23,
                 "syllable_break":24,
                 "sol_char_len":31,
                 "sol_syla_len":32,
                } 
position_features = { 
                 "preceding_gaps":42,
                 "prec_gaps_cover_sentence":43,
                 "position_gap":48,
                } 
                
bert_features = { 
                 "bert_word_prob":59,
                 "bert_entropy":60,
                } 
                
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
                word, _, hint, gap, feat = line.strip().split('\t')
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

def fix_feature(single_instance: np.array, gap: str) -> np.array:
    # Get a single feature list and sort/ create a new one
    # Fetch indices
    size_idx = list(size_features.values())
    pos_idx = list(position_features.values())
    bert_idx = list(bert_features.values())
    # Fetch features
    size = np.array([single_instance[x] for x in size_idx])
    pos = np.array([single_instance[x] for x in pos_idx])
    bert = np.array([single_instance[x] for x in bert_idx])
    # del feats:
    new_features = np.delete(single_instance, size_idx + pos_idx + bert_idx)
    size_fixed = size
    size_fixed[3] = float(len(gap))
    # return new features
    modified_feat = new_features.tolist() + bert.tolist() + pos.tolist() + size_fixed.tolist()
    return modified_feat

infolder = sys.argv[1]
outfolder = sys.argv[2]
infiles = os.listdir(infolder)

tab_char = ord("\t")

for infile in infiles:
    data = load_silver_data(f"{infolder}/{infile}")
    new_data = {}
    for idx in range(len(data)):
        new_data[idx] = { "features":[] }
        for g, f in zip(data[idx]["gap"], data[idx]["features"]):
            feat = f
            if len(f) > 1:
                nf = fix_feature(f,g)
                feat = ','.join([str(x) for x in nf])
            new_data[idx]["features"].append(feat)
    with open(f"{outfolder}/{infile}",'w') as outlog:
        for idx in range(len(new_data)):
            for w, h, g, f in zip(data[idx]["word"],data[idx]["hint"], data[idx]["gap"], new_data[idx]["features"]):
                outlog.write(f"{w}{chr(tab_char)}{h}{chr(tab_char)}{g}{chr(tab_char)}{f}{chr(tab_char)}{os.linesep}")
            outlog.write("----\n")
    
    
    
    
    
