import os
import sys

import numpy as np
from sklearn.impute import SimpleImputer


def load_tc_file(infile):
    file_ok = True
    tc_data = {'text':[],
               'features':{}}
    with open(infile,'r') as lines:
        for line in lines:
            if len(line.strip().split('\t')) < 2:
                tc_data['text'].append(line.strip())
            else:
                token, idx, hint, gap, feature_vec = line.strip().split('\t')
                feats = []
                for elem in feature_vec.split(','):
                    try:
                        feats.append(float(elem))
                    except ValueError:
                        feats.append(np.nan)
                try:
                    assert(len(feats) == 61)
                except AssertionError:
                    print("Error in file {}. Should have 61 features but got {}".format(infile,len(feats)))
                    file_ok = False
                    break
                tc_data['features'][int(idx)] = feats
                tc_data['text'].append("\t".join([token,str(idx),hint,gap]))
    return tc_data, file_ok

infolder = sys.argv[1]
outfolder = sys.argv[2]

infiles = os.listdir(infolder)

data = {}
idx = []
feats = []

for infile in sorted(infiles):
    subdat, file_ok = load_tc_file(os.path.join(infolder,infile))
    if file_ok:
        data[infile] = subdat
        for k,v in data[infile]['features'].items():
            idx.append("{}_{}".format(infile, k))
            feats.append(np.array(v))

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputed_feats = imputer.fit_transform(np.stack(feats))

for outfile, vals in data.items():
    outstrings = []
    try:
        for text in vals['text']:
            if len(text.strip().split('\t')) < 2:
                outstrings.append("{}\n".format(text.strip()))
            else:
                token, k, hint, gap = text.strip().split('\t')
                feature_idx = idx.index('{}_{}'.format(outfile,k))
                imp_feats = imputed_feats[feature_idx]
                feat_string = ','.join([str(x) for x in imp_feats])
                outstrings.append("{}\n".format('\t'.join([token, k, hint, gap, feat_string])))
    except ValueError:
        print("Error in file {}".format(outfile))
        continue
    with open(os.path.join(outfolder, outfile),'w') as outlog:
        for s in outstrings:
            outlog.write(s)
                
                
                
