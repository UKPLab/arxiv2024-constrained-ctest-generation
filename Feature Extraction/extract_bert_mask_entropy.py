import sys
import os

import numpy as np

from transformers import pipeline
from scipy.special import softmax
from scipy.stats import entropy

def read_tc_file(infile):
    sentences = []
    masked_sentences = []
    with open(infile,'r') as lines:
        sentence = []
        token = []
        for line in lines:
            if line.strip() == '----':
                for i, word in enumerate(sentence):
                     if type(word) != str:
                        masked = word[0][:len(word[2])] + "{}"
                        tmp_masked = token[:]
                        tmp_masked[i] = masked
                        txt = " ".join(tmp_masked)
                        target = " " + word[0][len(word[2]):]
                        sentences.append((txt,target))
                        tmp_masked_2 = token[:]
                        tmp_masked_2[i] = "{}"
                        masked_sentences.append((" ".join(tmp_masked_2)," " + word[0]))
                sentence = []
                token = []
                continue
            tok = line.strip().split()
            if len(tok) > 1:
                token.append(tok[0])
                sentence.append(tok)
            else:
                token.append(tok[0])
                sentence.append(tok[0])
    return sentences, masked_sentences

# Extract prediction probability of gaps using bert
# Extract entropy across top 50 predictions

nlp = pipeline("fill-mask", model="bert-base-cased")
nlp.top_k = 50

infolder = sys.argv[1]
outfolder = sys.argv[2]

for infile in os.listdir(infolder):
    with open(os.path.join(outfolder,infile),'w') as outlog:
        masked, msents = read_tc_file(os.path.join(infolder,infile))
        try:
            for gap, sent in zip(masked,msents):
                # Use only gap for entropy score
                result = nlp(gap[0].format(nlp.tokenizer.mask_token))
                top50 = [x['score'] for x in result]
                ent = entropy(softmax(np.array(top50)))
                # Use whole word for pred prob score
                res = nlp(sent[0].format(nlp.tokenizer.mask_token), targets=[sent[1]])
                outlog.write("{}\t{}\t{}\n".format(sent[1], res[0]['score'], ent))
        except (RuntimeError,ValueError) as e:
            outlog.write("Error in sentence during masking.")
            continue

