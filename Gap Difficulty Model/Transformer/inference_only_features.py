# Implements a bert model where the gaps are labeled with error-rates
# We follow the feature request from:
# https://github.com/huggingface/transformers/issues/3646

import os
import math
import argparse
import logging
import random

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizerFast 

from neuralnets.feature_model import FeatureBERT
from readers import FeatureDatasetPadding

#######################
#   helper functions
#######################

def load_data(infolder, mask_token):
    # Load tc_files with scores and gaps.
    tc_files = os.listdir(infolder)
    data = {'ids':[],
            'sents':[],
            'labels':[],
            'features':[],
            'masks':[]}
    nil_label = -100 # We use the labels as a flag to compute the mask tokens later on so that we can keep the whole token as is 
    nil_gap = "<no-gap>"
    for infile in tc_files:
        
        with open(os.path.join(infolder, infile),'r') as lines:
            sentence = []
            labels = []
            ids = []
            masked = []
            features = []
            for i, line in enumerate(lines):
                if line.strip() == '----':
                    # Add sentence data.
                    data['ids'].append(ids[:])
                    data['sents'].append(sentence[:])
                    data['labels'].append(labels[:])
                    data['features'].append(features[:])
                    data['masks'].append(masked[:])
                    sentence = []
                    labels = []
                    ids = []
                    masked = []
                    features = []
                    continue
                if len(line.strip().split('\t')) > 1:
                    token, gap_id, hint, _, score, feats = line.strip().split('\t')
                    sentence.append(token)
                    labels.append(float(score))
                    ids.append("{}_{}_{}".format(infile,i,gap_id))
                    masked.append(f"{hint}{mask_token}") # Generate masked token
                    feat = np.fromstring(feats, dtype=float, sep=',')
                    feat = np.delete(feat,49) # Delete BERT pred prob
                    feat = np.delete(feat,49) # Delete BERT entropy
                    features.append(feat)
                else:
                    sentence.append(line.strip())
                    labels.append(nil_label)
                    ids.append("{}_{}_-1".format(infile,i))
                    masked.append(nil_gap)
                    features.append(np.zeros(59)) # empty vector
                    
    return data


def generate_examples(examples, tokenizer):
    """ Create encoded sentences with a single mask (in the gap) 
        Aligns the encoded sentence, labels, and features accordingly.
    """
    # Check nils
    nil_label = -100 
    nil_gap = "<no-gap>"
    sents_raw = []
    labels = []
    features = []
    ids = []
    # Generate sentence with mask
    for i, masks in enumerate(examples['masks']):
        for j, mask in enumerate(masks):
            if mask == nil_gap:
                continue 
            try:
                assert (examples['labels'][i][j] != nil_label)
            except AssertionError:
                logging.info(">>>>> Error in {examples['ids'][i][j]}. Mismatch in labels.")
            tmp_sent = examples['sents'][i].copy()
            tmp_sent[j] = mask
            sents_raw.append(tmp_sent[:])
            labels.append(examples['labels'][i][j])
            features.append(examples['features'][i][j])
            ids.append(examples['ids'][i][j])

    tokenized_inputs = tokenizer(
        sents_raw, truncation=True, padding=True, is_split_into_words=True
    )

    return tokenized_inputs, labels, features, ids

#######################
#   Main code
#######################

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Train simple fast model for token regression')
    parser.add_argument('--test-folder', default='data/merged/test_fixed', help='Test data.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--result-path', default="results_test/", help='results folder')
    parser.add_argument('--model', default='prajjwal1/bert-tiny', help='huggingface model to use') # Only for testing. 
    parser.add_argument('--model-path', default='models/rmsebert-tiny_rs-42_loss-ldd1_ep-2_bs-5_useFeat-True.pt', help='model to use') # Only for testing.
    parser.add_argument('--features', type=bool, default=False, help='Use features?') # 
    # Use deberta for real prediction
    args = parser.parse_args()

    batch_size = args.batch_size
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ########################
    #     Preprocessing
    ########################

    logging.info(">>>>>>> Loading data...")   
    # Set paths
    model_name = args.model_path.split('/')[-1]
    model_path_pearson = os.path.join(args.model_path, "pearson"+model_name)
    model_path_rmse = os.path.join(args.model_path, "rmse"+model_name)
    results_file = os.path.join(args.result_path, model_name.replace('.pt','.txt'))
    
    # Init Tokenizer
    if 'roberta' in args.model:
        tokenizer =  RobertaTokenizerFast.from_pretrained(args.model, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model) 

    # Load Data
    test_data = load_data(args.test_folder, tokenizer.mask_token)
    test_encoded, test_labels, test_features, test_ids = generate_examples(test_data, tokenizer)
    test_data = FeatureDatasetPadding(test_ids, test_encoded, test_labels, test_features) # Ignore collators etc for testing.
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)  #

    logging.info(">>>>>>> Data loaded!")   

    #######################
    #    Model Training
    #######################
            
    # Init Model
    if args.features:
        model = FeatureBERT(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1) # Trigger regressions setup.

    model.eval()
    model.to(device)

    with torch.no_grad():
        test_y_idx = []
        test_y_true = []
        test_y_pred = []

        with torch.no_grad():
            for test_batch, tidx in test_loader:
                test_input_ids = test_batch['input_ids'].long().to(device)
                test_attention_mask = test_batch['attention_mask'].long().to(device)
                test_labels = test_batch['labels'].to(device)
                
                if args.features:
                    test_features = test_batch['features'].long().to(device)
                    outputs = model(test_input_ids, test_features, attention_mask=test_attention_mask)
                else:
                    outputs = model(test_input_ids, attention_mask=test_attention_mask)
                    
                if args.features:
                    test_y_pred += [x.cpu().detach().item() for x in outputs]
                else:
                    test_y_pred += [x.cpu().detach().item() for x in outputs.logits]
                test_y_true += test_labels.tolist()
                test_y_idx += tidx

    test_rmse = math.sqrt(mean_squared_error(test_y_true, test_y_pred))
    test_pearson,_ = pearsonr(test_y_true, test_y_pred)

    logging.info(">>>>>>> Test done! Test RMSE: {} \t Test Pearson: {}".format(test_rmse, test_pearson))   
    logging.info(">>>>>>> Storing results...")   

    with open(results_file,'w') as outlog:
        outlog.write("Pearson: {} ---- RMSE: {}\n".format(test_pearson, test_rmse))
        for idx, t, p in zip(test_y_idx, test_y_true, test_y_pred):
            outlog.write("{}\t{}\t{}\n".format(idx, t, p))

    logging.info(">>>>>>> Done!")   

if __name__ == '__main__':
    main()
