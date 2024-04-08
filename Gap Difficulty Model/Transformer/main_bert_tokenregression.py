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
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import RobertaTokenizerFast 
from transformers import AdamW
from transformers import Trainer
from transformers import TrainingArguments

from readers import HuggingfaceDatasetPadding, HuggingfaceDataset
from collators import DataCollatorForTokenRegression
from losses import MSELossIgnore, L1LossIgnore

#######################
#   helper functions
#######################

def load_data(infolder):
    # Load tc_files with scores and gaps.
    tc_files = os.listdir(infolder)
    data = {'ids':[],
            'sents':[],
            'labels':[],
            'gaps':[]}
    nil_label = -100 # We use the labels as a flag to compute the mask tokens later on so that we can keep the whole token as is 
    nil_gap = "<no-gap>"
    for infile in tc_files:
        
        with open(os.path.join(infolder, infile),'r') as lines:
            sentence = []
            labels = []
            ids = []
            gaps = []
            for i, line in enumerate(lines):
                if line.strip() == '----':
                    # Add sentence data.
                    data['ids'].append(ids[:])
                    data['sents'].append(sentence[:])
                    data['labels'].append(labels[:])
                    sentence = []
                    labels = []
                    ids = []
                    gaps = []
                    continue
                if len(line.strip().split('\t')) > 1:
                    token, gap_id, _, gap, score, _ = line.strip().split('\t')
                    sentence.append(token)
                    labels.append(float(score))
                    ids.append("{}_{}_{}".format(infile,i,gap_id))
                    gaps.append(gap)
                else:
                    sentence.append(line.strip())
                    labels.append(nil_label)
                    ids.append("{}_{}_-1".format(infile,i))
                    gaps.append(nil_gap)
                    
    return data


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["sents"], truncation=True, padding=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def filter_nogaps(pred, true):
    new_pred, new_true = [],[]
    for p,t in zip(pred,true):
        if int(t) == -100:
            continue
        new_pred.append(p[0])
        new_true.append(t)

    return new_pred, new_true

def filter_nogaps_batches(pred, true):
    new_pred, new_true = [],[]
    for p,t in zip(pred,true):
        pp, tt = filter_nogaps(p,t)
        new_pred += pp
        new_true += tt
    return new_pred, new_true
    
def filter_nogaps_test(pred, true, idx):
    new_pred, new_true, new_idx = [],[],[]
    for p,t, i in zip(pred[0],true[0],idx[0]):
        if int(t) == -100:
            continue
        new_pred.append(p[0])
        new_true.append(t)
        new_idx.append(i)

    return new_pred, new_true, new_idx

#######################
#   Main code
#######################

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Train simple fast model for token regression')
    parser.add_argument('--train-folder', default='data/merged/train', help='Training data.')
    parser.add_argument('--test-folder', default='data/merged/test', help='Test data.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--result-path', default="results/", help='results folder')
    parser.add_argument('--model', default='./bert-tiny', help='huggingface model to use') # Only for testing. 
    parser.add_argument('--model-path', default='models/', help='huggingface model to use') # Only for testing.
    parser.add_argument('--loss', default='ldd1', help='used loss. l1 or mse') 
    parser.add_argument('--max-len', type=int, default=512, help='Maximum sequence length.') # 256 for deberta and 512 for roberta/bert 
    # Use deberta for real prediction
    args = parser.parse_args()

    # Some default parameters
    max_length = args.max_len
    #validation_split = 0.2
    learning_rate = 5e-5
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
    model_name = "{}_rs-{}_loss-{}_ep-{}_bs-{}.pt".format(args.model.split('/')[-1], args.seed, args.loss, args.epochs, args.batch_size)
    model_path = os.path.join(args.model_path, model_name)
    results_file = os.path.join(args.result_path, model_name.replace('.pt','.txt'))
    
    # Init Tokenizer
    if 'roberta' in args.model:
        tokenizer =  RobertaTokenizerFast.from_pretrained(args.model, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model) 

    # Load Data
    train_data = load_data(args.train_folder)
    test_data = load_data(args.test_folder)

    # Encode Data
    train_encoded = tokenize_and_align_labels(train_data, tokenizer)
    test_encoded = tokenize_and_align_labels(test_data, tokenizer)
   
    # Create datasets
    train_data = HuggingfaceDatasetPadding(train_data['ids'], train_encoded, train_data['labels'], max_length)
    test_data = HuggingfaceDatasetPadding(test_data['ids'], test_encoded, test_data['labels'], max_length) # Ignore collators etc for testing.

    data_collator = DataCollatorForTokenRegression(tokenizer=tokenizer)

    # Create loaders
    train_loader = DataLoader(train_data, collate_fn=data_collator,  batch_size=batch_size, shuffle=True) #
    test_loader = DataLoader(test_data, collate_fn=data_collator, batch_size=1, shuffle=False)  #

    logging.info(">>>>>>> Data loaded!")   

    #######################
    #    Model Training
    #######################
            
    # Init Model
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=1) # Trigger regressions setup.
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate) 

    results = []   
    best_rmse = (100.0,0)
    best_pearson = (0.0,0)
    
    if args.loss == 'l1':
        loss_fct = L1LossIgnore()
    else:
        loss_fct = MSELossIgnore()

    for epoch in range(args.epochs):
        logging.info(">>>>>>>>>>>>>> Epoch {}...".format(epoch))
        
        for train_batch, tidx in train_loader:
            model.train()
            optimizer.zero_grad()
            train_input_ids = train_batch['input_ids'].long().to(device)
            train_attention_mask = train_batch['attention_mask'].long().to(device)
            train_labels = train_batch['labels'].to(device)

            outputs = model(train_input_ids, attention_mask=train_attention_mask)
            
            #TODO: do a masking of the labels (cast them back to int) to ignore -100 labels in the loss
            loss = loss_fct(outputs, train_labels) 

            loss.backward()
            optimizer.step() 
            
            y_train_pred = [x.cpu().detach().numpy() for x in outputs.logits]
            y_train_true = train_labels.tolist()
            
            pred, true = filter_nogaps_batches(y_train_pred, y_train_true)
            
            assert(len(pred) == len(true))
            try:
                train_rmse = math.sqrt(mean_squared_error(pred, true))   
            except ValueError:
                # If we have a sentence with no gaps at all, ignore: 
                train_rmse = -1
            logging.info(">>>>>>>>>>>>>> Loss: {}\tRMSE: {}".format(loss,train_rmse))
            
        # Evaluate after whole batch

        with torch.no_grad():
            test_y_idx = []
            test_y_true = []
            test_y_pred = []

            with torch.no_grad():
                for test_batch, tidx in test_loader:
                    test_input_ids = test_batch['input_ids'].long().to(device)
                    test_attention_mask = test_batch['attention_mask'].long().to(device)
                    test_labels = test_batch['labels'].to(device)
                    outputs = model(test_input_ids, attention_mask=test_attention_mask)

                    test_y_p = [x.cpu().detach().numpy() for x in outputs.logits]
                    test_y_t = test_labels.tolist()
                    test_pred, test_true, test_idx = filter_nogaps_test(test_y_p, test_y_t, tidx)
                    test_y_true += test_true
                    test_y_pred += test_pred
                    test_y_idx += test_idx

        test_rmse = math.sqrt(mean_squared_error(test_y_true, test_y_pred))
        test_pearson,_ = pearsonr(test_y_true, test_y_pred)
        
        if test_rmse < best_rmse[0]:
            best_rmse = (test_rmse, epoch)
        if test_pearson > best_pearson[0]:
            best_pearson = (test_pearson,epoch)
            # Store best model according to pearson correlation
            torch.save(model.state_dict(), model_path)

        results.append((epoch, test_rmse, test_pearson))
        logging.info(">>>>>>>>>>>>>> Epoch {}. RMSE: {}, Pearson: {}".format(epoch, test_rmse, test_pearson))   

    logging.info(">>>>>>> Test done! Test RMSE: {} \t Test Pearson: {}".format(test_rmse, test_pearson))   
    logging.info(">>>>>>> Storing results...")   

    with open(results_file,'w') as outlog:
        outlog.write("Best Pearson in ep {}: {} ---- best RMSE in ep {}: {}\n".format(best_pearson[1],best_pearson[0],best_rmse[1],best_rmse[0]))
        for ep, tr, tp in results:
            outlog.write("Epoch: {}\tTest RMSE: {}\tTest Pearson: {}\n".format(ep, tr, tp)) 
        for idx, t, p in zip(test_y_idx, test_y_true, test_y_pred):
            outlog.write("{}\t{}\t{}\n".format(idx, t, p))

    logging.info(">>>>>>> Done!")   

if __name__ == '__main__':
    main()
