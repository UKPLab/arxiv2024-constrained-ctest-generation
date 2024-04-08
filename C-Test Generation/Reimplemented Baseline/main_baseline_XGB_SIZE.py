""" Reimplemented SIZE baseline from lee et al 2019:
    https://aclanthology.org/P19-1035/ 
"""
import os
import json
import math
import argparse
import logging 
import random
import pickle
from time import perf_counter

import numpy as np
import xgboost

import utils.data_loader as dl
from ctest import PositionManipulator, GapManipulator

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Train simple fast model')
    parser.add_argument('--model', default='models/XGB_42_model.json', help='Model to convert.')
    parser.add_argument('--output-folder', default="results/SIZE", help='Output folder of the gurobi predictions')
    parser.add_argument('--data', default='example_data', help='Path to training data folder')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dictionary', default='resources/american-english.txt', help='Path to dictionary')
    parser.add_argument('--update-bert', type=bool, default=False, help='Update bert features or not. NOTE: this adds substantial time!')
    parser.add_argument('--tau', type=float, default=0.5, help='Target difficulty (error-rate) for the resulting C-Test')
    args = parser.parse_args()
    
    ###############
    #  Parameters
    ############### 
    tau = args.tau

    gap_feature_index_dict = {
                       "compound_break": 56,
                       "th_gap": 57,
                       "syllable_break": 58,
                       "length_of_solution_chars": 59,
                       "length_of_solution_sylls": 60, 
    }
    
    gap_index_feature_dict = {v:k for k,v in gap_feature_index_dict.items()}
    
    ###############
    #  Load model
    ###############
    # Build model from 'scratch'
    logging.info("====> Loading model")

    xgb_model =  xgboost.Booster()
    xgb_model.load_model(args.model)
    
    ###############
    #  Load data
    ###############
    logging.info(f"====> Loading data from {args.data}")
    documents = {}
    for infile in os.listdir(args.data):
        # Load instances only
        documents[infile] = dl.load_silver_data_instances_sents(os.path.join(args.data,infile))

    #####################
    #  Select gap scheme
    #####################
    gap_manip = GapManipulator(args.dictionary, args.update_bert)
    pos_manip = PositionManipulator()

    # Load static C-Test (gaps at odd indices)
    odd_scheme = [i for i in range(1,40,2)]

    for doc_name, doc_instance in documents.items(): 
    
        # Odd features related to DEF c-test
        odd_var_feats = pos_manip.get_pos_features(doc_instance, odd_scheme)

        # Compute mean difficulty
        diffs = {k:xgb_model.predict(xgboost.DMatrix(v['features'].reshape(1, -1))) for k, v in odd_var_feats.items()}
        gap_changes = {k:[] for k in odd_var_feats.keys()}
        
        exchange_set = {}
        deltas = {}

        current_diff = np.mean(list(diffs.values()))

        if current_diff < tau:
            for k, v in odd_var_feats.items():
                _, _, new_gap = gap_manip.get_gap_idx(v['word'], len(v['gap'])+1)
                exchange_set[k] = xgb_model.predict(xgboost.DMatrix(np.array(list(new_gap.values())).reshape(1, -1)))
                deltas[k] = exchange_set[k]-diffs[k]
            while np.mean(list(diffs.values())) < tau:
                # identify gap and replace with the max estimated change
                replace_key = max(deltas, key=deltas.get)
                if len(diffs[replace_key]) != len(exchange_set[replace_key]):
                    # Only replace if there was a change
                    diffs[replace_key] = exchange_set[replace_key]
                    # Keep track of change
                    gap_changes[replace_key].append(1)
                    # Update the inc/dec set
                _, _, new_gap = gap_manip.get_gap_idx(odd_var_feats[replace_key]['word'], len(odd_var_feats[replace_key]['gap'])+1)
                exchange_set[replace_key] =  xgb_model.predict(xgboost.DMatrix(np.array(list(new_gap.values())).reshape(1, -1)))
                deltas[replace_key] = exchange_set[replace_key]-diffs[replace_key]
                new_diff = np.mean(list(diffs.values()))
                if new_diff == current_diff:
                    # If there is no change, stop.
                    break
                current_diff = new_diff
        else:
            for k, v in odd_var_feats.items():
                _, _, new_gap = gap_manip.get_gap_idx(v['word'], len(v['gap'])-1)
                exchange_set[k] = xgb_model.predict(xgboost.DMatrix(np.array(list(new_gap.values())).reshape(1, -1)))
                deltas[k] = diffs[k]-exchange_set[k]
            while np.mean(list(diffs.values())) > tau:
                # identify gap and replace with the max estimated change
                replace_key = max(deltas, key=deltas.get)
                if len(diffs[replace_key]) != len(exchange_set[replace_key]):
                    diffs[replace_key] = exchange_set[replace_key]
                    # Keep track of change
                    gap_changes[replace_key].append(-1)
                    # Update the inc/dec set
                _, _, new_gap = gap_manip.get_gap_idx(odd_var_feats[replace_key]['word'], len(odd_var_feats[replace_key]['gap'])-1)
                exchange_set[replace_key] =  xgb_model.predict(xgboost.DMatrix(np.array(list(new_gap.values())).reshape(1, -1)))
                deltas[replace_key] = diffs[replace_key]-exchange_set[replace_key]
                new_diff = np.mean(list(diffs.values()))
                if new_diff == current_diff:
                    break
                current_diff = new_diff
                

        results = {"gaps":[],
                   "no-gaps":[]}

        for k, v in doc_instance.items():
            if k in odd_var_feats.keys():
                results["gaps"].append((k, v['word'], len(v['gap']) + sum(gap_changes[k])))
            else:
                results["no-gaps"].append((k, v['word']))
                
        with open(os.path.join(args.output_folder,f"{doc_name}_{tau}"),'w') as f:
            json.dump(results, f)
        
    logging.info("======> Done.")
                           


if __name__ == '__main__':
    main()
