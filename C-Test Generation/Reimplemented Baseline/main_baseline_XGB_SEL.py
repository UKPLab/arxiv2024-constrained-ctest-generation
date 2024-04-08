""" Reimplemented SEL baseline from lee et al 2019:
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
    parser.add_argument('--output-folder', default="results/SEL", help='Output folder of the gurobi predictions')
    parser.add_argument('--data', default='example_data', help='Path to training data folder')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--tau', type=float, default=0.5, help='Target difficulty (error-rate) for the resulting C-Test')
    args = parser.parse_args()
    
    ###############
    #  Parameters
    ############### 
    tau = args.tau
    pos_manip = PositionManipulator()
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
    
    for doc_name, doc_instance in documents.items(): 
        # Filter instances only
        gap_data = pos_manip.filter_empty(doc_instance)
        
        # Strategy: create two sets of gaps: g1 > tau , g2 < tau
        # Select the gap that is closes to the target difficulty while alternating.
        # 1. predict all gaps
        # 2. build two sets (larger and smaller tau) -> easier and harder
        # 3. while n < 20, select gaps alternating from sets; starting with easier set.
        
        easier = []
        harder = []
        
        for k, v in gap_data.items():
            pred = xgb_model.predict(xgboost.DMatrix(v['features'].reshape(1, -1)))
            if pred-tau > 0:
                harder.append((abs(pred-tau),k))
            else:
                easier.append((abs(pred-tau),k))

                   
        easier.sort()
        harder.sort()
        
        gaps = []
                
        while len(gaps) < 20:
            if len(easier) > 0:
                sample = easier[0]
                gaps.append(sample[1])
                del easier[0]
            if len(harder) > 0:
                sample = harder[0]
                gaps.append(sample[1])
                del harder[0]

        no_gaps = sorted([x[1] for x in easier] + [x[1] for x in harder] )
        
        results = {"gaps":[],
                   "no-gaps":[]}

        for var in gaps:
            results["gaps"].append((var, doc_instance[var]['word'], len(doc_instance[var]['gap'])))
        for var in no_gaps:
            results["no-gaps"].append((var, doc_instance[var]['word']))
                
        with open(os.path.join(args.output_folder,f"{doc_name}_{tau}"),'w') as f:
            json.dump(results, f)
        
    logging.info("======> Done.")
                           


if __name__ == '__main__':
    main()
