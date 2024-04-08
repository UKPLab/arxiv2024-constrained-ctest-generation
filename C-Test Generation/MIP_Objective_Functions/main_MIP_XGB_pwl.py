""" Code for generating a MIP gurobipy approximation for the XGB model.
"""
import os
import sys
import json
import math
import argparse
import logging 
import random
import time

import numpy as np
import xgboost

import gurobipy as gp
from gurobi_ml.xgboost import add_xgboost_regressor_constr

import utils as uu
import utils.data_loader as dl
from ctest import PositionManipulator, GapManipulator

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Train simple fast model')
    parser.add_argument('--model', default='models/XGB_42_model.json', help='Model to convert.')
    parser.add_argument('--output-name', default="XGB", help='Output folder of the gurobi predictions')
    parser.add_argument('--output-folder', default="results/", help='Output folder of the gurobi predictions')
    parser.add_argument('--input-file', default='example_data/GUM_academic_art_7.txt', help='Path input file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dictionary', default='resources/american-english.txt', help='Path to dictionary')
    parser.add_argument('--update-bert', type=bool, default=False, help='Update bert features or not. NOTE: this adds substantial time!')
    parser.add_argument('--tau', type=float, default=0.5, help='Target difficulty')
    parser.add_argument('--num-gaps', type=int, default=20, help='Number of gaps sampled')
    args = parser.parse_args()
    
    ###############
    #  Parameters
    ############### 
    infile_name = args.input_file.split('/')[-1]
    tau = args.tau
    
    logging.info(f"====> Processing {infile_name}...")
    
    # Skip if already exists:
    if os.path.exists(os.path.join(args.output_folder,f"{args.output_name}_{args.model.split('/')[-1]}_{infile_name}_{tau}")):
        logging.info(f"====> {infile_name} already done. Skipping.")
        sys.exit(0)

    pos_feature_index_dict = {
                           "gaps_in_sent": 51,
                           "preceding_gaps": 52,
                           "preceding_gaps_sent": 53,
                           "occurs_as_gap": 54,
                          }
                       
    pos_index_feature_dict = {v:k for k,v in pos_feature_index_dict.items()}
    
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
    base_file = dl.load_silver_data_instances_sents(args.input_file)

    #####################
    #  Select gap scheme
    #####################
    pos_manip = PositionManipulator()
    gap_manip = GapManipulator(args.dictionary, args.update_bert)

    logging.info(f"====> Building position optimization model for difficulty {tau}.")
    
    opt_model = gp.Model()
    
    ########################
    # Build & run optimizer
    ########################
    # Create dataset consisting only of the features
    gap_data = pos_manip.filter_empty(base_file)
    # Select subset of gaps
    gap_data_small = {k:v for k,v in gap_data.items() if k in sorted(list(gap_data.keys()))}
    
    try:
        assert (len(gap_data_small) == 40)
    except AssertionError:
        logging.info(f"==============> ERROR! Required 40 instances but received {len(gap_data_small)}. ")
        logging.info(f"==============> Skipping {infile_name} ")
        sys.exit(0)
    
    input_var = opt_model.addVars(len(gap_data_small),61, name="input_var") 
    
    gap_key_idx_mapping = {key:i for i, key in enumerate(sorted(gap_data_small.keys()))}
    gap_idx_key_mapping = {i:k for k, i in gap_key_idx_mapping.items()}
    gap_idx_sent_mapping = {i:k[1] for i, k in gap_idx_key_mapping.items()} # Index consists of (token_idx, sent_idx)
    gap_sent_idx_mapping = {s:[] for _, s in gap_idx_sent_mapping.items()}
    # Fetch matrix consisting of gaps
    occurs_as_gaps = pos_manip.get_occurs_as_gap_matrix(gap_data_small)
    
    for k,v in gap_idx_sent_mapping.items():
        gap_sent_idx_mapping[v].append(k)
        
    gap_size_lookup = {}
    gap_decision_variables = {} 
    ########################
    #    Model Gap Size
    ########################
    for gap_idx, gap_instance in gap_data_small.items():
        word = gap_instance["word"]    
        # We first fetch features etc.
        gap_fixed_feats, _, _ = uu.split_position_gap_features(gap_instance["features"], args.update_bert)
        feature_space = gap_manip.get_feature_space_idx_gap_size(word)
        gap_upper = len(word) 
        gap_lower = 1

        # Set fixed features via constraints:
        for i, val in enumerate(gap_fixed_feats):
            input_var[(gap_key_idx_mapping[gap_idx],i)].lb = val
            input_var[(gap_key_idx_mapping[gap_idx],i)].ub = val

        gap_sizes = [i for i in range(gap_lower, gap_upper)]
        
        var_bounds_gap = {i:{"ub":max([feature_space[j][i] for j in feature_space.keys()]),
                         "lb":min([feature_space[j][i] for j in feature_space.keys()])} 
                        for i in gap_index_feature_dict.keys()
                     }
        
        # Set bounds for variable features:
        for feat_key, vals in var_bounds_gap.items():
            input_var[(gap_key_idx_mapping[gap_idx],feat_key)].lb = vals["lb"]
            input_var[(gap_key_idx_mapping[gap_idx],feat_key)].ub = vals["ub"]
        
        # Add constraints for the selection
        # Make tuple dict
        gap_size_lookup[gap_idx] = {(gi,fi):feature_space[gi][fi]  for gi in gap_sizes for fi in gap_index_feature_dict.keys()}
        gap_decision_variables[gap_idx] = opt_model.addVars(gap_sizes, vtype=gp.GRB.BINARY)

        # Select **exactly** one
        opt_model.addConstr(gp.quicksum(gap_decision_variables[gap_idx]) == 1)

        # gap index equals the sum of idx * decision variables       
        opt_model.addConstrs(input_var[(gap_key_idx_mapping[gap_idx],fi)] == gp.quicksum(gap_size_lookup[gap_idx][(gi,fi)]*gap_decision_variables[gap_idx][gi] for gi in gap_sizes) for fi in gap_index_feature_dict.keys())
            
    ########################
    #    Model Position
    ########################

    var_bounds_pos = pos_manip.get_bounds(gap_data_small)

    # Set bounds of variable position features
    for key, val in var_bounds_pos.items():
        for i, v in val.items():
            input_var[(gap_key_idx_mapping[key],i)].ub = v
            input_var[(gap_key_idx_mapping[key],i)].lb = 0.0
            
    # Bound occurs as gap features (set to 0 if the word only occurs once)
    occurs_as_gaps_lookup = {}
    for k,v in occurs_as_gaps.items():
        if len(v) > 0:
            occurs_as_gaps_lookup[gap_key_idx_mapping[k]] = [gap_key_idx_mapping[x] for x in v]
        else:
            input_var[(gap_key_idx_mapping[key],pos_feature_index_dict['occurs_as_gap'])].ub = 0.0
            input_var[(gap_key_idx_mapping[key],pos_feature_index_dict['occurs_as_gap'])].lb = 0.0
                
    # We model the selection problem as the sum of binary variables times the lookup.
    position_decision_variables = opt_model.addVars(len(gap_data_small), vtype=gp.GRB.BINARY, name="position_decision_variables")
    # Select **exactly** 20 gaps
    opt_model.addConstr(gp.quicksum(position_decision_variables) == args.num_gaps)
    
    # Gaps in sent feature constraint:
    # For each instance: instance_idx
    # We look up the sentence: gap_idx_sent_mapping[instance_idx]
    # and select all variables in the sentence: position_decision_variables[sent]
    # Finally, we sum over them via quicksum.
    opt_model.addConstrs(input_var[(instance_idx,pos_feature_index_dict["gaps_in_sent"])] == gp.quicksum(position_decision_variables[sent] for sent in gap_sent_idx_mapping[gap_idx_sent_mapping[instance_idx]]) for instance_idx in range(len(gap_data_small))) 
    
    # Preceding gaps feature constraint:
    # For each instance: instance_idx
    # Sum over all binary variables i that are smaller then the idx (we can just use range)
    opt_model.addConstrs(input_var[(instance_idx,pos_feature_index_dict["preceding_gaps"])] == gp.quicksum(position_decision_variables[i] for i in range(instance_idx)) for instance_idx in range(len(gap_data_small))) 
    
    # Preceding gaps in sent feature constraint:
    # For each instance: instance_idx
    # Sum over all binary variables i that are smaller then the idx (we can just use range)
    # Difference: we use the smallest start idx from sents
    opt_model.addConstrs(input_var[(instance_idx,pos_feature_index_dict["preceding_gaps_sent"])] == gp.quicksum(position_decision_variables[i] for i in range(min(gap_sent_idx_mapping[gap_idx_sent_mapping[instance_idx]]), instance_idx)) for instance_idx in range(len(gap_data_small))) 
    
    opt_model.update()
    
    # Occurs as gap:
    # Build similar idx vectors 
    # Add generatormaxconstraints for the occurs as gap feature:
    # Only if we have any same possible gaps in the data:
    if len(occurs_as_gaps_lookup) > 0:
        occurs_as_gap_features = opt_model.addVars(len(occurs_as_gaps_lookup), vtype=gp.GRB.BINARY, name="occurs_as_gap")
        occurs_as_gap_var = opt_model.addVars(len(occurs_as_gaps_lookup), vtype=gp.GRB.BINARY)
        for i, v in enumerate(occurs_as_gaps_lookup.values()):
            opt_model.addGenConstrMax(occurs_as_gap_var[i], [position_decision_variables[x] for x in v], 0.0)
        
        opt_model.update()
        for i, k in enumerate(occurs_as_gaps_lookup.keys()):
            opt_model.addConstr(input_var[(k,pos_feature_index_dict['occurs_as_gap'])] == occurs_as_gap_var[i])

    output_vars = opt_model.addVars(len(gap_data_small), lb=-1000, ub=1000, vtype=gp.GRB.CONTINUOUS)        
    # Add neural networks
    for i in range(len(gap_data_small)):
        add_xgboost_regressor_constr(opt_model, xgb_model, input_var.select(i,'*'), output_vars[i])

    sub_capped_outputs = opt_model.addVars(len(gap_data_small))
    # Bound ouput by [0.0, 1.0]
    for i in range(len(gap_data_small)):
        opt_model.addGenConstrMax(sub_capped_outputs[i], [output_vars[i]], 0.0)
        
    capped_outputs = opt_model.addVars(len(gap_data_small))
    for i in range(len(gap_data_small)):
        opt_model.addGenConstrMin(capped_outputs[i], [sub_capped_outputs[i]], 1.0)
        
    # Final difficulty is the sum over all gaps divided by the number of gaps (20)
    total_gap_sum = opt_model.addVar()
    opt_model.addConstr(total_gap_sum == gp.quicksum(capped_outputs)/args.num_gaps)
    
    ################################
    # Defining the final objective
    # Use picewise linear approximations
    ################################
    # Our output variable
    objective_var = opt_model.addVar(lb=0.0, ub=1000, vtype=gp.GRB.CONTINUOUS)
    # Our piecewise linear points
    x_axis = [0, tau, 10]
    y_axis = [tau, 0, 1-tau]
    
    opt_model.addGenConstrPWL(total_gap_sum, objective_var, x_axis, y_axis, name="PWLConstr")
    opt_model.setObjective(objective_var, gp.GRB.MINIMIZE)
        
    logging.info("====> Running optimization for positions (with variable gap size)")
    
    # Only measure solving time!
    start_time = time.perf_counter()
    
    opt_model.optimize()
    
    # End measuring time:
    end_time = time.perf_counter()
    
    results = {"gaps":[],
               "no-gaps":[],
               "time":end_time-start_time}

    for var in position_decision_variables:
        tok_iii, sent_iii = gap_idx_key_mapping[var]
        if position_decision_variables[var].X >= 0.9:
            for gap_var in gap_decision_variables[(tok_iii, sent_iii)]:
                if gap_decision_variables[(tok_iii, sent_iii)][gap_var].X >= 0.9:
                    results["gaps"].append(((tok_iii, sent_iii), base_file[(tok_iii, sent_iii)]['word'], gap_var))
        else:
            results["no-gaps"].append(((tok_iii, sent_iii), base_file[(tok_iii, sent_iii)]['word']))
            
    with open(os.path.join(args.output_folder,f"{args.output_name}-pwl_{args.model.split('/')[-1]}_{infile_name}_{tau}"),'w') as f:
        json.dump(results, f)       
        
    logging.info("======> Done.")
                           


if __name__ == '__main__':
    main()
