import os
import math
import argparse
import logging
import random
from tqdm import tqdm

import numpy as np
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from readers import load_data_regression

#######################
#   Main code
#######################

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Train simple fast model')
    parser.add_argument('--train', default='data/train_imp.txt', help='Training data.')
    parser.add_argument('--test', default='data/test_imp.txt', help='Test data.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--result', default="results_svr/", help='results folder')
    parser.add_argument('--model', default="models_svr/", help='models folder')
    parser.add_argument('--c', type=float, default=1, help='C parameter for the svm')
    args = parser.parse_args()

    # Some default parameters
    np.random.seed(args.seed)
    random.seed(args.seed)
    c = args.c

    ########################
    #     Preprocessing
    ########################

    logging.info(">>>>>>> Loading data...")   
    # Set paths
    model_name = "CTest_SVR_{}_{}.pt".format(args.seed,c)
    model_path = os.path.join(args.model, model_name) 
    results_name = "CTest_SVR_{}_{}.txt".format(args.seed,c)
    results_file = os.path.join(args.result, results_name) 
    
    # Load Data
    train = load_data_regression(args.train)
    test = load_data_regression(args.test)

    # Create datasets
    train_X = [np.array(v['feature']) for k,v in sorted(train.items())]
    train_y = [np.array(v['score']) for k,v in sorted(train.items())]

    test_X = [np.array(v['feature']) for k,v in sorted(test.items())]
    test_y = [np.array(v['score']) for k,v in sorted(test.items())]
    test_idx = [v['token'] for k,v in sorted(test.items())]

    logging.info(">>>>>>> Data loaded!")   

    #######################
    #    Model Training
    #######################

    model_svr = SVR(C=c, epsilon=0.2)
    model_svr.fit(train_X, train_y)
    
    # Use dev set to tune our hyperparameters
    pred_svr = model_svr.predict(test_X)
    true_svr = test_y

    test_score = mean_squared_error(true_svr,pred_svr)
    pearson_score = pearsonr(true_svr,pred_svr)      
                    
    # Store results 
    with open(results_file,'w') as outlog:
        outlog.write("RMSE score: {} - Pearson score: {}\n".format(math.sqrt(test_score), pearson_score))
        for idx, t, p in zip(test_idx, true_svr, pred_svr):
            outlog.write("{}\t{}\t{}\n".format(idx, t, p))
        
    # Save model:
    filename = '{}.pkl'.format(os.path.join(args.model, model_name))
    pickle.dump(model_svr,open(filename,'wb'))
 
    
    logging.info(">>>>>>> Done!")   

if __name__ == '__main__':
    main()
