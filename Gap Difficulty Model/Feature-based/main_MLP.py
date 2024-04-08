import os
import math
import argparse
import logging
import random
from tqdm import tqdm
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import AdamW
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from neuralnets import CTestConfigMLR, CTestConfigMLR_Linear, CTestConfigMLR_ReLU
from readers import load_data_regression, MLPDataset

#######################
#   Main code
#######################

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    # Add early stopping
    parser = argparse.ArgumentParser(description='Train simple fast model')
    parser.add_argument('--train', default='data/train_imp.txt', help='Training data.')
    parser.add_argument('--test', default='data/test_imp.txt', help='Test data.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--early', type=int, default=20, help='random seed')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('--result', default="results_tuning/", help='results folder')
    parser.add_argument('--model', default="models/", help='models folder')
    parser.add_argument('--configs', default="configs/", help='configuration folder')
    parser.add_argument('--linear', type=bool, default=False, help="use linear output layer")
    parser.add_argument('--relu', type=bool, default=False, help="use relu output layer")
    args = parser.parse_args()

    # Some default parameters
    learning_rate = 5e-5
    batch_size = args.batch_size
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu') # Use CPU for training MLP

    ########################
    #     Preprocessing
    ########################
    
    logging.info(">>>>>>> Loading data...")   
    # Load Data
    train_dev = load_data_regression(args.train)
    test = load_data_regression(args.test)
    # Create datasets
    train_dev_data = MLPDataset(train_dev)
    test_data = MLPDataset(test)

    # Create loaders

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    logging.info(">>>>>>> Data loaded!")   
    
    #######################
    #    Model TUNING
    #######################
    for config in os.listdir(args.configs):
        config_path = os.path.join(args.configs,config)
        
        # Set paths
        if args.linear:
            model_name = "{}_MLP-linear_{}.pt".format(config, args.seed)
            results_name = "{}_MLP-linear_{}.txt".format(config, args.seed)
        elif args.relu:
            model_name = "{}_MLP-relu_{}.pt".format(config, args.seed)
            results_name = "{}_MLP-relu_{}.txt".format(config, args.seed)
        else:
            model_name = "{}_MLP-sigmoid_{}.pt".format(config, args.seed)
            results_name = "{}_MLP-sigmoid_{}.txt".format(config, args.seed)
            
        model_path = os.path.join(args.model, model_name) 
        results_file = os.path.join(args.result, results_name) 
        
        with open(config_path,'r') as f:
            config_dict = json.load(f)
        
        logging.info("Configuration {}".format(config_dict))
        
        # Init Model
        if args.linear:
            model = CTestConfigMLR_Linear(config_dict)
            logging.info(">>>>>>>>>>>>>> Using linear output node!")
        elif args.relu:
            model = CTestConfigMLR_ReLU(config_dict)
            logging.info(">>>>>>>>>>>>>> Using ReLU output node!")
        else:
            model = CTestConfigMLR(config_dict)
            
        model.to(device)
        model.train()

        optimizer = AdamW(model.parameters(), lr=learning_rate) 
        loss_fn = nn.MSELoss() 
        
        early_stopping_counter = 0
        # 80-20 split for train-dev
        train_size = int(0.8*len(train_dev_data))
        dev_size = len(train_dev_data) - train_size 
        
        best_rmse = 1000.0
        dev_loss = 100000.0
        best_pearson = 0.0
        best_epoch = 0

        for epoch in range(args.epochs):              
            logging.info(">>>>>>>>>>>>>> Epoch {} - Training Model".format(epoch))
            
            # Train-dev (80-20) splits at each epochs
            train_data, dev_data = random_split(train_dev_data,[train_size, dev_size])
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)
            for train_batch in train_loader:
                model.train()
                optimizer.zero_grad()
                train_features = train_batch['features'].to(device)
                train_labels = train_batch['labels'].to(device)           
                outputs = model(train_features)
                
                loss = loss_fn(outputs.reshape(-1), train_labels) 
                
                loss.backward()
                optimizer.step()    

            # Evaluate after epoch
            model.eval()
            dev_y_true = []
            dev_y_pred = []

            with torch.no_grad():
                running_dev_loss = 0.0 
                for dev_batch in dev_loader:
                    dev_features = dev_batch['features'].to(device)
                    dev_labels = dev_batch['labels'].to(device)
                    outputs = model(dev_features)
                    
                    # Check loss for earhly stopping:
                    running_dev_loss += loss_fn(outputs.reshape(-1), dev_labels)

                    dev_y_pred += [x.cpu().detach().item() for x in outputs]
                    dev_y_true += dev_labels.tolist()
                    
                if dev_loss < running_dev_loss:
                    early_stopping_counter += 1
                else:
                    devl_loss = running_dev_loss
                    early_stopping_counter = 0

            dev_rmse = math.sqrt(mean_squared_error(dev_y_true, dev_y_pred))
            dev_pearson, _ = pearsonr(dev_y_true, dev_y_pred)
            logging.info(">>>>>>> Epoch {} - RMSE dev: {} - Pearson dev: {}".format(epoch, dev_rmse, dev_pearson))   
            # Store best model only
            if dev_rmse <= best_rmse:
                best_epoch = epoch
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()}, 
                            model_path)
                best_rmse = dev_rmse
                best_pearson = dev_pearson
                
            if early_stopping_counter == args.early:
                logging.info(">>>>>>> Epoch {} - No improvement after {} epochs.".format(epoch, args.early))
                                    
        logging.info(">>>>>>> Testing...")
        # Final Prediction
        best_model = torch.load(model_path)
        # load best model 
        model.load_state_dict(best_model['model_state_dict'])
        model.eval()
        test_idx = []
        test_y_true = []
        test_y_pred = []

        with torch.no_grad():
            for test_batch in test_loader:
                test_features = test_batch['features'].to(device)
                test_labels = test_batch['labels'].to(device)
                outputs = model(test_features)

                test_idx += test_batch['id']
                test_y_pred += [x.cpu().detach().item() for x in outputs]
                test_y_true += [x.cpu().detach() for x in test_labels]
                
        test_rmse = math.sqrt(mean_squared_error(test_y_true, test_y_pred))
        test_pearson,_ = pearsonr(test_y_true, test_y_pred)
        logging.info(">>>>>>> Final RMSE: {} - Pearson: {}".format(test_rmse, test_pearson))   
                        
        # Store results 
        with open(results_file,'w') as outlog:
            outlog.write("DEV Epoch {}: RMSE score: {} - Pearson score: {}\n".format(best_epoch, best_rmse, best_pearson))
            outlog.write("TESTING: RMSE score: {} - Pearson score: {}\n".format(test_rmse, test_pearson))
            for idx, t, p in zip(test_idx, test_y_true, test_y_pred):
                outlog.write("{}\t{}\t{}\n".format(idx, t, p))
            
        logging.info(">>>>>>> Done!")   

if __name__ == '__main__':
    main()
