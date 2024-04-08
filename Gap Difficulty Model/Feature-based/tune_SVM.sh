#!/bin/bash
for c in 0.00001 0.0001 0.001 0.01 0.1 1 2 4 8 16 32 64 128 256 512 1024 2048 10000 100000 
do
    python main_SVM.py --seed 28695 --train data/train_imp.txt --test data/test_imp.txt --result results_svr/ --model models_svr/ --c $c
done

