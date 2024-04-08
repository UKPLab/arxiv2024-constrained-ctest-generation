#!/bin/bash
for i in 28695 37432 67581 69550 73485 84331 86691 86862 87483 106530     
do
    python main_MLP.py --seed $i --epochs 250 --batch-size 10 --train data/train_imp.txt --test data/test_imp.txt --result results_tuning_regression_relu/ --model models_tuning_regression_relu/ --configs configs/ --early 250  --relu True
done
