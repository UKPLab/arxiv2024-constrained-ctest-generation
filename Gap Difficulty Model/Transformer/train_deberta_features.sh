#!/bin/bash

for seed in 28695 37432 67581 69550 73485 84331 86691 86862 87483 106530     
do

    # use features, deberta base
    python main_bert_tokenregression_features.py --train-folder data/train --test-folder data/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-base --model-path models --loss mse --features True

    # use features, deberta large
    python main_bert_tokenregression_features.py --train-folder data/train --test-folder data/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-large --model-path models --loss mse --features True

    # no features, deberta base
    python main_bert_tokenregression_features.py --train-folder data/train --test-folder data/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-base --model-path models --loss mse 
    
    # no features, deberta large
    python main_bert_tokenregression_features.py --train-folder data/train --test-folder data/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-large --model-path models --loss mse 

done
