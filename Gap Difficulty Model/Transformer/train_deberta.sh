#!/bin/bash
for seed in 28695 37432 67581 69550 73485 84331 86691 86862 87483 106530     
do

    # Base model
    python main_bert_tokenregression.py --train-folder data/train --test-folder data/merged/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-base --model-path models --loss mse --max-len 512

    # Large model
    python main_bert_tokenregression.py --train-folder data/train --test-folder data/merged/test --seed $seed --epochs 250 --batch-size 5 --result-path results/ --model microsoft/deberta-v3-large --model-path models --loss mse --max-len 512

done
