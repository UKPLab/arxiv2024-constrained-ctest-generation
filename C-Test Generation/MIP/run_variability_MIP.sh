#!/bin/bash
FILES="example_data/*"
for tau in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for f in $FILES
    do
        python main_MIP_XGB.py --model models/XGB_42_model.json --output-folder results --output-name prefix --input-file $f --tau $tau --update-bert True # --debug-mip True --debug-log debugging_model_position.ilp 
    done
done
