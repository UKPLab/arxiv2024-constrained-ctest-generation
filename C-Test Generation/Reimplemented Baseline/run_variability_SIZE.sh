#!/bin/bash
for tau in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do 

    python main_baseline_XGB_SIZE.py --tau $tau --data example_data --output-folder results/SIZE 


done

