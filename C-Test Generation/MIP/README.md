# MIP

This folder provides the implementation of our MIP generation strategy using the XGB model. We use [Gurobi](https://www.gurobi.com/) as the solver for our optmization problem. It is likely that the optimization model has more parameters than the defaul license of Gurobi allows. Gurobi offers [free licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) for academic and educational purposes. After registration, you will obtain a license file you can reference in your virtual environment for running the experiments:

    export GRB_LICENSE_FILE=<path-to-license>

Additionally, we require following [spacy](https://spacy.io/) model to run the experiments:

    python -m spacy download en_core_web_sm

We further use the [pyphen](https://pyphen.org/) package for hyphenation and the standard ubuntu dictionary for [American English](https://manpages.ubuntu.com/manpages/trusty/man5/american-english.5.html) to check for compound breaks (found in the `resources` folder).

The generation can be run via:

    python main_MIP_XGB.py --model models/XGB_42_model.json --output-folder results --input-file data/GUM_academic_art_4.txt --tau 0.1 --update-bert True

With `--tau` being the target difficulty (a value between [0, 1]) and `--update-bert` the indicator if the BERT-base features should be updated (adds overhead). A prefix to the resulting output can be added via the option `--output-name--output-name`. 

For debugging the MIP model, the option `--debug-mip` should be set to `True` and a path for the logging file provided via `--debug-log`.