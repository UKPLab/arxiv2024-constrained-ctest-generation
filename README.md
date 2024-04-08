# Constrained C-Test Generation via Mixed-Integer Programming

My Project does a lot of very interesting stuff, like this and that.

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

> **Abstract:** This work proposes a novel method to generate C-Tests; a deviated form of cloze tests (a gap filling exercise) where only the last part of a word is turned into a gap. In contrast to previous works that only consider varying the gap size or gap placement to achieve locally optimal solutions, we propose a mixed-integer programming (MIP) approach. This allows us to consider gap size and placement simultaneously, achieving globally optimal solutions and to directly integrate state-of-the-art models for gap difficulty prediction into the optimization problem. A user study with 40 participants across four C-Tests generation strategies (including GPT-4) shows that our approach (*MIP*) significantly outperforms two of the baseline strategies (based on gap placement and GPT-4); and performs on-par with the third (based on gap size). Our analysis shows that GPT-4 still struggles to fulfill explicit constraints during generation and that *MIP* produces C-Tests that correlate best with the perceived difficulty. We publish our code, model, and collected data consisting of 32 English C-Tests with 20 gaps each (3,200 in total) under an open source license.


Contact person: Ji-Ung Lee, ji-ung.lee@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/
https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Project structure

1. [Code](#code)
    1. [Feature Extraction](#features)
    2. [Gap Difficulty Model](#gapmodel)
        1. [Transformer](#mlm)
        2. [Feature-based](#feature)
    3. [C-Test Generation](#generation)
        1. [MIP](#mip)
        2. [Reimplemented Baseline](#reimplbaselines)
    4. [User Study](#study)
        1. [Study Interface](#interface)
        2. [Latin Hypercube Design](#sampling)
        3. [Analysis](#analysis)
2. [Data](#data)
    1. [User Study Data](#studydata)
    2. [GPT-4 Prompts and Responses](#gpt4)
    3. [Variability Data](#variability)
3. [Model](#model)


## Code <a name="code"></a>
The code is split into four major components. Before running any of the components, first setup an appropriate virtual enviroment (for instance, using conda) and install all packages:

    conda create --name=<envname> python=3.10
    conda activate <envname>
    pip install -r requirements.txt

We run all our experiments with python 3.10.

### Feature Extraction <a name="features"></a>

[Feature Extraction](#features) contains our feature extraction pipeline, derived from [Beinborn (2016)](https://tuprints.ulb.tu-darmstadt.de/5647/) and [Lee et al. (2020)](https://aclanthology.org/2020.acl-main.390/) extracts 61 features for a given input text in several successive steps. The feature extraction pipeline proposed by Beinborn (2016) requires a working DKPro-Core environment that include some licensed resources. [Lee et al. (2019)](https://github.com/UKPLab/acl2019-ctest-difficulty-manipulation/tree/master/code#setting-up-the-resources) provide explicit instructions on setting up the DKPro-Core environment. 

For easier processing, we compiled the whole pipeline of Beinborn (2016) into two executable `.jar` files; one for sentence scoring (`sentence_scoring.jar`) and one for feature extraction (`feature_extraction.jar`). The `.jar` files will be provided separately as they are too large (265 MB in total) for the submission upload. The required file format for the pipelines are `tc files` used by [DKPro TC](https://dkpro.github.io/dkpro-tc/). This format is comparable to the [CoNLL-U Format](https://universaldependencies.org/format.html) with a single token per line and additional annotations added via tab separators. In addition, sentence endings are explicitly marked via `----` . 

Our whole extraction pipeline consists of five parts:

1. Sentence Scoring
2. Feature Extraction (Beinborn, 2016)
3. Feature Extraction (Lee, 2020)
4. Feature Imputing
5. Aggregating and Re-indexing

First, set the DKPro environment via:

    export "DKPRO_HOME=/home/jilee/Desktop/ARR/GrKMathe/grk-feature-extraction/resources/DKPro"

In addition, we need to explicitly set the path to the treetagger library:

    export "TREETAGGER_HOME=/home/jilee/Desktop/ARR/GrKMathe/grk-feature-extraction/resources/DKPro/treetagger/lib"

Finally, you can run the full feature extraction pipeline via:

    run_feature_extraction.sh <input-folder> <tmp-folder> <output-folder>

Note, that the tmp-folder is only used for storing the intermediate outputs and will be deleted afterwards. The C-Tests in the tc format are generated using the default strategy. An appropriate generator is provided by [Lee et al. (2019)](https://github.com/UKPLab/acl2019-ctest-difficulty-manipulation/tree/master/code#setting-up-the-resources).

### Gap Difficulty Model <a name="gapmodel"></a>

For the gap difficulty prediction models, we provide the code in two separate projects for MLM models and Feature-based models.

#### Transformer <a name="mlm"></a>
Implementation for using Bert-like models in a regression setup with masks. Major changes compared to a default MLM classification setup are: 

* Added ignore_index (int -100) for MSE (`mse`) and L1 (`l1`) losses in pytorch.
* Extended data collator to return padded float values (required for regression)

There are two scripts provided, one for training (`main_bert_tokenregression.py`) and one for testing (`inference_only.py`). You can run them via:

    python main_bert_tokenregression.py --train-folder <training-data> --test-folder <test-data> --seed 42 --epochs 250 --batch-size 5 --result-path <results-folder> --model microsoft/deberta-v3-base --model-path models --loss mse --max-len 512
    python inference_only.py --test-folder <test-data> --seed 42 --batch-size 5 --result-path <results-folder> --model <model-type> --model-path <path-to-trained-model> --max-len 512

*UPDATE*:
Two additional scripts have been added for training Bert-like models using the `[CLS]` token prediction and the hand crafted features used by Beinborn 2016 (`--features True`). For example use cases, please see the respective bash scripts `train_<model>_features.sh`. 

#### Feature-based <a name="feature"></a>

This folder provides code for training and testing the feature based models used in our work. We investigate following models:

* XGBoost (XGB)
* Multi-layer Perceptrons (MLP)
* Linear Regression (LR)
* Support Vector Machines (SVM)

We tune 100 randomly generated configurations for the MLP found in `configs` generated with `python create_configs.py`. We further the activation (`relu` or `linear`) and tune `c` for our SVM (in the `.sh` files). XGB and LR are not further tuned. Unfortunately, we are not allowed to further share our training and development data. Please contact [Lee et al. (2020)](https://aclanthology.org/2020.acl-main.390/) for access to the data.

### C-Test Generation <a name="generation"></a>

We provide code for three C-Test generation strategies, namely, our MIP-based approach (MIP), and the reimplemented baselines for SEL and SIZE using the XGB model (Reimplemented Baseline). We provide preprocessed data for running the variability experiments under `data/Variability Data`. To run the models, add a respective `data` folder containing the preprocessed data as well as a `model` folder with the trained model. Results will be output in a respective `results` folder.

To process the gap size features (MIP and SIZE generation strategies), we require following [spacy](https://spacy.io/) model:

    python -m spacy download en_core_web_sm

We need the [pyphen](https://pyphen.org/) package for hyphenation and the standard ubuntu dictionary for [American English](https://manpages.ubuntu.com/manpages/trusty/man5/american-english.5.html) to check for compound breaks (found in the `resources` folder).

To run the SEL and SIZE baselines, please follow the instructions provided by [Lee et al. (2019)](https://github.com/UKPLab/acl2019-ctest-difficulty-manipulation).

#### MIP <a name="mip"></a>

This folder provides the implementation of our MIP generation strategy using the XGB model. We use [Gurobi](https://www.gurobi.com/) as the solver for our optmization problem. It is likely that the optimization model has more parameters than the defaul license of Gurobi allows. Gurobi offers [free licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) for academic and educational purposes. After registration, you will obtain a license file you can reference in your virtual environment for running the experiments:

    export GRB_LICENSE_FILE=<path-to-license>

The generation can be run via:

    python main_MIP_XGB.py --model models/XGB_42_model.json --output-folder results --input-file example_data/GUM_academic_art_4.txt --tau 0.1 --update-bert True

With `--tau` being the target difficulty (a value between [0, 1]) and `--update-bert` the indicator if the BERT-base features should be updated (adds overhead). A prefix to the resulting output can be added via the option `--output-name--output-name`. 

For debugging the MIP model, the option `--debug-mip` should be set to `True` and a path for the logging file provided via `--debug-log`.

#### Reimplemented Baseline <a name="reimplbaselines"></a>

We implement both baselines SEL and SIZE proposed by [Lee et al. (2019)](https://aclanthology.org/P19-1035/) using the trained XGB model. Runscripts for the variability experiments to assess the performance against the original implementations are provided in the shell scripts. The data to run these experiments is provided under `data/Variability Data`. To run the models, add a respective `data` folder containing the preprocessed data as well as a `model` folder with the trained model. Results will be output in a respective `results` folder.

### User Study <a name="study"></a>

Code regarding the user study consists of three parts. First, the interface we implement and use written in Flask using SQLAlchemy as a backend for our (MySQL) database. Second, the Latin Hypercube Sampling implemented as a constratined optimization problem, and third, the R scripts implementing the GAMM we use for our analysis.

#### Interface <a name="interface"></a>

This implements the interface used in our user study. For the reviewing, all links and names that may lead to deanonymization have been removed. The user study runs on a [Flask](https://flask.palletsprojects.com/en/3.0.x/) application with a database connected via [SQLAlchemy](https://www.sqlalchemy.org/).

After setting up the virtual environment, create a database (with an example user `admin` with the password `admin`):

    mysql -u admin -p
    CREATE DATABASE c-test;

Then import the database structure (including c-tests and selection)

    mysql -u admin -p c-test < c-test.sql

The application can be started via:

    cd c_test
    python __init__.py


Exporting data from the database can be done via:

    mysqldump -u admin -p c-test --add-drop-table > c-test.sql


#### Sampling <a name="sampling"></a>

This script generates user study configurations with maximal distance and equal distribution. The code can be run via:

    python fetch_minizinc_solutions.py

This generates some files in the `output` folder. First, a file containing all distances and possible combinations (`all_distances.dzn`) and second, one valid configuration (`final_combinations.json`). The code relies upon a constrainted optimization model found in the ```minizinc``` folder. For more information on [Minizinc](https://www.minizinc.org/); an open source constrained modeling language, please check their [documentation](https://www.minizinc.org/resources.html).

#### Analysis <a name="analysis"></a>

We provide simple preprocessing scripts for R as well as the respective R scripts for the significance analysis conducted in the paper. 

After setting up the virtual environment, install [R](https://www.r-project.org/) and [RStudio](https://posit.co/download/rstudio-desktop/) (RStudio is not required, but convenient). The R scripts use following packages:

* [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html): Mixed GAM Computation Vehicle with Automatic Smoothness Estimation
* [itsadug](https://cran.r-project.org/web/packages/itsadug/index.html): Interpreting Time Series and Autocorrelated Data Using GAMMs
* [report](https://cran.r-project.org/web/packages/report/index.html): Automated Reporting of Results and Statistical Models
* [xtable](https://cran.r-project.org/web/packages/xtable/index.html): Export Tables to LaTeX or HTML
* [hash](https://cran.r-project.org/web/packages/hash/index.html): Full Featured Implementation of Hash Tables/Associative Arrays/Dictionaries 

The raw data found in `study_data_raw` can be converted via `format_data_for_r.py` and `format_data_for_r_feedback.py`, which will be written into the `r_data` folder. The statistical models are provided in `r_models`.

## Data <a name="data"></a>

Our data is comprised of three parts. The data we collected in our study, the detailed GPT-4 prompts and responses, and the preprocessed instances for our preliminary experiments from the GUM corpus.
The data is available in tudatalib : https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4205

### User Study Data <a name="studydata"></a>

All detailed results are provided in the folder `csv`. This folder consists of five `.csv` files containing the C-Tests, the users (names obfucsated), and their questionnaire responses. `strategy.csv` denotes the order in which each C-Test has been seen and `ctest_user_mapping.csv` individual user responses for each C-Test.

In addition, we provide C-Tests generated with different strategies and their aggregated respective error-rates under `aggregated_ctests`.

### GPT-4 Prompts and Responses <a name="gpt4"></a>

This folder provides the full input prompts (`full_prompts`) and responses (`full_responses`) generated by GPT-4. If there was a regeneration necessary due to a lack of gaps, this is indicated in the first line of the response file, e.g., _2nd Try:_ . `c_tests` provides the format used in the user study consisting of the C-Test where _#GAP#_ indicates a gap, the solutions, the tokenized, and original texts.

### Variability Data <a name="variability"></a>

Contains 100 preprocessed C-Tests for the variability experiments from the [GUM](https://gucorpling.org/gum/) corpus. The text passages were randomly sampled according to the criteria mentioned in the paper.

## Model

The trained XGB model used for MIP and reimplemented baselines. You can use the model as follows:

    import xgboost
    xgb_model =  xgboost.Booster()
    xgb_model.load_model(path_to_model)
    xgb_model.predict(xgboost.DMatrix(features_vector.reshape(1, -1))) # Prediction for a single instance



## Requirements
**(change this as needed!)**

* Java x.x and higher
* Maven
* 64-bit Linux versions
* Windows x
* XX GB RAM

## Installation
**(change this as needed!)**

* Step 1

```
$nice_command

$some_script.sh
```

* Step 2

Do something and something

* ...
* Step n


## Running the experiments
**(change this as needed!)**

```
$cd bla/bla/bla
$some_cool_commands_here
```

### Expected results
**(change this as needed!)**

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description
**(change this as needed!)**

* `x, --xxxx`
  * This parameter does something nice
...
* `z, --zzzz`
  * This parameter does something even nicer
  
