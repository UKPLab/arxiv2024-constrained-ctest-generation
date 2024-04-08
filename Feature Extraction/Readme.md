# Feature Extraction

[Feature Extraction](#features) contains our feature extraction pipeline, derived from [Beinborn (2016)](https://tuprints.ulb.tu-darmstadt.de/5647/) and [Lee et al. (2020)](https://aclanthology.org/2020.acl-main.390/) extracts 61 features for a given input text in several successive steps. The feature extraction pipeline proposed by Beinborn (2016) requires a working DKPro-Core environment that include some licensed resources. [Lee et al. (2019)](https://github.com/UKPLab/acl2019-ctest-difficulty-manipulation/tree/master/code#setting-up-the-resources) provide explicit instructions on setting up the DKPro-Core environment. 

For easier processing, we compiled the whole pipeline of Beinborn (2016) into two executable `.jar` files; one for sentence scoring (`sentence_scoring.jar`) and one for feature extraction (`feature_extraction.jar`). The `.jar` files will be provided upon acceptance as they are too large (265 MB in total) for the submission upload. The required file format for the pipelines are `tc files` used by [DKPro TC](https://dkpro.github.io/dkpro-tc/). This format is comparable to the [CoNLL-U Format](https://universaldependencies.org/format.html) with a single token per line and additional annotations added via tab separators. In addition, sentence endings are explicitly marked via `----` . 

Our whole extraction pipeline consists of five parts:

1. Sentence Scoring
2. Feature Extraction (Beinborn, 2016)
3. Feature Extraction (Lee, 2020)
4. Feature Imputing
5. Aggregating and Re-indexing

We first generate an appropriate virtual environment using conda:

    conda create --name=feature-extraction python=3.10
    conda activate feature-extraction
    pip install -r requirements.txt

Next, set the DKPro environment via:

    export "DKPRO_HOME=/home/jilee/Desktop/ARR/GrKMathe/grk-feature-extraction/resources/DKPro"

In addition, we need to explicitly set the path to the treetagger library:

    export "TREETAGGER_HOME=/home/jilee/Desktop/ARR/GrKMathe/grk-feature-extraction/resources/DKPro/treetagger/lib"

Finally, you can run the full feature extraction pipeline via:

    run_feature_extraction.sh <input-folder> <tmp-folder> <output-folder>

Note, that the tmp-folder is only used for storing the intermediate outputs and will be deleted afterwards. The C-Tests in the tc format are generated using the default strategy. An appropriate generator is provided by [Lee et al. (2019)](https://github.com/UKPLab/acl2019-ctest-difficulty-manipulation/tree/master/code#setting-up-the-resources).