# Reimplemented Baseline

We implement both baselines SEL and SIZE proposed by [Lee et al. (2019)](https://aclanthology.org/P19-1035/) using the trained XGB model. Runscrips for the variability experiments to assess the performance against the original implementations are provided in the shell scripts. The data to run these experiments is provided under `data/Variability Data`. To run the models, add a respective `data` folder containing the preprocessed data as well as a `model` folder with the trained model. Results will be output in a respective `results` folder.

The SIZE reimplementation requires [spacy](https://spacy.io/) to run with the following model:

    python -m spacy download en_core_web_sm

We further use the [pyphen](https://pyphen.org/) package for hyphenation and the standard ubuntu dictionary for [American English](https://manpages.ubuntu.com/manpages/trusty/man5/american-english.5.html) to check for compound breaks (found in the `resources` folder).