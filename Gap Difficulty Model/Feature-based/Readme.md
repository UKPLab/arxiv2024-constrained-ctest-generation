# Feature-based Models

This folder provides code for training and testing the feature based models used in our work. We investigate following models:

* XGBoost (XGB)
* Multi-layer Perceptrons (MLP)
* Linear Regression (LR)
* Support Vector Machines (SVM)

We tune 100 randomly generated configurations for the MLP found in `configs` generated with `python create_configs.py`. We further the activation (`relu` or `linear`) and tune `c` for our SVM (in the `.sh` files). XGB and LR are not further tuned. Unfortunately, we are not allowed to further share our training and development data. Please contact [Lee et al. (2020)](https://aclanthology.org/2020.acl-main.390/) for access to the data.