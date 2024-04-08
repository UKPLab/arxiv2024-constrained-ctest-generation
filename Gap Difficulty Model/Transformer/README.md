# Transformer
Implementation for using Bert like models in a regression setup with masks. Changes compared to a default MLM classification setup include: 

* Added ignore_index (int -100) for MSE (`mse`) and L1 (`l1`) losses in pytorch.
* Extended data collator to return padded float values (required for regression)

To run the code, first create a virtual environment (e.g., conda) and install the required packages:

    conda create --name=mlm python=3.10
    pip install -r requirements.txt

There are two scripts provided, one for training (`main_bert_tokenregression.py`) and one for testing (`inference_only.py`). You can run them via:

    python main_bert_tokenregression.py --train-folder <training-data> --test-folder <test-data> --seed 42 --epochs 250 --batch-size 5 --result-path <results-folder> --model microsoft/deberta-v3-base --model-path models --loss mse --max-len 512
    python inference_only.py --test-folder <test-data> --seed 42 --batch-size 5 --result-path <results-folder> --model <model-type> --model-path <path-to-trained-model> --max-len 512

The `.sh` files contain the training scripts for base and large models across 10 different seeds.

*UPDATE*:
Two additional scripts have been added for training Bert-like models using the `[CLS]` token prediction and the hand crafted features used by Beinborn 2016 (`--features True`). For example use cases, please see the respective bash scripts `train_<model>_features.sh`. 