#!/bin/bash
input_data=$1 # Folder with TC files
tmp_folder=$2 # Folder to create the temporary features
output_folder=$3 # Output folder

mkdir $tmp_folder
mkdir $output_folder

echo "Setting up folders..."
mkdir $tmp_folder/input_a
mkdir $tmp_folder/input_b
mkdir $tmp_folder/sentence_scores_a
mkdir $tmp_folder/sentence_scores_b
mkdir $tmp_folder/features_a
mkdir $tmp_folder/features_b
mkdir $tmp_folder/bert_a
mkdir $tmp_folder/bert_b
mkdir $tmp_folder/merged_a
mkdir $tmp_folder/merged_b
mkdir $tmp_folder/imputed_a
mkdir $tmp_folder/imputed_b
mkdir $tmp_folder/cleaned_a
mkdir $tmp_folder/cleaned_b
echo "...done."

echo "Generate second half of the tc data..."
python generate_tc_2nd_half.py $input_data $tmp_folder/input_a $tmp_folder/input_b
echo "...done."

echo "Running sentence scoring..."
java -XX:MaxHeapSize=4G -jar sentence_scoring.jar $tmp_folder/input_a $tmp_folder/sentence_scores_a
java -XX:MaxHeapSize=4G -jar sentence_scoring.jar $tmp_folder/input_b $tmp_folder/sentence_scores_b
echo "...Sentence scoring done."

echo "Running feature extraction..."
java -XX:MaxHeapSize=4G -jar feature_extraction.jar $tmp_folder/input_a $tmp_folder/features_a $tmp_folder/sentence_scores_a
java -XX:MaxHeapSize=4G -jar feature_extraction.jar $tmp_folder/input_b $tmp_folder/features_b $tmp_folder/sentence_scores_b
# Unzip the resulting data:
gunzip $tmp_folder/features_a/training-data.arff.gz
gunzip $tmp_folder/features_b/training-data.arff.gz
echo "...Feature extraction done."

echo "Running BERT feature extraction..."
python extract_bert_mask_entropy.py $tmp_folder/input_a $tmp_folder/bert_a
python extract_bert_mask_entropy.py $tmp_folder/input_b $tmp_folder/bert_b
echo "...BERT feature extraction done."

echo "Merging all features..."
python merge_features.py $tmp_folder/features_a/training-data.arff $tmp_folder/bert_a $tmp_folder/input_a $tmp_folder/merged_a
python merge_features.py $tmp_folder/features_b/training-data.arff $tmp_folder/bert_b $tmp_folder/input_b $tmp_folder/merged_b
echo "...Features merged, imputing"
python impute_data.py $tmp_folder/merged_a $tmp_folder/imputed_a
python impute_data.py $tmp_folder/merged_b $tmp_folder/imputed_b
echo "...Imputing done."

echo "Fixing feature positions..."
python fix_feature_order.py $tmp_folder/imputed_a $tmp_folder/cleaned_a
python fix_feature_order.py $tmp_folder/imputed_b $tmp_folder/cleaned_b
echo "...done."

echo "Generating final files..."
python merge_texts.py $tmp_folder/cleaned_a $tmp_folder/cleaned_b $output_folder
echo "Removing tmp folder..."
rm -r $tmp_folder
rm -r target
echo "...feature extraction fully done."

