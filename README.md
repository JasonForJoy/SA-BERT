# BERT for Multi-Turn Response Selection
Implementation of the BERT model for multi-turn response selection in retrieval-based chatbots with Tensorflow

## Dependencies
Python 3.6 <br>
Tensorflow 1.13.1

## Download 
- Download the [BERT](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), 
  and move to path: ./uncased_L-12_H-768_A-12 <br>
  
- Download the [Ubuntu V1](https://drive.google.com/file/d/1-rNv34hLoZr300JF3v7nuLswM7GRqeNc/view),
  and move to path: ./data/Ubuntu_V1_Xu/Ubuntu_Corpus_V1 <br>

## Preprocess
Generate the tfrecord data
```
cd data/Ubuntu_V1_Xu/
python data_preprocess.py 
```

## Training
```
cd scripts/
bash ubuntu_v1_train.sh
```

## Testing
Modify the variable ```restore_model_dir``` in ```ubuntu_v1_test.sh```
```
cd scripts/
bash ubuntu_v1_test.sh
```
A "output_test.txt" file which records scores for each context-response pair will be saved to the path of ```restore_model_dir```. <br>
Copy this ```output_test.txt``` file to ```scripts/``` and run ```python compute_metrics.py```, various metrics will be shown.

## Update
Please feel free to open issues if you have some problems.
