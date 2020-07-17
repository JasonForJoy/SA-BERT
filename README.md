# SA-BERT for Multi-Turn Response Selection
This repository contains the source code and pre-trained models for the CIKM 2020 paper [Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/2004.03588.pdf) by Gu et al. <br>

## Results
<img src="image/UbuntuV1_V2.png">
<img src="image/Douban_Ecommerce.png">

## Cite
If you use the source code and pre-trained models, please cite the following paper:
**"Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots"**
Jia-Chen Gu, Tianda Li, Quan Liu, Zhen-Hua Ling, Zhiming Su, Si Wei, Xiaodan Zhu. _CIKM (2020)_

```
 @inproceedings{gu2020speaker,
  author = {Gu, Jia-Chen and
            Li, Tianda and
            Liu, Quan and
            Ling, Zhen-Hua and
            Su Zhiming and
            Wei Si and 
            Zhu Xiaodan
            },
  title = {Speaker-Aware BERT for Multi-Turn Response Selection in Retrieval-Based Chatbots},
  booktitle = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management},
  series = {CIKM '20},
  year = {2020},
  publisher = {ACM},
  } 
```


## Updating
Currently, this repository contains only the implementation of fine-tuning BERT for multi-turn response selection in retrieval-based chatbots with Tensorflow. <br>
We will release our code and model as soon as possible. Please stay tuned if you are interested.

## Results
| Model       |  R_2@1  |  R_10@1  |  R_10@2  |  R_10@5  |
| ----------- | ------- | -------- | -------- | -------- |
| BERT_base   | 0.950   |  0.810   |  0.897   |  0.975   |

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
