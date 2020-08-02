
CUDA_VISIBLE_DEVICES=3 python -u ../test.py \
    --test_dir ../data/Ubuntu_V1_Xu/processed_test.tfrecord \
    --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
    --max_seq_length 512 \
    --eval_batch_size 50 \
    --restore_model_dir ../output/Ubuntu_V1_Xu/1569550213 > log_test.txt 2>&1 &
