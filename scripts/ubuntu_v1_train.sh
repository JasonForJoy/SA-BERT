
CUDA_VISIBLE_DEVICES=0 python -u ../train.py \
  --task_name Ubuntu_V1_Xu \
  --train_dir ../data/Ubuntu_V1_Xu/processed_train.tfrecord \
  --valid_dir ../data/Ubuntu_V1_Xu/processed_valid.tfrecord \
  --output_dir ../output/Ubuntu_V1_Xu \
  --do_lower_case True \
  --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 512 \
  --do_train True  \
  --do_eval True  \
  --train_batch_size 25 \
  --eval_batch_size 25 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --warmup_proportion 0.1 > log_train.txt 2>&1 &