
CUDA_VISIBLE_DEVICES=3 python -u ../adapt_switch.py \
  --task_name adaptation \
  --sample_num 5000000 \
  --mid_save_step 15000 \
  --input_file ../data/Ubuntu_V1_Xu/pretrain_data.tfrecord \
  --output_dir ../uncased_L-12_H-768_A-12_adapted \
  --vocab_file ../uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length 512 \
  --max_predictions_per_seq 25 \
  --train_batch_size 20 \
  --eval_batch_size 20 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --warmup_proportion 0.1 > log_adaptation.txt 2>&1 &
