# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling_switch as modeling
import optimization
import tensorflow as tf
from time import time
import datetime

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer("sample_num", '126',
                     "total sample number")

flags.DEFINE_integer("mid_save_step", '15000',
                     "Epoch is so long, mid_save_step 15000 is roughly 3 hours")

flags.DEFINE_string("input_file", 'output/test.tfrecord',
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'adaptation', 
                    "The name of the task to train.")

flags.DEFINE_string("vocab_file", 'uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", './L-12_H-768_A-12_adapted',
                    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("init_checkpoint", 'uncased_L-12_H-768_A-12/bert_model.ckpt',
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer("max_seq_length", 320,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 10,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", True, 
                  "Whether to run training.")

flags.DEFINE_bool("do_eval", True, 
                  "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 8, 
                     "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, 
                     "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, 
                   "The initial learning rate for Adam.")

flags.DEFINE_float("warmup_proportion", 0.1, 
                   "Number of warmup steps.")

flags.DEFINE_integer("num_train_epochs", 10, 
                     "num_train_epochs.")



def model_fn_builder(features, is_training, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    input_ids, input_mask, segment_ids, switch_ids, masked_lm_positions, \
    masked_lm_ids, masked_lm_weights, next_sentence_labels = features

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        switch_ids=switch_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    matrix = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
              next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels)

    return train_op, total_loss, matrix, input_ids



def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
              next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])  # [batch_size*max_predictions_per_seq, dim]
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)                # [batch_size*max_predictions_per_seq, ]
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])  # [batch_size, 2]
    next_sentence_predictions = tf.argmax(
        next_sentence_log_probs, axis=-1, output_type=tf.int32)            # [batch_size, ]
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.metrics.mean(
        values=next_sentence_example_loss)
    # next_sentence_mean_loss = tf.reduce_mean(next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }



def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)  # [batch_size*max_predictions_per_seq, dim]

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)   # [batch_size*max_predictions_per_seq, vocab_size]

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])  # [batch_size*max_predictions_per_seq, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)               # [1, ]
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)  # [batch_size, 2]
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # [batch_size, ]
    loss = tf.reduce_mean(per_example_loss)                                 # [1, ]
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  # sequence_tensor = [batch_size, seq_length, width]
  # positions = [batch_size, max_predictions_per_seq]
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example



def run_epoch( epoch, sess, evaluate, eval_op, input_ids, lm_losses, saver, root_path, save_step, mid_save_step, phase, batch_size=16, train_op=tf.constant(0)):
    t_loss = 0
    n_all = 0
    t0 = time()
    t1 = time()

    masked_lm_accuracy = 0.0
    masked_lm_mean_loss = 0.0
    next_sentence_accuracy = 0.0
    next_sentence_mean_loss = 0.0

    step = 0

    print('running begin ... ')
    try:
        while True:
            step = step + 1
            y, matrix, batch_loss, _, _ = sess.run([input_ids, evaluate, lm_losses, train_op, eval_op] )
            masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss = matrix

            n_sample = len(y)
            n_all += n_sample

            t_loss += batch_loss * n_sample
            # save every epoch or 3 hour
            # if (step % save_step == 0) or (step % 15000 == 0):
            if (step % mid_save_step == 2):
                # c_time = str(datetime.datetime.now()).replace(' ', '-').split('.')[0]
                c_time = str(int(time()))
                save_path = os.path.join(root_path, 'bert_model_{0}_epoch_{1}'.format(c_time, epoch))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                saver.save(sess, os.path.join(save_path,'bert_model_{}.ckpt'.format(c_time)), global_step = step)
                print('save model epoch {}'.format(int(step/save_step)))
                print('masked_lm_accuracy  {:.6f},  masked_lm_mean_loss {:.6f},  next_sentence_accuracy {:.6f},  next_sentence_mean_loss{:.6f}'.format(
                    masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss
                ))

                print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
                      format(phase, t_loss / n_all, time() - t1))
                t1=time()
                print('Sample seen {} total time {}'.format(n_all,time() - t0))

    except tf.errors.OutOfRangeError:
        print('Epoch {} Done'.format(epoch))
        # c_time = str(datetime.datetime.now()).replace(' ', '-').split('.')[0]
        c_time = str(int(time()))
        save_path = os.path.join(root_path, 'bert_model_{0}_epoch_{1}'.format(c_time, step / save_step))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(sess, os.path.join(save_path, 'bert_model_{}.ckpt'.format(c_time)), global_step=step)
        print('save model epoch {}'.format(int(step / save_step)))

        print(
            'masked_lm_accuracy  {:.6f},  masked_lm_mean_loss {:.6f},  next_sentence_accuracy {:.6f},  next_sentence_mean_loss{:.6f}'.format(
                masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss
            ))
        print("{} Loss: {:.4f}, {:.2f} Seconds Used:".
              format(phase, t_loss / n_all, time() - t1))
        t1 = time()
        print('Sample seen {} total time {}'.format(n_all, time() - t0))
        pass



def parse_exmp(serial_exmp):
    input_data = tf.parse_single_example(serial_exmp,
                                       features={
                                           "input_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "switch_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "masked_lm_positions":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_ids":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
                                           "masked_lm_weights":
                                               tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
                                           "next_sentence_labels":
                                               tf.FixedLenFeature([1], tf.int64),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    input_ids = input_data["input_ids"]
    input_mask = input_data["input_mask"]
    segment_ids = input_data["segment_ids"]
    switch_ids = input_data["switch_ids"]
    m_lp = input_data["masked_lm_positions"]
    m_lids = input_data["masked_lm_ids"]
    m_lm_w = input_data["masked_lm_weights"]
    nsl = input_data["next_sentence_labels"]
    return input_ids, input_mask, segment_ids, switch_ids, m_lp, m_lids, m_lm_w, nsl


def print_configuration_op(FLAGS):
    print('My Configurations:')
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    print_configuration_op(FLAGS)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    root_path = FLAGS.output_dir
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    num_train_steps = FLAGS.sample_num // FLAGS.train_batch_size * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    buffer_size = 1000
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(FLAGS.train_batch_size)
    iterator = dataset.make_initializable_iterator()
    save_step = FLAGS.sample_num // FLAGS.train_batch_size

    input_ids, input_mask, segment_ids, switch_ids, masked_lm_positions, \
        masked_lm_ids, masked_lm_weights, next_sentence_labels = iterator.get_next()
    features = [input_ids, input_mask, segment_ids, switch_ids, masked_lm_positions, \
        masked_lm_ids, masked_lm_weights, next_sentence_labels]
    train_op, loss, matrix, input_ids = model_fn_builder(
        features,  # ----model_fn_builder----
        is_training=True,
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)


    masked_lm_accuracy, masked_acc_op = matrix["masked_lm_accuracy"]
    masked_lm_mean_loss, masked_loss_op= matrix["masked_lm_loss"]
    next_sentence_accuracy, next_sentence_op = matrix["next_sentence_accuracy"]
    next_sentence_mean_loss, next_sentence_loss_op = matrix["next_sentence_loss"]

    evaluate = [masked_lm_accuracy, masked_lm_mean_loss, next_sentence_accuracy, next_sentence_mean_loss]
    eval_op = [masked_acc_op, masked_loss_op, next_sentence_op, next_sentence_loss_op]

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(FLAGS.num_train_epochs):
            sess.run(iterator.initializer, feed_dict={filenames: [FLAGS.input_file]})
            run_epoch(epoch, sess, evaluate, eval_op, input_ids, loss, saver, root_path, save_step, 
              FLAGS.mid_save_step,'train', batch_size=FLAGS.train_batch_size, train_op=train_op)



if __name__ == "__main__":
    tf.app.run()

