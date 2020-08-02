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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import operator
from time import time
from collections import defaultdict
import tensorflow as tf
import optimization
import tokenization
import modeling_switch as modeling
import metrics

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("test_dir", 'test.tfrecord',
                    "The input test data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("restore_model_dir", 'output/',
                    "The output directory where the model checkpoints have been written.")

flags.DEFINE_string("task_name", 'TestModel', 
                    "The name of the task.")

flags.DEFINE_string("bert_config_file", 'uncased_L-12_H-768_A-12/bert_config.json',
                    "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")

flags.DEFINE_integer("max_seq_length", 320,
                     "The maximum total input sequence length after WordPiece tokenization. "
                     "Sequences longer than this will be truncated, and sequences shorter "
                     "than this will be padded.")

flags.DEFINE_bool("do_eval", True, 
                  "Whether to run eval on the dev set.")

flags.DEFINE_integer("eval_batch_size", 32, 
                     "Total batch size for predict.")


def print_configuration_op(FLAGS):
    print('My Configurations:')
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


def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return  sample_nums


def print_weight(name):
    with open('valid/weight_log' + name + str(random.randint(0, 100)), 'w') as fw:
        variables = tf.trainable_variables()
        for variable in variables:
            fw.write(str(variable.eval()))
            fw.write('\n')


def parse_exmp(serial_exmp):
    input_data = tf.parse_single_example(serial_exmp,
                                       features={
                                           "ques_ids":
                                               tf.FixedLenFeature([], tf.int64),
                                           "ans_ids":
                                               tf.FixedLenFeature([], tf.int64),
                                           "input_sents":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "input_mask":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "segment_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "switch_ids":
                                               tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                                           "label_ids":
                                               tf.FixedLenFeature([], tf.float32),
                                       }
                                       )
    # So cast all int64 to int32.
    for name in list(input_data.keys()):
        t = input_data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        input_data[name] = t

    ques_ids = input_data["ques_ids"]
    ans_ids = input_data['ans_ids']
    sents = input_data["input_sents"]
    mask = input_data["input_mask"]
    segment_ids= input_data["segment_ids"]
    switch_ids= input_data["switch_ids"]
    labels = input_data['label_ids']
    return ques_ids, ans_ids, sents, mask, segment_ids, switch_ids, labels


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, switch_ids, labels, ques_ids, ans_ids,
                 num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      switch_ids=switch_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  target_loss_weight = [1.0, 1.0]
  target_loss_weight = tf.convert_to_tensor(target_loss_weight)

  flagx = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
  flagy = tf.cast(tf.equal(labels, 0), dtype=tf.float32)

  all_target_loss = target_loss_weight[1] * flagx + target_loss_weight[0] * flagy

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    # if is_training:
    #   # I.e., 0.1 dropout
    #   output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.layers.dropout(output_layer, rate=0.1, training=is_training)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    probabilities = tf.sigmoid(logits, name="prob")
    logits = tf.squeeze(logits,[1])
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = tf.multiply(losses, all_target_loss)

    mean_loss = tf.reduce_mean(losses, name="mean_loss") +  sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.sign(probabilities - 0.5), tf.sign(labels - 0.5))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
    #
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)

    return mean_loss, logits, probabilities, accuracy, model, output_layer


best_score = 0.0
def run_test(dir_path, op_name, sess, training, accuracy, prob, pair_ids, output_layer):
    results = defaultdict(list)
    num_test = 0
    num_correct = 0.0
    n_updates = 0
    mrr = 0
    t0 = time()
    try:
        while True:
            n_updates += 1

            batch_accuracy, predicted_prob, pair_ = sess.run([accuracy, prob, pair_ids], feed_dict={training: False})
            question_id, answer_id, label = pair_
            
            num_test += len(predicted_prob)
            num_correct += len(predicted_prob) * batch_accuracy
            for i, prob_score in enumerate(predicted_prob):
                # question_id, answer_id, label = pair_id[i]
                results[question_id[i]].append((answer_id[i], label[i], prob_score[0]))

            if n_updates%2000 == 0:
                tf.logging.info("n_update %d , %s: Mins Used: %.2f" %
                                (n_updates, op_name, (time() - t0) / 60.0))

    except tf.errors.OutOfRangeError:
        # calculate top-1 precision
        print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct / num_test))
        accu, precision, recall, f1, loss = metrics.classification_metrics(results)
        print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

        mvp = metrics.mean_average_precision(results)
        mrr = metrics.mean_reciprocal_rank(results)
        top_1_precision = metrics.top_1_precision(results)
        total_valid_query = metrics.get_num_valid_query(results)
        print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(
            mvp, mrr, top_1_precision, total_valid_query))

        out_path = os.path.join(dir_path, "output_test.txt")
        print("Saving evaluation to {}".format(out_path))
        with open(out_path, 'w') as f:
          f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
          for us_id, v in results.items():
            v.sort(key=operator.itemgetter(2), reverse=True)
            for i, rec in enumerate(v):
              r_id, label, prob_score = rec
              rank = i+1
              f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, r_id, prob_score, rank, label))
    return mrr


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    print_configuration_op(FLAGS)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    test_data_size = total_sample(FLAGS.test_dir)
    tf.logging.info('test data size: {}'.format(test_data_size))

    filenames = tf.placeholder(tf.string, shape=[None])
    shuffle_size = tf.placeholder(tf.int64)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)  # Parse the record into tensors.
    dataset = dataset.repeat(1)
    # dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(FLAGS.eval_batch_size)
    iterator = dataset.make_initializable_iterator()
    ques_ids, ans_ids, sents, mask, segment_ids, switch_ids, labels = iterator.get_next()  # output dir
    pair_ids = [ques_ids, ans_ids, labels]

    training = tf.placeholder(tf.bool)
    mean_loss, logits, probabilities, accuracy, model, output_layer = create_model(bert_config,
                                                                       is_training = training,
                                                                       input_ids = sents,
                                                                       input_mask = mask,
                                                                       segment_ids = segment_ids,
                                                                       switch_ids = switch_ids,
                                                                       labels = labels,
                                                                       ques_ids = ques_ids,
                                                                       ans_ids = ans_ids,
                                                                       num_labels = 1,
                                                                       use_one_hot_embeddings = False)


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    if FLAGS.do_eval:
        with tf.Session(config=config) as sess:
            tf.logging.info("*** Restore model ***")

            ckpt = tf.train.get_checkpoint_state(FLAGS.restore_model_dir)
            variables = tf.trainable_variables()
            saver = tf.train.Saver(variables)
            saver.restore(sess, ckpt.model_checkpoint_path)

            tf.logging.info('Test begin')
            sess.run(iterator.initializer,
                     feed_dict={filenames: [FLAGS.test_dir], shuffle_size: 1})
            run_test(FLAGS.restore_model_dir, "test", sess, training, accuracy, probabilities, pair_ids, output_layer)


if __name__ == "__main__":

  tf.app.run()

