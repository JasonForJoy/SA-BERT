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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import numpy as np
import tensorflow as tf
from tqdm import tqdm

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_file", './Ubuntu_Corpus_V1/train.txt',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("response_file", './Ubuntu_Corpus_V1/responses.txt',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("output_file", './pretrain_data.tfrecord',
                    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", '../../uncased_L-12_H-768_A-12/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool("do_lower_case", True,
                  "Whether to lower case the input text. Should be True for uncased "
                  "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 512, 
                     "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 25,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, 
                     "Random seed for data generation.")

flags.DEFINE_integer("dupe_factor", 10,
                     "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, 
                   "Masked LM probability.")

flags.DEFINE_float("short_seq_prob", 0.1,
                   "Probability of creating sequences which are shorter than the maximum length.")



class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, switch_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.switch_ids = switch_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "switch_ids: %s\n" % (" ".join([str(x) for x in self.switch_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    switch_ids = list(instance.switch_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      switch_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(switch_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["switch_ids"] = create_int_feature(switch_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(context, response, switch, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  sid_r = np.arange(0, len(context))
  rng.shuffle(sid_r)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in tqdm(range(dupe_factor)):
      for i in tqdm(range(len(sid_r))):

          sent_a = []
          switch_a = []
          for j in range(len(context[i])):
              utterance_a = context[i][j]
              utterance_a = tokenization.convert_to_unicode(utterance_a)
              utterance_a = tokenizer.tokenize(utterance_a)
              sent_a.extend(utterance_a)
              switch_a.extend([switch[i][j]] * len(utterance_a))
          assert len(sent_a) == len(switch_a)

          if random.random() < 0.5:
              sent_b = response[sid_r[i]]
              is_random_next = True
          else:
              sent_b = response[i]
              is_random_next = False

          sent_b = tokenization.convert_to_unicode(sent_b)
          sent_b = tokenizer.tokenize(sent_b)
          instances.extend(
              create_instances_from_document(
                  sent_a, sent_b, switch_a, is_random_next, max_seq_length, short_seq_prob,
                  masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    tokens_a, tokens_b, switch_a, is_random_next, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []

  truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

  assert len(tokens_a) >= 1
  assert len(tokens_b) >= 1

  tokens = []
  segment_ids = []
  switch_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  switch_ids.append(0)
  for i, token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(0)
    switch_ids.append(switch_a[i])

  tokens.append("[SEP]")
  segment_ids.append(0)
  switch_ids.append(0)

  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)
    switch_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)
  switch_ids.append(1)

  (tokens, masked_lm_positions,
   masked_lm_labels) = create_masked_lm_predictions(
      tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
  instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        switch_ids=switch_ids,
        is_random_next=is_random_next,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
  instances.append(instance)

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def print_configuration_op(FLAGS):
  print('My Configurations:')
  for name, value in FLAGS.__flags.items():
    value = value.value
    if type(value) == float:
      print(' %s:\t %f' % (name, value))
    elif type(value) == int:
      print(' %s:\t %d' % (name, value))
    elif type(value) == str:
      print(' %s:\t %s' % (name, value))
    elif type(value) == bool:
      print(' %s:\t %s' % (name, value))
    else:
      print('%s:\t %s' % (name, value))
  print('End of configuration')
  
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  print_configuration_op(FLAGS)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # 1. load context-response pairs
  response_dict = {}
  with open(FLAGS.response_file, 'rt') as f:
      for line in f:
          line = line.strip()
          fields = line.split('\t')
          if len(fields) != 2:
              print("WRONG LINE: {}".format(line))
              r_text = 'unknown'
          else:
              r_text = fields[1]
          response_dict[fields[0]] = r_text

  context = []
  response = []
  switch = []
  with open(FLAGS.train_file, 'rb') as f:
      lines = f.readlines()
      for index, line in enumerate(lines):
          line = line.decode('utf-8').strip()
          fields = line.split('\t')
          context_i = fields[1]
          utterances_i = context_i.split(" __EOS__ ")
          # utterances = [utterance + " __EOS__" for utterance in utterances]
          new_utterances_i = []
          switch_i = []
          for j, utterance in enumerate(utterances_i):
              new_utterances_i.append(utterance + " __EOS__")
              if j%2 == 0:
                  switch_i.append(0)
              else:
                  switch_i.append(1)
          assert len(new_utterances_i) == len(switch_i)

          if fields[2] != "NA":
              pos_ids = [id for id in fields[2].split('|')]
              for r_id in pos_ids:
                  context.append(new_utterances_i)

                  switch.append(switch_i)

                  response_i = response_dict[r_id]
                  response.append(response_i)

          if index % 10000 == 0:
              print('Done:', index)

  tf.logging.info("Reading from input files: {} context-response pairs".format(len(context)))

  
  # 2. create training instances
  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      context, response, switch, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  
  # 3. write instance to example files
  output_files = [FLAGS.output_file]
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("train_file")
  flags.mark_flag_as_required("response_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()

