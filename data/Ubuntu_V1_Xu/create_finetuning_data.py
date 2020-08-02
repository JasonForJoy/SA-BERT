# coding=utf-8
import collections
import tokenization
import tensorflow as tf
from tqdm import tqdm


tf.flags.DEFINE_string("response_file", "./Ubuntu_Corpus_V1/responses.txt", 
                       "path to response file")
tf.flags.DEFINE_string("train_file", "./Ubuntu_Corpus_V1/train.txt", 
	                   "path to train file")
tf.flags.DEFINE_string("valid_file", "./Ubuntu_Corpus_V1/valid.txt", 
	                   "path to valid file")
tf.flags.DEFINE_string("test_file", "./Ubuntu_Corpus_V1/test.txt", 
                       "path to test file")

tf.flags.DEFINE_string("vocab_file", "../../uncased_L-12_H-768_A-12/vocab.txt", 
                       "path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 512, 
	                    "max sequence length of concatenated context and response")
tf.flags.DEFINE_bool("do_lower_case", True,
                     "whether to lower case the input text")



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


def load_responses(fname):
    responses={}
    with open(fname, 'rt') as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')
            if len(fields) != 2:
                print("WRONG LINE: {}".format(line))
                r_text = 'unknown'
            else:
                r_text = fields[1]
            responses[fields[0]] = r_text
    return responses


def load_dataset(fname, responses):

    processed_fname = "processed_" + fname.split("/")[-1]
    dataset_size = 0
    print("Generating the file of {} ...".format(processed_fname))

    with open(processed_fname, 'w') as fw:
        with open(fname, 'rt') as fr:
            for line in fr:
                line = line.strip()
                fields = line.split('\t')
                
                us_id = fields[0]
                context = fields[1]

                if fields[2] != "NA":
                    pos_ids = [id for id in fields[2].split('|')]
                    for r_id in pos_ids:
                        r_utter = responses[r_id]
                        dataset_size += 1
                        fw.write("\t".join([str(us_id), context, r_id, r_utter, 'follow']))
                        fw.write('\n')

                if fields[3] != "NA":
                    neg_ids = [id for id in fields[3].split('|')]
                    for r_id in neg_ids:
                        r_utter = responses[r_id]
                        dataset_size += 1
                        fw.write("\t".join([str(us_id), context, r_id, r_utter, 'unfollow']))
                        fw.write('\n')
    
    print("{} dataset_size: {}".format(processed_fname, dataset_size))            
    return processed_fname


class InputExample(object):
    def __init__(self, guid,ques_ids, text_a, ans_ids, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, ques_ids, ans_ids, input_sents, input_mask, segment_ids, switch_ids, label_id):
        self.ques_ids = ques_ids
        self.ans_ids = ans_ids
        self.input_sents = input_sents
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.switch_ids=switch_ids
        self.label_id = label_id

def read_processed_file(input_file):
    lines = []
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            concat = []
            temp = line.rstrip().split('\t')
            concat.append(temp[0]) # contxt id
            concat.append(temp[1]) # contxt
            concat.append(temp[2]) # response id
            concat.append(temp[3]) # response
            concat.append(temp[4]) # label
            lines.append(concat)
    return lines

def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        ques_ids = line[0]
        text_a = tokenization.convert_to_unicode(line[1])
        ans_ids = line[2]
        text_b = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(InputExample(guid=guid, ques_ids=ques_ids, text_a=text_a, ans_ids=ans_ids, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i

    features = []  # feature
    for (ex_index, example) in enumerate(examples):
        ques_ids = int(example.ques_ids)
        ans_ids = int(example.ans_ids)

        # tokens_a = tokenizer.tokenize(example.text_a)  # text_a tokenize
        text_a_utters = example.text_a.split(" __EOS__ ")
        tokens_a = []
        text_a_switch = []
        for text_a_utter_idx, text_a_utter in enumerate(text_a_utters):
            if text_a_utter_idx%2 == 0:
                text_a_switch_flag = 0
            else:
                text_a_switch_flag = 1
            text_a_utter_token = tokenizer.tokenize(text_a_utter + " __EOS__")
            tokens_a.extend(text_a_utter_token)
            text_a_switch.extend([text_a_switch_flag]*len(text_a_utter_token))
        assert len(tokens_a) == len(text_a_switch)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  # text_b tokenize

        if tokens_b:  # if has b
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # truncate
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because  # (?)
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        switch_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        switch_ids.append(0)
        for token_idx, token in enumerate(tokens_a):
            tokens.append(token)
            segment_ids.append(0)
            switch_ids.append(text_a_switch[token_idx])
        tokens.append("[SEP]")
        segment_ids.append(0)
        switch_ids.append(0)

        if tokens_b:
            for token_idx, token in enumerate(tokens_b):
                tokens.append(token)
                segment_ids.append(1)
                switch_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            switch_ids.append(1)

        input_sents = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_sents)  # mask

        # Zero-pad up to the sequence length.
        while len(input_sents) < max_seq_length:
            input_sents.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            switch_ids.append(0)

        assert len(input_sents) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(switch_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index%2000 == 0:
            print('convert_{}_examples_to_features'.format(ex_index))

        features.append(
            InputFeatures(  # object
                ques_ids=ques_ids,
                ans_ids = ans_ids,
                input_sents=input_sents,
                input_mask=input_mask,
                segment_ids=segment_ids,
                switch_ids=switch_ids,
                label_id=label_id))

    return features


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["ques_ids"] = create_int_feature([instance.ques_ids])
        features["ans_ids"] = create_int_feature([instance.ans_ids])
        features["input_sents"] = create_int_feature(instance.input_sents)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["switch_ids"] = create_int_feature(instance.switch_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    print("write_{}_instance_to_example_files".format(total_written))

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


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS
    print_configuration_op(FLAGS)

    responses = load_responses(FLAGS.response_file)
    train_filename = load_dataset(FLAGS.train_file, responses)
    valid_filename = load_dataset(FLAGS.valid_file, responses)
    test_filename  = load_dataset(FLAGS.test_file, responses)

    filenames = [train_filename, valid_filename, test_filename]
    filetypes = ["train", "valid", "test"]
    files = zip(filenames, filetypes)

    label_list = ["unfollow", "follow"]
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    for (filename, filetype) in files:
        examples = create_examples(read_processed_file(filename), filetype)
        features = convert_examples_to_features(examples, label_list, FLAGS.max_seq_length, tokenizer)
        new_filename = filename[:-4] + ".tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
