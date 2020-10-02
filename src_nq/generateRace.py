# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:27:12 2020

@author: oklen
"""

import csv
import sys
from io import open
import json
from os import listdir
from os.path import isfile, join
import logging

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, all_sen):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.all_sen = all_sen

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # @classmethod
    # def _read_tsv(cls, input_file, quotechar=None, remove_header=False):
    #     """Reads a tab separated value file."""
    #     with open(input_file, "r", encoding="utf-8-sig") as f:
    #         reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    #         lines = []
    #         if remove_header:
    #             next(reader)
    #         for line in reader:
    #             if sys.version_info[0] == 2:
    #                 line = list(unicode(cell, 'utf-8') for cell in line)
    #             lines.append(line)
    #         return lines


class RaceProcessor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'RACE'

    def _read_samples(self, data_dir, set_type, level=None):
        # if self.level == None:
        #     data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
        #                  '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        # else:
        # data_dirs = ['{}/{}/{}'.format(data_dir, set_type, self.level)]
        if level is None:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
                         '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        else:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, level)]

        examples = []
        example_id = 0
        for data_dir in data_dirs:
            # filenames = glob.glob(data_dir + "/*txt")
            filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    article = data_raw['article']
                    for i in range(len(data_raw['answers'])):
                        example_id += 1
                        truth = str(ord(data_raw['answers'][i]) - ord('A'))
                        question = data_raw['questions'][i]
                        options = data_raw['options'][i]
                        for k in range(len(options)):
                            guid = "%s-%s-%s" % (set_type, example_id, k)
                            option = options[k]
                            examples.append(
                                    InputExample(guid=guid, text_a=article, text_b=option, label=truth,
                                                 text_c=question))

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


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 do_lower_case=False, is_multi_choice=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    if is_multi_choice:
        features = [[]]
    else:
        features = []
    for (ex_index, example) in enumerate(examples):
        if do_lower_case:
            example.text_a = example.text_a.lower()
            example.text_b = example.text_b.lower()
            example.text_c = example.text_c.lower()

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        all_sen = []
        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 3)
        elif example.text_b and not example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        
        tokens = tokens_a + [sep_token]
        all_sen.append([1,len(tokens)+1])
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_c:
            # tokens_b += [sep_token] + tokens_c
            tokens_b += tokens_c

        if tokens_b:
            nb = len(tokens)+2
            tokens += tokens_b + [sep_token]
            all_sen.append(nb,len(tokens))
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode in ["classification", "multi-choice"]:
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                all_sen=all_sen))
        if len(features[-1]) == num_labels:
            features.append([])
    if is_multi_choice:
        if len(features[-1]) == 0:
            features = features[:-1]

    return features


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()
