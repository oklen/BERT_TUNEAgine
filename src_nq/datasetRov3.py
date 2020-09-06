import os
import sys
import six
import json
import pandas as pd

import logging

from generate_exampleDreamAB import DrExample,EdgeType,InputFeatures

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NqDataset(object):
    def __init__(self, args, input_file, is_training):
        if not is_training:
            self.examples = self.read_nq_examples(input_path=input_file,args = args)

        prefix = "cached_{0}_{1}_{2}_{3}".format(str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length),args.DataName)
        prefix = os.path.join(args.feature_path, prefix)

        cached_path = os.path.join(prefix, os.path.split(input_file)[1] + ".pkl")
        self.features = self.read_nq_features(cached_path, is_training)

    @staticmethod
    def read_nq_examples(input_path,args=None):
        logging.info("Reading examples from {}.".format(input_path))
        examples = []
        exCnt = args.train_size
        train = pd.read_csv(args.input_pattern, sep='\t', header=0)
        for i in range(len(train)):
            exCnt = exCnt -1
            if(exCnt == 0): break
            examples.append(DrExample(train['sentence'][i].split(' '),train['label'][i]))
            example = DrExample()
            examples.append(example)
        return examples

    @staticmethod
    def read_nq_features(cached_path, is_training=False):
        if not os.path.exists(cached_path):
            logging.info("{} doesn't exists.".format(cached_path))
            exit(0)
        logging.info("Reading features from {}.".format(cached_path))
        features = None
        with open(cached_path, "rb") as reader:
            features = pickle.load(reader)

#        for i, feature in enumerate(features):
#            feature.unique_id = i
        return features
