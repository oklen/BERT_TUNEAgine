# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function
from transformers import AutoTokenizer, AutoModelWithLMHead,AlbertTokenizer,RobertaTokenizer

import argparse
import collections
import json
import logging
import enum
import os
import random
import sys
import re
import pandas as pd

sys.path.append("/nq_model/")

from io import open

import numpy as np
import torch
from tqdm import tqdm

from glob import glob
import multiprocessing

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


from spacy.lang.en import English
from modules.graph_encoderAB import NodePosition, Graph, EdgeType, get_edge_position


nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4
    

class DrExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 talk,
                 question,
                 choice,
                 answer,unique_id):
        self.talk = talk
        self.question = question
        self.choice = choice
        self.answer = answer
        self.unique_id = unique_id


class InputFeatures(object):
    """A single set of features of data."""
    
    def show(self):
        print("unique_id:",self.unique_id)
        print("example_id:",self.example_id)
        print("doc_span_index:",self.doc_span_index)
        print("tokens:",self.tokens)
        print("token_to_orig_map:",self.token_to_orig_map)
        print("token_is_max_context:",self.token_is_max_context)
        print("input_ids:",self.input_ids)
        print("input_mask:",self.input_mask)
        print("segment_ids:",self.segment_ids)

        
    def __init__(self,
                 unique_id,
                 all_sen,
                 input_ids,
                 input_mask,
                 segment_ids,
                 graph,
                 label):
        self.unique_id = unique_id

#        self.doc_span_index = doc_span_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.all_sen = all_sen
        self.segment_ids = segment_ids
        self.graph = graph
        self.label = label
        #self.show()

        #exit(0)


class NodeInfo(object):
    def __init__(self,
                 start_position,
                 end_position,
                 node_idx):
        self.start_position = start_position
        self.end_position = end_position
        self.node_idx = node_idx


def get_doc_tree(is_sentence_end, is_paragraph_end, orig_tok_idx):
    doc_len = len(is_sentence_end)
    document = []
    paragraph = []
    sentence = []
    for i in range(doc_len):
        sentence.append((orig_tok_idx[i], i))

        if is_sentence_end[i]:
            paragraph.append((-1, sentence))
            sentence = []
        if is_paragraph_end[i]:
            assert len(sentence) == 0
            document.append(paragraph)
            paragraph = []
    assert len(sentence) == 0
    assert len(paragraph) == 0
    return document


def get_candidate_type(token, counts, max_position):
    if token == "<Table>":
        counts["Table"] += 1
        return min(counts["Table"], max_position) - 1
    elif token == "<P>":
        counts["Paragraph"] += 1
        return min(counts["Paragraph"] + max_position, max_position * 2) - 1
    elif token in ("<Ul>", "<Dl>", "<Ol>"):
        counts["List"] += 1
        return min(counts["List"] + max_position * 2, max_position * 3) - 1
    elif token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        counts["Other"] += 1
        return min(counts["Other"] + max_position * 3, max_position * 4) - 1
    else:
        logging.info("Unknoww candidate type found: %s", token)
        counts["Other"] += 1
        return min(counts["Other"] + max_position * 3, max_position * 4) - 1


def convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    """Loads a data file into a list of `InputBatch`s."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    features = []
    
    max_len_exceed = 0
    
    for example in examples:
        name = set()
        tok_is_sentence_begin = []
        tok_is_sentence_end = []
        tok_is_question_begin = None
        tok_is_question_end = None
        tok_is_choice_begin = None
        tok_is_choice_end = None
        input_ids = []
        feature_now = []
        tokens = ['[CLS]']

        for sentence in example.talk:

            tok_is_sentence_begin.append(len(tokens))
#            tokens.append('[CLS]')
#            sentence = sentence[3:-1]  //Dont't erase W: M:
            sentence = tokenizer.tokenize(sentence)
            tokens += sentence
            name.add(tokens[tok_is_sentence_begin[-1]])
            tok_is_sentence_end.append(len(tokens))
        names = []
        for i in name:
            names.append(i)
        #tokens.append('[SEP]')

            
#        tok_is_question_begin = len(tokens)
#
#        tokens += tokenizer.tokenize('Q:'+example.question[:-1]) #"ADD Q: for classification"
#        tok_is_question_end=len(tokens)
#        print(example.question[:-1])
#        tokens.append('[SEP]') for merge QA 1/2

        label = 0
#        print(example.choice)
#        print(example.answer)
        for choice in example.choice:
            
            toPut=tokenizer.tokenize(example.question) + ['[SEP]'] + tokenizer.tokenize(choice)
            toPut.append('[SEP]')
            #Delect from end
            while len(tokens)+len(toPut)+1>args.max_seq_length:
                tokens.pop()
                tok_is_sentence_end[-1] = len(tokens)
                if tok_is_sentence_end[-1] == tok_is_sentence_begin[-1]:
                    tok_is_sentence_begin.pop()
                    tok_is_sentence_end.pop()

            tokens.append('[SEP]')
            tok_is_question_begin = len(tokens)

            tok_is_question_end = len(tokens)  + len(tokenizer.tokenize(example.question))
            tok_is_choice_begin = tok_is_question_end+1
            
            mtokens = tokens+toPut
            
            tok_is_choice_end = len(mtokens)-1
            
            if choice == example.answer: 
                label = 1
            else: label = 0

            tok_is_choice_end = len(mtokens) - 1     
            
            #Delete From begin to fit max_seq_length
#            while len(tokens) > args.max_seq_length:
#                bp = tok_is_sentence_end[0]-1
#                tokens = tokens[tok_is_sentence_end[0]:]
#                tokens = ['[CLS]']+tokens #recover [CLS]
#                tok_is_sentence_begin = tok_is_sentence_begin[1:]
#                tok_is_sentence_end = tok_is_sentence_end[1:]
#                for i in range(len(tok_is_sentence_begin)):
#                    tok_is_sentence_begin[i]-=bp
#                    tok_is_sentence_end[i]-=bp
#                
##                tok_is_sentence_begin[0]-=1 
#                
#                tok_is_question_begin-=bp
#                tok_is_question_end-=bp
#                tok_is_choice_begin-=bp
#                tok_is_choice_end-=bp
            


            input_ids = tokenizer.convert_tokens_to_ids(mtokens)
            segment_ids = [0]*tok_is_question_begin + [1]*(len(input_ids)-tok_is_question_begin)
            input_mask = [1]*len(input_ids)
            while len(input_ids) < args.max_seq_length:
                input_ids.append(0)
                segment_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == len(segment_ids)
            assert len(segment_ids) == len(input_mask)
            assert len(input_ids) <= args.max_seq_length
            
            if len(mtokens) > args.max_seq_length:
                logging.info("MAX LEN EXCEEED:{}".format(len(input_ids)))
#                print(example.talk)
            graph = Graph()

#            edge_index_now = -1
            
##            print(len(tokens),tok_is_choice_end,tok_is_question_begin)
#
#    TOKEN_TO_SENTENCE = 0
#
#
#    SENTENCE_TO_TOKEN = 1
#    SENTENCE_TO_QA = 2
#    QA_TO_SENTENC = 3
#    
#    QA_TO_CLS = 4
#    SENTENCE_TO_CLS = 5
#    SENTENCE_TO_NEXT = 6
#    SENTENCE_TO_BEFORE = 7
#    
#    A_TO_B = 8
#    B_TO_A = 9
#    
#    QUESTION_TOKEN_TO_SENTENCE = 10
#    CHOICE_TOKEN_TO_SENTENCE = 11
#    QA_TO_A = 12
#    QA_TO_B = 13
#    
#    
#    
            for node_i in range(len(mtokens)):
                graph.add_node(node_i)
            ALL_SEN = []
            AB = [[],[]]
#            if len(names) > 2: print(name)
#            continue 
            
            for i,(tok_begin,tok_end) in enumerate(zip(tok_is_sentence_begin,tok_is_sentence_end)):
                AB[names.index(tokens[tok_begin])%2].append(tok_begin)
#                if len(ALL_SEN) != 0:
#                    graph.add_edge(ALL_SEN[-1][0],tok_begin,EdgeType.SENTENCE_TO_NEXT)
#                    graph.add_edge(tok_begin,ALL_SEN[-1][0],EdgeType.SENTENCE_TO_BEFORE)
                ALL_SEN.append((tok_begin,tok_end))

                for index in range(tok_begin+1,tok_end):
                    graph.add_edge(index,tok_begin,EdgeType.TOKEN_TO_SENTENCE)
                    graph.add_edge(tok_begin,index,EdgeType.SENTENCE_TO_TOKEN)
#                graph.add_edge(tok_begin,tok_is_question_begin,EdgeType.SENTENCE_TO_QA)
#                graph.add_edge(tok_is_question_begin,tok_begin,EdgeType.QA_TO_SENTENCE)
#                graph.add_edge(tok_begin,0,EdgeType.SENTENCE_TO_CLS)
                
            for tok_index in range(tok_is_question_begin,tok_is_question_end):
                graph.add_edge(tok_index,tok_is_question_begin,EdgeType.QUESTION_TOKEN_TO_SENTENCE)
            for tok_index in range(tok_is_choice_begin,tok_is_choice_end):
                graph.add_edge(tok_index,tok_is_choice_begin,EdgeType.CHOICE_TOKEN_TO_SENTENCE)
                
            graph.add_edge(tok_is_question_begin,0,EdgeType.QUESTION_TO_CLS)
            graph.add_edge(tok_is_choice_begin,0,EdgeType.CHOICE_TO_CLS)
            
            
            A = AB[0]
            B = AB[1]
            
            for anode in A:
                NEXT_B = 100000
                BEFORE_B = -1
                for bnode in B:
                    if bnode < anode and bnode>BEFORE_B:
                        BEFORE_B = bnode
                    if bnode > anode and bnode<NEXT_B:
                        NEXT_B = bnode
#                    graph.add_edge(anode,bnode,EdgeType.A_TO_B)
#                    graph.add_edge(anode,bnode,EdgeType.B_TO_A)
                if NEXT_B != 100000:
                    graph.add_edge(anode,NEXT_B,EdgeType.A_TO_NB)
                    graph.add-edge(NEXT_B,anode,EdgeType.B_TO_BA)
                if BEFORE_B != -1:
                    graph.add_edge(anode,BEFORE_B,EdgeType.A_TO_BB)
                    graph.add_edge(BEFORE_B,anode,EdgeType.B_TO_NA)
                
                for bonde in B:
                    if bnode != NEXT_B and bnode != BEFORE_B:
                    graph.add_edge(anode,bnode,EdgeType.A_TO_B)
                    graph.add_edge(anode,bnode,EdgeType.B_TO_A)

            for anode in A:
                graph.add_edge(tok_is_question_begin,anode,EdgeType.QUESTION_TO_A)
                graph.add_edge(anode,tok_is_question_begin,EdgeType.A_TO_QUESTION)
                graph.add_edge(tok_is_choice_begin,anode,EdgeType.CHOICE_TO_A)
                graph.add_edge(anode,tok_is_choice_begin,EdgeType.A_TO_CHOICE)
                graph.add_edge(anode,0,EdgeType.A_TO_CLS)
                
            for bnode in B:
                graph.add_edge(tok_is_question_begin,bnode,EdgeType.QUESTION_TO_B)
                graph.add_edge(bnode,tok_is_question_begin,EdgeType.B_TO_QUESTION)
                graph.add_edge(tok_is_choice_begin,bnode,EdgeType.CHOICE_TO_B)
                graph.add_edge(bnode,tok_is_choice_begin,EdgeType.B_TO_CHOICE)
                graph.add_edge(bnode,0,EdgeType.B_TO_CLS)
            
            
            ALL_SEN.append([tok_is_question_begin,tok_is_choice_end])
            

            feature_now.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    all_sen=ALL_SEN,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    graph=graph,
                    label=label))

        random.shuffle(feature_now)
        features.append(feature_now)

    logging.info("  Saving features into cached file {}".format(cached_path))
    with open(cached_path, "wb") as writer:
        random.shuffle(features)
        pickle.dump(features, writer)

    return cached_path


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def run_convert_examples_to_features(args, examples, tokenizer, is_training, cached_path):
    pool = []
    p = multiprocessing.Pool(args.num_threads)
    for i in range(args.num_threads):
        start_index = len(examples) // args.num_threads * i
        end_index = len(examples) // args.num_threads * (i + 1)
        if i == args.num_threads - 1:
            end_index = len(examples)
        pool.append(p.apply_async(convert_examples_to_features, args=(
            args, examples[start_index: end_index], tokenizer, is_training, cached_path + ".part" + str(i))))
    p.close()
    p.join()

    features = []
    for i, thread in enumerate(pool):
        cached_path_tmp = thread.get()
        logging.info("Reading thread {} output from {}".format(i, cached_path_tmp))
        with open(cached_path_tmp, "rb") as reader:
            features_tmp = pickle.load(reader)
        os.remove(cached_path_tmp)
        features += features_tmp

    logging.info("  Saving features from into cached file {0}".format(cached_path))

    with open(cached_path, "wb") as writer:
        pickle.dump(features, writer)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pattern", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--vocab_file", default=None, type=str, required=True)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--include_unknowns", default=0.03, type=float)
    parser.add_argument("--max_position", default=50, type=int)
    parser.add_argument("--num_threads", default=16, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_num', type=int, default=-1)
    parser.add_argument('--end_num', type=int, default=-1)
    parser.add_argument('--generate_count', type=int, default=100)
    parser.add_argument('--hard_mode',type=bool,default=False)
    parser.add_argument('--DataName',type=str,default="SST")
    

    args = parser.parse_args()

    #tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    print("Vocab SIze!",tokenizer.vocab_size)
    

    prefix = "cached_{0}_{1}_{2}_{3}".format(str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length),args.DataName)

    prefix = os.path.join(args.output_dir, prefix)
    os.makedirs(prefix, exist_ok=True)

    
    for input_path in glob(args.input_pattern):

        if args.start_num >= 0 and args.end_num >= 0:
            continue
        cached_path = os.path.join(prefix, os.path.split(input_path)[1] + ".pkl")
        if os.path.exists(cached_path):
            logging.info("{} already exists.".format(cached_path))
            continue
        is_training = True #Always set to true
        logging.info("train:{}".format(is_training))
#        print(input_path)
        examples = []
        with open(args.input_pattern) as f:
            texts = f.read()
            texts= json.loads(texts)
            for text in texts:
#                print(text)
                talk = text[0]
                for flo in text[1]:
                    question = flo['question']
                    choice =  flo['choice']
                    choice.sort(key=len) #sort by length
                    answer = flo['answer']
        
                    unique_id = text[2]
                    examples.append(DrExample(talk,question,choice,answer,unique_id))
            run_convert_examples_to_features(args=args,
                                             examples=examples,
                                             tokenizer=tokenizer,
                                             is_training=is_training,
                                             cached_path=cached_path)


if __name__ == "__main__":
    main()
