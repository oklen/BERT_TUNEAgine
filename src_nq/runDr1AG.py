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

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

sys.path.append(os.getcwd())
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from glob import glob

from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

from modules.graph_encoder import Config, EdgeType, NodeType, EdgePosition
from generate_exampleDream import InputFeatures, DrExample, AnswerType

from src_nq.modelDR import NqModel
from src_nq.datasetRo import NqDataset
from src_nq.optimization import AdamW,WarmupLinearSchedule
from tools.nq_eval_tools import nq_evaluate

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

logger = logging.getLogger(__name__)
#def warmup_linear(x, warmup=0.002):
#    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
#        After `t_total`-th training step, learning rate is zero. """
#    if x < warmup:
#        return x/warmup
#    return max((x-1.)/(warmup-1.), 0)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
NqBatch = collections.namedtuple('NqBatch',
                                 ['unique_ids', 'input_ids', 'input_mask', 'segment_ids', 'st_mask',
                                  'edges_src', 'edges_tgt', 'edges_type', 'edges_pos','label'])


def batcher(device, is_training=False):
    def batcher_dev(mul_features):
        unique_idss = []
        input_idss = []
        input_masks = []
        segment_idss = []
        st_masks = []
        edges_srcs = []
        edges_tgts = []
        edges_types = []
        edges_poss = []
        labels  = []
        for i,features in  enumerate(mul_features):
            unique_ids = [f.unique_id for f in features]
            input_ids = [f.input_ids for f in features]
            input_mask = [f.input_mask for f in features]
            segment_ids = [f.segment_ids for f in features]
            st_mask = [f.graph.st_mask for f in features]
    #        st_index = torch.tensor([f.graph.st_index for f in features], dtype=torch.long)
            edges_src = [f.graph.edges_src for f in features]
            edges_tgt = [f.graph.edges_tgt for f in features]
            edges_type = [f.graph.edges_type for f in features]
            edges_pos = [f.graph.edges_pos for f in features]
            label = [f.label for f in features]
    
            
#            edges_src[i] += 512 * i
#            edges_tgt[i] += 512 * i
            for index,tmp in enumerate(edges_src):
                for i in range(len(tmp)):
                    tmp[i]+=index*len(st_mask[0])
                    
            for index,tmp in enumerate(edges_tgt):
                for i in range(len(tmp)):
                    tmp[i]+=index*len(st_mask[0])
                    
            edges_src = [x for y in edges_src for x in y]
            edges_tgt = [x for y in edges_tgt for x in y]
            edges_type = [x for y in edges_type for x in y]
            edges_pos = [x for y in edges_pos for x in y]
            
            unique_idss.append(unique_ids)
            input_idss.append(input_ids)
            input_masks.append(input_mask)
            segment_idss.append(segment_ids)
            st_masks.append(st_mask)
            edges_srcs.append(edges_src)
            edges_tgts.append(edges_tgt)
            edges_types.append(edges_type)
            edges_poss.append(edges_pos)
            labels.append(label)
            
        unique_idss = torch.tensor(unique_idss,dtype=torch.long)
        input_idss = torch.tensor(input_idss,dtype=torch.long)
        input_masks = torch.tensor(input_masks,dtype=torch.long)
        segment_idss = torch.tensor(segment_idss,dtype=torch.long)
        st_masks = torch.tensor(st_masks,dtype=torch.long)
        edges_srcs = torch.tensor(edges_srcs,dtype=torch.long)
        edges_tgts = torch.tensor(edges_tgts,dtype=torch.long)
        edges_types = torch.tensor(edges_types,dtype=torch.long)
        edges_poss = torch.tensor(edges_poss,dtype=torch.long)
        labels  = torch.tensor(labels,dtype=torch.long)
        
#        for i,ed in enumerate(edges_srcs):
#            ed+=st_masks.size(2)*i
#        for i,ed in enumerate(edges_tgts):
#            ed+=st_masks.size(2)*i
        
#        print(edges_tgts)
        return NqBatch(unique_ids=unique_idss,
                       input_ids=input_idss.to(device),
                       input_mask=input_masks.to(device),
                       segment_ids=segment_idss.to(device),
                       st_mask=st_masks.to(device),
#                       st_index=st_index,
                       edges_src=edges_srcs.to(device),
                       edges_tgt=edges_tgts.to(device),
                       edges_type=edges_types.to(device),
                       edges_pos=edges_poss.to(device),
                       label=labels.to(device))

    return batcher_dev


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "tok_start_logits", "tok_end_logits", "tok_ref_indexes",
                                    "para_logits", "para_ref_indexes", "doc_logits"])


def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length):
    """Write final predictions to the json file and log-odds of null if needed."""
    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_id].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_token", "end_token", "start_logit", "end_logit",
         "candidate_score", "doc_long_score", "doc_short_score"])

    all_predictions = []
    all_nbest_predictions = []

    for (example_index, example) in enumerate(all_examples):
        token_idx_to_candidate = {}
        for candidate_idx, candidate in enumerate(example.la_candidates):
            if not candidate["top_level"]:
                continue
            for token_idx in range(candidate["start_token"], candidate["end_token"]):
                token_idx_to_candidate[token_idx] = candidate_idx

        features = example_id_to_features[example.example_id]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            candidate_score = {}
            for (candidate_idx, score) in zip(result.para_ref_indexes, result.para_logits):
                if candidate_idx != -1:
                    candidate_score[candidate_idx] = score

            end_index = 0
            queue = []
            for start_index in range(len(result.tok_start_logits)):
                if start_index not in feature.token_to_orig_map or feature.token_to_orig_map[start_index] < 0:
                    continue
                start_token = feature.token_to_orig_map[start_index]

                if token_idx_to_candidate[start_token] not in candidate_score:
                    continue

                while len(queue) > 0 and start_index > queue[0]:
                    queue = queue[1:]

                while end_index < len(result.tok_start_logits):
                    if end_index < start_index or end_index not in feature.token_to_orig_map or \
                        feature.token_to_orig_map[end_index] < 0:
                        end_index += 1
                        continue

                    end_token = feature.token_to_orig_map[end_index]
                    if token_idx_to_candidate[start_token] != token_idx_to_candidate[end_token]:
                        break
                    length = end_token - start_token + 1
                    if length > max_answer_length:
                        break
                    while len(queue) > 0 and result.tok_end_logits[end_index] > result.tok_end_logits[queue[-1]]:
                        queue = queue[:-1]
                    queue.append(end_index)
                    end_index += 1

                end_token = feature.token_to_orig_map[queue[0]]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_token=start_token,
                        end_token=end_token,
                        start_logit=result.tok_start_logits[start_index] - result.tok_start_logits[0],
                        end_logit=result.tok_end_logits[queue[0]] - result.tok_end_logits[0],
                        candidate_score=candidate_score[token_idx_to_candidate[start_token]] -
                                        result.para_logits[0],
                        doc_long_score=max(result.doc_logits[1:]) - result.doc_logits[AnswerType.UNKNOWN],
                        doc_short_score=result.doc_logits[AnswerType.SHORT] - result.doc_logits[
                            AnswerType.UNKNOWN]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.candidate_score + x.doc_long_score, x.start_logit + x.end_logit),
            # key=lambda x: x.start_logit + x.end_logit,
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction",
            ["la_start_token", "la_end_token", "sa_start_token", "sa_end_token", "start_logit", "end_logit",
             "candidate_score", "doc_long_score", "doc_short_score"])

        seen = set()

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # TODO check valid

            start_token = pred.start_token
            end_token = pred.end_token

            candidate_idx = token_idx_to_candidate[start_token]
            if candidate_idx not in seen:
                seen.add(candidate_idx)
                nbest.append(
                    _NbestPrediction(
                        la_start_token=example.la_candidates[candidate_idx]["start_token"],
                        la_end_token=example.la_candidates[candidate_idx]["end_token"],
                        sa_start_token=start_token,
                        sa_end_token=end_token + 1,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        candidate_score=pred.candidate_score,
                        doc_long_score=pred.doc_long_score,
                        doc_short_score=pred.doc_short_score))

        prediction = collections.OrderedDict()
        prediction["example_id"] = example.example_id

        # prediction["long_answer_score"] = nbest[0].start_logit + nbest[0].end_logit
        prediction["long_answer_score"] = nbest[0].candidate_score + prelim_predictions[0].doc_long_score

        prediction["short_answers_score"] = nbest[0].start_logit + nbest[0].end_logit

        prediction["long_answer"] = {"start_token": nbest[0].la_start_token,
                                     "end_token": nbest[0].la_end_token,
                                     "start_byte": -1,
                                     "end_byte": -1}

        prediction["short_answers"] = [{"start_token": nbest[0].sa_start_token,
                                        "end_token": nbest[0].sa_end_token,
                                        "start_byte": -1,
                                        "end_byte": -1}]
        prediction["yes_no_answer"] = "none"

        all_predictions.append(prediction)

        nbest_predictions = []
        for i, entry in enumerate(nbest):
            prediction = collections.OrderedDict()

            prediction["example_id"] = example.example_id

            prediction["long_answer_score"] = nbest[i].candidate_score
            prediction["short_answers_score"] = nbest[i].start_logit + nbest[i].end_logit

            if nbest[i].la_start_token != -1:
                prediction["long_answer"] = {"start_token": nbest[i].la_start_token,
                                             "end_token": nbest[i].la_end_token,
                                             "start_byte": -1,
                                             "end_byte": -1}

                prediction["short_answers"] = [{"start_token": nbest[i].sa_start_token,
                                                "end_token": nbest[i].sa_end_token,
                                                "start_byte": -1,
                                                "end_byte": -1}]
                prediction["yes_no_answer"] = "none"
            nbest_predictions.append(prediction)
        all_nbest_predictions.append(nbest_predictions)
    return all_predictions, all_nbest_predictions


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def eval_model(args, device, model, data_pattern):
    all_predictions = []
    raw_results = []
    for data_path in glob(data_pattern):
        eval_dataset = NqDataset(args, data_path, is_training=False)
        eval_examples = eval_dataset.examples
        eval_features = eval_dataset.features

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        eval_sampler = SequentialSampler(eval_features)
        eval_dataloader = DataLoader(eval_features, sampler=eval_sampler, batch_size=args.predict_batch_size,
                                     collate_fn=batcher(device, is_training=False), num_workers=0)

        model.eval()
        part_results = []
        logger.info("Start evaluating")
        for batch in eval_dataloader:
            if len(part_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(part_results)))
            with torch.no_grad():
                batch_tok_start_logits, batch_tok_end_logits, batch_tok_ref_indexes, \
                batch_para_logits, batch_para_ref_indexes, batch_doc_logits = \
                    model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask,
                          (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label)
            for i, unique_id in enumerate(batch.unique_ids):
                tok_start_logits = batch_tok_start_logits[i].detach().cpu().tolist()
                tok_end_logits = batch_tok_end_logits[i].detach().cpu().tolist()
                tok_ref_indexes = batch_tok_ref_indexes[i].detach().cpu().tolist()
                para_logits = batch_para_logits[i].detach().cpu().tolist()
                para_ref_indexes = batch_para_ref_indexes[i].detach().cpu().tolist()
                doc_logits = batch_doc_logits[i].detach().cpu().tolist()
                unique_id = int(unique_id)
                part_results.append(RawResult(unique_id=unique_id,
                                              tok_start_logits=tok_start_logits,
                                              tok_end_logits=tok_end_logits,
                                              tok_ref_indexes=tok_ref_indexes,
                                              para_logits=para_logits,
                                              para_ref_indexes=para_ref_indexes,
                                              doc_logits=doc_logits))
        raw_results.extend(part_results)
        part_predictions, part_nbest_predictions = get_predictions(eval_examples, eval_features,
                                                                   part_results, args.n_best_size,
                                                                   args.max_answer_length)
        all_predictions += part_predictions

    import pickle
    raw_results_path = os.path.join(args.output_dir, "raw_results")
    logger.info("Writing Raw results to: {}".format(raw_results_path))
    with open(raw_results_path, "wb") as writer:
        pickle.dump(raw_results, writer)

    final_predictions = collections.OrderedDict()
    final_predictions["predictions"] = all_predictions
    predictions_path = os.path.join(args.output_dir, "tmp_predictions")
    logger.info("Writing predictions to: {}".format(predictions_path))
    with open(predictions_path, "w") as writer:
        writer.write(json.dumps(final_predictions, indent=4) + "\n")

    eval_results = nq_evaluate(gold_path=data_pattern, predictions_path=predictions_path, num_threads=16)
    logger.info("***** Eval results *****".format())
    for key in sorted(eval_results.keys()):
        logger.info("  %s = %s", key, str(eval_results[key] * 100))
    model.train()
    return eval_results["Long Answer F1"], eval_results["Short Answer F1"]


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_dir", default=None, type=str, required=True, help="")
    parser.add_argument("--my_config", default=None, type=str, required=True)
    parser.add_argument("--feature_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_pattern", default=None, type=str, help="training data path.")
    parser.add_argument("--valid_pattern", default=None, type=str, help="validation data path.")
    parser.add_argument("--test_pattern", default=None, type=str, help="test data path.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--report_steps", default=100, type=int, help="report steps when training.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--train_size',type=int,default=10000,help="Use how many train data")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--frame_name',type=str,default='elgeish/cs224n-squad2.0-albert-large-v2')
    parser.add_argument('--DataName',type=str,default="SST")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    args = parser.parse_args()
    #print(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_pattern:
            raise ValueError(
                "If `do_train` is True, then `train_pattern` must be specified.")

    if args.do_predict:
        if not args.test_pattern:
            raise ValueError(
                "If `do_predict` is True, then `test_pattern` must be specified.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare model
    my_config = Config(args.my_config)
    my_config.num_edge_types = sum(EdgePosition.max_edge_types)
    my_config.forward_edges = [EdgeType.TOKEN_TO_SENTENCE,
                               EdgeType.SENTENCE_TO_PARAGRAPH,
                               EdgeType.PARAGRAPH_TO_DOCUMENT]
    #print(my_config)
    if args.do_train:
        pretrained_config_file = os.path.join(args.model_dir, CONFIG_NAME)
        bert_config = BertConfig(pretrained_config_file)
        pretrained_model_file = os.path.join(args.model_dir, WEIGHTS_NAME)

        model = NqModel(bert_config=bert_config, my_config=my_config)
        #model_dict = model.state_dict()
        #pretrained_model_dict = torch.load(pretrained_model_file, map_location=lambda storage, loc: storage)
        #pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict.keys()}
        #model_dict.update(pretrained_model_dict)
        #model.load_state_dict(model_dict)
    else:
        pretrained_config_file = os.path.join(args.model_dir, CONFIG_NAME)
        bert_config = BertConfig(pretrained_config_file)
        model = NqModel(bert_config=bert_config, my_config=my_config)
        pretrained_model_file = os.path.join(args.model_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(pretrained_model_file))
    

    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    num_train_features = None
    num_train_optimization_steps = None
    #train_dataset = None
    #train_features = None
    
    #Load saved parameter here
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    
    if args.do_train:
        num_train_features = 0
        for data_path in glob(args.train_pattern):
            train_dataset = NqDataset(args, data_path, is_training=True)
            train_features = train_dataset.features
            num_train_features += len(train_dataset.features)
        print(num_train_features,args.train_batch_size,args.gradient_accumulation_steps)
        num_train_optimization_steps = int(
            (num_train_features / args.train_batch_size) / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
#    print(param_optimizer)
    #print([i for i,j in model.named_parameters()])

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer ]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        if args.warmup_steps > 0:
            args.warmup_proportion = min(args.warmup_proportion, args.warmup_steps / num_train_optimization_steps)
#        optimizer = BertAdam(optimizer_grouped_parameters,
#                             lr=args.learning_rate,
#                             warmup=args.warmup_proportion,
#                             t_total=num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=int(args.warmup_proportion * num_train_optimization_steps)
                                     if args.warmup_proportion > 0 else args.warmup_steps,
                                     t_total=num_train_optimization_steps)
        

    global_step = 0
    last_acc = 84.3704066634003
    
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", num_train_features)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        tr_loss, report_loss = 0.0, 0.0
        nb_tr_examples = 0
        model.zero_grad()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            logging.info("Loggin TEST!")
            for data_path in glob(args.train_pattern):
                #logging.info("Reading data from {}.".format(data_path))
                model.train()
                train_dataset = NqDataset(args, data_path, is_training=True)
                train_features = train_dataset.features
                #logging.info("Data Load Done!")
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_features)
                else:
                    train_sampler = DistributedSampler(train_features)
                train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size,
                                              collate_fn=batcher(device, is_training=True), num_workers=0)
                train_features = train_dataset.features
                logging.info("Data ready {} ".format(len(train_features)))

                for step, batch in enumerate(train_dataloader):
    
                    loss = model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask,
                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.local_rank != -1:
                        loss = loss + 0 * sum([x.sum() for x in model.parameters()])
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scheduler.step()
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    tr_loss += loss.item()
                    nb_tr_examples += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0 and (
                        global_step + 1) % args.report_steps == 0 and (
                        args.local_rank == -1 or torch.distributed.get_rank() == 0):
                        lr_this_step = get_lr(optimizer)
                        logging.info("Epoch={} iter={} lr={:.12f} train_ave_loss={:.6f} .".format(
                            # _, global_step, lr_this_step, tr_loss / nb_tr_examples))
                            _, global_step, lr_this_step, (tr_loss - report_loss) / args.report_steps))
                        report_loss = tr_loss
                        
            model.eval()
            model.zero_grad()
            model.ACC = model.ALL = 0
            train_dataset = NqDataset(args, "test.json", is_training=True)
            train_features = train_dataset.features
            #logging.info("Data Load Done!")
            
            train_sampler = RandomSampler(train_features)

            train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size,
                                          collate_fn=batcher(device, is_training=True), num_workers=0)
            
            train_features = train_dataset.features
            logging.info("Data ready {} ".format(len(train_features)))
            tgobal_step = 0
            ttr_loss = 0
            optimizer.zero_grad()
            logging.info("***** Running evalating *****")
            with torch.no_grad():
                for step, batch in enumerate(train_dataloader):
                    tgobal_step+=1
                    loss = model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask,
                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label)
                    ttr_loss+=loss.item()
            logging.info("ACC:{}% LOSS:{}".format(model.ACC/model.ALL*100,ttr_loss/tgobal_step))
            model.zero_grad()
            optimizer.zero_grad()

            if model.ACC/model.ALL*100>last_acc:
                logging.info("Save Model")
                last_acc = model.ACC/model.ALL*100
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                    
                

if __name__ == "__main__":
    main()
