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
import gc

sys.path.append(os.getcwd())
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
#from torch.optim import AdamW


from tqdm import tqdm, trange
from glob import glob


from modules.graph_encoderABDUG6LS import Config, EdgeType, NodeType, EdgePosition
from generate_exampleDreamAB import InputFeatures

from src_nq.modelDRAB3LS import NqModel
from src_nq.datasetRov3 import NqDataset
from src_nq.optimization import WarmupLinearSchedule,WarmupConstantSchedule,AdamW


WEIGHTS_NAME = "pytorch_modelAB.bin"
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
                                 ['input_ids', 'input_mask', 'segment_ids', 'st_mask',
                                  'edges_src', 'edges_tgt', 'edges_type', 'edges_pos','label','all_sen'])


def batcher(device, is_training=False):
    def batcher_dev(mul_features):
        input_idss = []
        input_masks = []
        segment_idss = []
        st_masks = []
        edges_srcs = []
        edges_tgts = []
        edges_types = []
        edges_poss = []
        sen_bes = []
        labels  = []
        for i,features in  enumerate(mul_features):
#            unique_ids = [f.unique_id for f in features]
            input_ids = [f.input_ids for f in features]
            input_mask = [f.input_mask for f in features]
            segment_ids = [f.segment_ids for f in features]
            st_mask = [f.graph.st_mask for f in features]
            sen_be = [f.all_sen for f in features]
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
            
            for sen in sen_be:
                while len(sen) < 64: 
                    sen.append((-1,-1))
#            unique_idss.append(unique_ids)
            input_idss.append(input_ids)
            input_masks.append(input_mask)
            segment_idss.append(segment_ids)
            st_masks.append(st_mask)
            edges_srcs.append(edges_src)
            edges_tgts.append(edges_tgt)
            edges_types.append(edges_type)
            edges_poss.append(edges_pos)
            labels.append(label)
            sen_bes.append(sen_be)
            
#        unique_idss = torch.tensor(unique_idss,dtype=torch.long)
        input_idss = torch.tensor(input_idss,dtype=torch.long)
        input_masks = torch.tensor(input_masks,dtype=torch.long)
        segment_idss = torch.tensor(segment_idss,dtype=torch.long)
        st_masks = torch.tensor(st_masks,dtype=torch.long)
        edges_srcs = torch.tensor(edges_srcs,dtype=torch.long)
        edges_tgts = torch.tensor(edges_tgts,dtype=torch.long)
        edges_types = torch.tensor(edges_types,dtype=torch.long)
        edges_poss = torch.tensor(edges_poss,dtype=torch.long)
        labels  = torch.tensor(labels,dtype=torch.long)
        sen_bes = torch.tensor(sen_bes,dtype=torch.long)
        
#        for i,ed in enumerate(edges_srcs):
#            ed+=st_masks.size(2)*i
#        for i,ed in enumerate(edges_tgts):
#            ed+=st_masks.size(2)*i
        
#        print(edges_tgts)
        if run_og:
            return NqBatch(input_ids=input_idss.to(device),
                           input_mask=input_masks.to(device),
                           segment_ids=segment_idss.to(device),
                           st_mask=st_masks.to(device),
                           edges_src=edges_srcs.to(device),
                           edges_tgt=edges_tgts.to(device),
                           edges_type=edges_types.to(device),
                           edges_pos=edges_poss.to(device),
                           label=labels.to(device),
                           all_sen=sen_bes.to(device))
        else:
            return NqBatch(
                   input_ids=input_idss,
                   input_mask=input_masks,
                   segment_ids=segment_idss,
                   st_mask=st_masks,
                   edges_src=edges_srcs,
                   edges_tgt=edges_tgts,
                   edges_type=edges_types,
                   edges_pos=edges_poss,
                   label=labels,
                   all_sen=sen_bes)
    return batcher_dev


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "tok_start_logits", "tok_end_logits", "tok_ref_indexes",
                                    "para_logits", "para_ref_indexes", "doc_logits"])



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
    parser.add_argument("--warmup_steps", default=100, type=int)
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
    parser.add_argument('--run_og',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    

    # parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    # parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--adv-lr', type=float, default=5e-5)
    parser.add_argument('--adv-steps', type=int, default=3, help="should be at least 1")
    parser.add_argument('--adv-init-mag', type=float, default=1e-5)
    parser.add_argument('--norm-type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv-max-norm', type=float, default=0, help="set to 0 to be unlimited")
    # parser.add_argument('--gpu', type=str, default="0")
    # parser.add_argument('--expname', type=str, default="default")
    # parser.add_argument('--comet', default=False, action="store_true")
    # parser.add_argument('--comet_key', default="", type=str)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)
    
    args = parser.parse_args()
    #print(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
#        torch.cuda.set_device(args.local_rank)
#        print(args.local_rank)
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
#    my_config.forward_edges = [EdgeType.TOKEN_TO_SENTENCE,
#                               EdgeType.SENTENCE_TO_PARAGRAPH,
#                               EdgeType.PARAGRAPH_TO_DOCUMENT]
    #print(my_config)
    if args.do_train:


        model = NqModel( my_config=my_config,args=args)
        #model_dict = model.state_dict()
        #pretrained_model_dict = torch.load(pretrained_model_file, map_location=lambda storage, loc: storage)
        #pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict.keys()}
        #model_dict.update(pretrained_model_dict)
        #model.load_state_dict(model_dict)
#    else:
#        pretrained_config_file = os.path.join(args.model_dir, CONFIG_NAME)
##        bert_config = BertConfig(pretrained_config_file)
#        model = NqModel( my_config=my_config)
#        pretrained_model_file = os.path.join(args.model_dir, WEIGHTS_NAME)
#        model.load_state_dict(torch.load(pretrained_model_file))

    if args.fp16:
        model.half()
    global run_og 
    run_og = args.run_og
    if args.run_og:
        if n_gpu:
            model.cuda()
        if args.local_rank != -1:
#            model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        model.bert.to("cuda:0")
        model.encoder.to("cuda:1")
        model.tok_outputs.to("cuda:0")
        model.tok_dense.to("cuda:0")
        model.dropout.to("cuda:0")

    
#    if args.local_rank != -1:
#        try:
#            from apex.parallel import DistributedDataParallel as DDP
#        except ImportError:
#            raise ImportError(
#                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
#
#        model = DDP(model)
#    elif n_gpu > 1:
#        model = torch.nn.DataParallel(model)

    num_train_features = None
    num_train_optimization_steps = None
    
    #train_dataset = None
    #train_features = None
    
#    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
#    model.load_state_dict(torch.load(output_model_file))
    
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
#        optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate,momentum=0.9)

        # scheduler = WarmupLinearSchedule(optimizer,
        #                              warmup_steps=int(args.warmup_proportion * num_train_optimization_steps)
        #                              if args.warmup_proportion > 0 else args.warmup_steps,
        #                              t_total=num_train_optimization_steps)
        scheduler = WarmupConstantSchedule(optimizer,
                                     warmup_steps=int(args.warmup_proportion * num_train_optimization_steps)
                                     if args.warmup_proportion > 0 else args.warmup_steps)

    global_step = 0
    last_acc = 85.0
    
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", num_train_features)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        tr_loss, report_loss = 0.0, 0.0
        nb_tr_examples = 0
        model.zero_grad()
        optimizer.zero_grad()
        
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
                # ============================ Code for adversarial training=============
                    # initialize delta

                    embeds_init = model.bert.embeddings.word_embeddings(batch.input_ids)
                    # embeds_init = model.embeddings.word_embeddings(batch[0])

                    if args.adv_init_mag > 0:
                        input_mask = batch.input_mask.to(embeds_init)
                        input_lengths = torch.sum(input_mask, 1)
                        # check the shape of the mask here..
        
                        if args.norm_type == "l2":
                            delta = torch.zeros_like(embeds_init).uniform_(-1,1) * input_mask.unsqueeze(2)
                            dims = input_lengths * embeds_init.size(-1)
                            mag = args.adv_init_mag / torch.sqrt(dims)
                            delta = (delta * mag.view(-1, 1, 1)).detach()
                        elif args.norm_type == "linf":
                            delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,
                                                                           args.adv_init_mag) * input_mask.unsqueeze(2)
        
                    else:
                        delta = torch.zeros_like(embeds_init)
        
                    # the main loop
                    # dp_masks = None
                    for astep in range(args.adv_steps):

                        delta.requires_grad_()
                        # inputs['inputs_embeds'] = delta + embeds_init
                        # inputs['dp_masks'] = dp_masks
        

                        loss = model(batch.input_ids,batch.input_mask, batch.segment_ids, batch.st_mask,
                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label,batch.all_sen,delta + embeds_init) # model outputs are always tuple in transformers (see doc)
                        # (1) backward
                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu parallel training
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
        
                        loss = loss / args.adv_steps
        
                        tr_loss += loss.item()
        
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
        
                        if astep == args.adv_steps - 1:
                            # further updates on delta
                            break
        
                        # (2) get gradient on delta
                        delta_grad = delta.grad.clone().detach()
        
                        # (3) update and clip
                        if args.norm_type == "l2":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                            if args.adv_max_norm > 0:
                                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                                exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                                reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                             + (1-exceed_mask)).view(-1, 1, 1)
                                delta = (delta * reweights).detach()
                        elif args.norm_type == "linf":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                            if args.adv_max_norm > 0:
                                delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                        else:
                            print("Norm type {} not specified.".format(args.norm_type))
                            exit()
        
                        if isinstance(model, torch.nn.DataParallel):
                            embeds_init = model.module.embeddings.word_embeddings(batch.input_ids)
                        else:
                            embeds_init = model.bert.embeddings.word_embeddings(batch.input_ids)

            # ============================ End (2) ==================
            
                    # loss = model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask,
                    #              (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label,batch.all_sen)
                    # if n_gpu > 1:
                    #     loss = loss.mean()  # mean() to average on multi-gpu.
                    # if args.gradient_accumulation_steps > 1:
                    #     loss = loss / args.gradient_accumulation_steps
                    # if args.local_rank != -1:
                    #     loss = loss + 0 * sum([x.sum() for x in model.parameters()])
                    # if args.fp16:
                    #     optimizer.backward(loss)
                    # else:
                    #     loss.backward()
                    
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
#                        gc.collect() 
#                        torch.cuda.empty_cache()
                        optimizer.step()
                        scheduler.step()
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
            
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_features)
            else:
                train_sampler = DistributedSampler(train_features)
            if args.local_rank == -1:
                train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size,
                                              collate_fn=batcher(device, is_training=True), num_workers=0)
            else:
                train_dataloader = DataLoader(train_features, sampler=train_sampler, batch_size=args.train_batch_size,
                                              collate_fn=batcher(device, is_training=True), num_workers=0,drop_last=True)
            
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
                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),batch.label,batch.all_sen)
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
