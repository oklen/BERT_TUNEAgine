import torch
import torch.nn as nn

#from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from transformers import BertModel, BertConfig
from modules.graph_encoder import NodeType, NodePosition, EdgeType, Encoder,GraphEncoder
from transformers import AutoTokenizer, AutoModelWithLMHead,AutoModel,AlbertModel,AlbertConfig,RobertaModel,RobertaConfig


class NqModel(BertPreTrainedModel):
    def __init__(self, bert_config, my_config):
        super(NqModel, self).__init__(bert_config)
        #albert_base_configuration = AlbertConfig(vocab_size=30000,hidden_size=768,num_attention_heads=12,intermediate_size=3072,
        #                                        attention_probs_dropout_prob=0)
        #self.bert =  AlbertModel.from_pretrained("albert-base-v2")
        self.bert = RobertaModel.from_pretrained("roberta-base")
        #self.bert =  AlbertModel(albert_base_configuration)
        
        #self.bert2 = BertModel(bert_config)

        #self.bert = BertModel(BertConfig())
        
        
        #self.bert =  RobertaModel(RobertaConfig(max_position_embeddings=514,vocab_size=50265))
        
        
        self.tok_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.para_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.doc_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.tok_outputs = nn.Linear(self.config.hidden_size, 2)
        self.para_outputs = nn.Linear(self.config.hidden_size, 1)
        self.answer_type_outputs = nn.Linear(self.config.hidden_size, 5)

        self.encoder = Encoder(my_config)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, st_mask, st_index, edges, start_positions=None,
                end_positions=None, answer_type=None):

#model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask, batch.st_index,
#                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),
#                                 batch.start_positions, batch.end_positions, batch.answer_type)
        sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids)
        
        #sequence_output2, _ = self.bert2(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #print(type(sequence_output),sequence_output.shape)
        #print(type(sequence_output2),sequence_output2.shape)
        #exit(0)
        #print("ALBERT DONE!")
        graph_output = self.encoder(sequence_output, st_mask, edges, output_all_encoded_layers=False)[0]
        
        # token
        tok_output = torch.tanh(self.tok_dense(graph_output[:, :NodePosition.MAX_TOKEN, :]))
        #tok_output = torch.tanh(self.tok_dense(sequence_output[:, :NodePosition.MAX_TOKEN, :]))
        
        tok_logits = self.tok_outputs(tok_output)
        tok_start_logits, tok_end_logits = tok_logits.split(1, dim=-1)
        tok_start_logits = tok_start_logits.squeeze(-1)
        tok_end_logits = tok_end_logits.squeeze(-1)
        tok_zero_mask = st_mask[:, :NodePosition.MAX_TOKEN].eq(0)
        tok_start_logits.masked_fill_(tok_zero_mask, -10000)
        tok_end_logits.masked_fill_(tok_zero_mask, -10000)

        # paragraph
        para_start = NodePosition.MAX_TOKEN + NodePosition.MAX_SENTENCE
        para_end = para_start + NodePosition.MAX_PARAGRAPH
        para_output = torch.tanh(self.para_dense(graph_output[:, para_start:para_end, :]))
        para_logits = self.para_outputs(para_output).squeeze(-1)
        para_zero_mask = st_mask[:, para_start:para_end].eq(0)
        para_logits.masked_fill_(para_zero_mask, -10000)

        # document
        doc_output = torch.tanh(self.doc_dense(graph_output[:, -1, :]))
        answer_type_logits = self.answer_type_outputs(doc_output)

        # training
        if answer_type is not None:
            loss = []
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

            # token
            tok_start_loss = loss_fct(tok_start_logits, start_positions[:, 0])
            tok_end_loss = loss_fct(tok_end_logits, end_positions[:, 0])
            loss.append(tok_start_loss + tok_end_loss)

            # paragraph
            para_loss = loss_fct(para_logits, start_positions[:, 2])
            loss.append(para_loss)

            # document
            answer_type_loss = loss_fct(answer_type_logits, answer_type)
            loss.append(answer_type_loss)

            return torch.sum(torch.stack(loss))
        else:
            return tok_start_logits, tok_end_logits, st_index[:, :NodePosition.MAX_TOKEN], \
                   para_logits, st_index[:, para_start:para_end], answer_type_logits
