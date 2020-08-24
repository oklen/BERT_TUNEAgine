import torch
import torch.nn as nn

#from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from modules.graph_encoderDr3ForAB import NodeType, NodePosition, EdgeType, Encoder,GraphEncoder
from transformers import AutoTokenizer, AutoModelWithLMHead,AutoModel,AlbertModel,AlbertConfig,RobertaModel,RobertaConfig
#  elgeish/cs224n-squad2.0-albert-large-v2
#  albert-large-v2

class NqModel(nn.Module):
    def __init__(self, bert_config, my_config):
        super(NqModel, self).__init__()
        #albert_base_configuration = AlbertConfig(vocab_size=30000,hidden_size=768,num_attention_heads=12,intermediate_size=3072,
        #                                        attention_probs_dropout_prob=0)
        self.my_mask = None
        self.bert =  RobertaModel.from_pretrained("roberta-large-mnli")
        #self.bert = RobertaModel.from_pretrained("roberta-base")
        my_config.hidden_size = self.bert.config.hidden_size

        self.right = 0
        self.all = 0
        #self.bert =  AlbertModel(albert_base_configuration)
        
        #self.bert2 = BertModel(bert_config)

        #self.bert = BertModel(BertConfig())
        
        
        #self.bert =  RobertaModel(RobertaConfig(max_position_embeddings=514,vocab_size=50265))

        #print(my_config,bert_config)
        self.tok_dense = nn.Linear(my_config.hidden_size*2, my_config.hidden_size*2)
#        self.para_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#        self.doc_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        self.dropout = nn.Dropout(my_config.hidden_dropout_prob)

        self.tok_outputs = nn.Linear(my_config.hidden_size*2, 1) # tune to avoid fell into bad places
#        config.max_token_len, config.max_token_relative
#        self.para_outputs = nn.Linear(self.config.hidden_size, 1)
#        self.answer_type_outputs = nn.Linear(self.config.hidden_size, 2)
        
#        self.tok_to_label = nn.Linear(my_config.max_token_len,2)
#        self.par_to_label = nn.Linear(my_config.max_paragraph_len,2)

        #self.encoder = Encoder(my_config)
        self.encoder = Encoder(my_config)
#        self.encoder2 = Encoder(my_config)
        self.my_config = my_config
#        self.my_mask = 

        self.ACC = 0
        self.ALL = 0

        #self.apply(self.init_bert_weights)

    def forward(self, input_idss, attention_masks, token_type_idss, st_masks, edgess, labels):

#model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask, batch.st_index,
#                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),
#                                 batch.start_positions, batch.end_positions, batch.answer_type)
        loss = []
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        tok_logits = []
        res_labels = []
#        print(input_idss.shape)

        edges_srcs, edges_tgts, edges_types, edges_poss = edgess
        for input_ids, attention_mask, token_type_ids, st_mask, label,edges_src, edges_tgt, edges_type, edges_pos in zip(input_idss, attention_masks, token_type_idss, st_masks, labels,edges_srcs, edges_tgts, edges_types, edges_poss):
#            print(input_ids.shape)
#            print(attention_mask.shape)

            
            sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids)
    
            #sequence_output2, _ = self.bert2(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            #print(type(sequence_output),sequence_output.shape)
            #print(type(sequence_output2),sequence_output2.shape)
            #exit(0)
            #print("ALBERT DONE!")
    #        print("BEFORE GRAPH:",sequence_output.shape)
            graph_output = self.encoder(sequence_output, st_mask, (edges_src, edges_tgt, edges_type, edges_pos), output_all_encoded_layers=False)
#            graph_output = self.encoder2(graph_output, st_mask, (edges_src, edges_tgt, edges_type, edges_pos), output_all_encoded_layers=False)[0]
#    
#            q_pos = edges_type.eq(EdgeType.QA_TO_SENTENCE).nonzero().view(-1).tolist()[0]
#            q_pos = edges_src[q_pos]
            #print("GRAPH DONE!")
            # token
            #tok_output = torch.tanh(self.tok_dense(graph_output[:, :NodePosition.MAX_TOKEN, :]))
            #tok_output = torch.tanh(self.tok_dense(_))
    
    
            #tok_logits = tok_logits.view(tok_logits.size(0),self.my_config.max_token_len)
            #tok_logits = self.tok_to_label(tok_logits)
    #        print(graph_output.shape,self.config.hidden_size)
#            print(graph_output[:,0])
            x = torch.cat((graph_output[:,0],sequence_output[:,0]),-1)
            x = self.dropout(x)
            tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
            for index,lab in enumerate(label):
                if lab == 1:
                    res_labels.append(index)
        

        # token
        tok_logits = torch.stack(tok_logits)
        res_labels = torch.tensor(res_labels,dtype=torch.long).to(tok_logits.device)
#        print(label)
#        print(tok_logits,res_labels)
        for index,score in enumerate(tok_logits):
            self.ALL+=1
            if torch.argmax(score) == res_labels:
                self.ACC+=1
#        print(self.ALL,self.ACC)
#        print("ACC:{}".format(self.ACC/self.ALL))
#        print(tok_logits.shape,res_labels.shape)
#        print(tok_logits)
#        print(res_labels)
        tok_label_loss = loss_fct(tok_logits, res_labels)
    
        loss.append(tok_label_loss)

        # paragraph
        
#        para_logits = para_logits.view(para_logits.size(0),self.my_config.max_paragraph_len)
#        para_logits = self.par_to_label(torch.tanh(para_logits))
#
#
#        para_loss = loss_fct(para_logits, label)
#        #loss.append(para_loss)
#
#        # document
#        answer_type_loss = loss_fct(answer_type_logits, label)
        #loss.append(answer_type_loss)
        if labels != None:
            return torch.sum(torch.stack(loss))
        else:
            return tok_logits