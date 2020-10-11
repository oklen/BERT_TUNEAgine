import torch
import torch.nn as nn

#from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from modules.graph_encoderABDUG4LS3V import NodeType, NodePosition, EdgeType, Encoder,GraphEncoder
from transformers import AutoTokenizer, AutoModelWithLMHead,AutoModel,AlbertModel,AlbertConfig,RobertaModel,RobertaConfig
import math
#  elgeish/cs224n-squad2.0-albert-large-v2
#  albert-large-v2
#

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class NqModel(nn.Module):
    def __init__(self, my_config,args):
        super(NqModel, self).__init__()
        #albert_base_configuration = AlbertConfig(vocab_size=30000,hidden_size=768,num_attention_heads=12,intermediate_size=3072,
        #                                        attention_probs_dropout_prob=0)
        self.my_mask = None
        self.args = args
        self.bert_config = AlbertConfig.from_pretrained("albert-xxlarge-v2")
        # self.bert_config = AlbertConfig.from_pretrained("albert-base-v2")
        
        # self.bert_config.hidden_dropout_prob = 0.1
        # self.bert_config.attention_probs_dropout_prob = 0.1
        
        # self.bert_config.gradient_checkpointing = True
        # self.bert_config.Extgradient_checkpointing = True
        # self.bert =  AlbertModel.from_pretrained("albert-base-v2",config = self.bert_config)
        self.bert =  AlbertModel.from_pretrained("albert-xxlarge-v2",config = self.bert_config)
#        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        my_config.hidden_size = self.bert.config.hidden_size
        my_config.num_attention_heads = self.bert.config.num_attention_heads

        self.right = 0
        self.all = 0
        #self.bert =  AlbertModel(albert_base_configuration)
        
        #self.bert2 = BertModel(bert_config)

        #self.bert = BertModel(BertConfig())
        
        
        #self.bert =  RobertaModel(RobertaConfig(max_position_embeddings=514,vocab_size=50265))

        #print(my_config,bert_config)
        # self.tok_dense = nn.Linear(my_config.hidden_size*6, my_config.hidden_size*6)
        # self.tok_dense = nn.Linear(my_config.hidden_size*2, my_config.hidden_size*2)

#        self.tok_dense2 = nn.Linear(my_config.hidden_size, my_config.hidden_size)
#        self.para_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#        self.doc_dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        self.dropout = nn.Dropout(my_config.hidden_dropout_prob)

        self.tok_outputs = nn.Linear(my_config.hidden_size*2, 1) # tune to avoid fell into bad places
        
#        self.tok_outputs2 = nn.Linear(my_config.hidden_size, 1)
#        config.max_token_len, config.max_token_relative
#        self.para_outputs = nn.Linear(self.config.hidden_size, 1)
#        self.answer_type_outputs = nn.Linear(self.config.hidden_size, 2)
        
#        self.tok_to_label = nn.Linear(my_config.max_token_len,2)
#        self.par_to_label = nn.Linear(my_config.max_paragraph_len,2)

        #self.encoder = Encoder(my_config)
        self.cls_to_space = nn.Linear(my_config.hidden_size,my_config.hidden_size)
        self.Dres_to_space = nn.Linear(my_config.hidden_size,my_config.hidden_size)
        
        self.encoder = Encoder(my_config)
#        self.encoder2 = Encoder(my_config)
        
        self.my_config = my_config
        if torch.__version__ == '1.1.0':
            self.output_act = gelu
        else:
            self.output_act = torch.nn.functional.gelu
        # self.output_act = torch.nn.functional.gelu
        
#        self.my_mask = 

        self.ACC = 0
        self.ALL = 0
        
        self.model_choice = None
        self.ground_answer = None
        
        self.ErrId = []
        
        #self.apply(self.init_bert_weights)

    def forward(self, input_idss, attention_masks, token_type_idss, st_masks, edgess, labels,all_sens,input_embs=None):

#model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask, batch.st_index,
#                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),
#                                 batch.start_positions, batch.end_positions, batch.answer_type)
        loss = []
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        tok_logits = []
        res_labels = []
        
#        print(input_idss.shape)

        edges_srcs, edges_tgts, edges_types, edges_poss = edgess
        # outer_i = 0
        # outer_j = 0
        for input_ids, attention_mask, token_type_ids, st_mask, label,edges_src, edges_tgt, edges_type, edges_pos,all_sen in zip(input_idss, attention_masks, token_type_idss, st_masks, labels,edges_srcs, edges_tgts, edges_types, edges_poss,all_sens):
            if self.args.run_og:
                if input_embs==None:
                    sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids)
                else:
                    sequence_output,_ = self.bert(None,  attention_mask,token_type_ids,inputs_embeds=input_embs)
                if getattr(self.bert_config, "gradient_checkpointingNot", False):
                    def create_custom_forward(module):
                        def custom_forward(*inputs,output_all_encoded_layers=False):
                            x = self.dropout(module(*inputs,output_all_encoded_layers=False))
#                            return self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1)
                            return self.tok_outputs(x).squeeze(-1)
                        return custom_forward

#                    sequence_output,_ = torch.utils.checkpoint.checkpoint(self.bert,input_ids,  attention_mask,token_type_ids)
                    tok_logits.append(torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.encoder),
                    sequence_output,
                    st_mask,
                    edges_src, edges_tgt, edges_type, edges_pos,))
                else:

                    graph_output = self.encoder(sequence_output, st_mask, edges_src, edges_tgt, edges_type, edges_pos, all_sen,output_all_encoded_layers=False)
                    # x = self.dropout(graph_output)
                    # x = self.dropout(graph_output)
                    # print(graph_output.shape)
                
#                    x = self.dropout(sequence_output[:,0])
#                    print(x)
#                    x = self.dropout(graph_output)
                    Output = graph_output.view(graph_output.size(0),-1,self.bert_config.hidden_size)
                    # print("shape:",Output.shape,_.shape)
                    # print(Output.shape,sequence_output[:,0].unsqueeze(1).shape)
                    # output_scores_t = torch.bmm(Output,_.unsqueeze(1).transpose(-1,-2))
                    # output_scores_t = torch.bmm(self.Dres_to_space(Output),self.cls_to_space(_).unsqueeze(1).transpose(-1,-2))
                    # output_scores =  torch.softmax(output_scores_t.transpose(-1,-2), -1).transpose(-1,-2)
                    # tok_logits.append(self.tok_outputs((output_scores*Output).view(graph_output.shape)).squeeze(-1))
                    tok_logits.append(self.tok_outputs(graph_output).squeeze(-1))
                    # tok_logits.append(graph_output.squeeze(-1))

            else:
                input_ids = input_ids.to('cuda:0')
                attention_mask = attention_mask.to('cuda:0')
                token_type_ids = token_type_ids.to('cuda:0')
                sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids)
        
                #sequence_output2, _ = self.bert2(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
                #print(type(sequence_output),sequence_output.shape)
                #print(type(sequence_output2),sequence_output2.shape)
                #exit(0)
                #print("ALBERT DONE!")
        #        print("BEFORE GRAPH:",sequence_output.shape)
                sequence_output = sequence_output.to('cuda:1')
                st_mask = st_mask.to('cuda:1')
                edges_src = edges_src.to('cuda:1')
                edges_tgt = edges_tgt.to('cuda:1')
                edges_type = edges_type.to('cuda:1')
                edges_pos = edges_pos.to('cuda:1')
#                if getattr(self.bert_config, "gradient_checkpointing", False):
#                    graph_output = torch.utils.checkpoint.checkpoint(
#                            self.encoder,
#                            sequence_output, st_mask, edges_src, edges_tgt, edges_type, edges_pos,
#                            )
#                else:
                graph_output = self.encoder(sequence_output, st_mask, edges_src, edges_tgt, edges_type, edges_pos, output_all_encoded_layers=False)
                
                x = self.dropout(graph_output.to('cuda:0'))
                tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
            # outer_i+=1
#            graph_output = self.encoder2(graph_output, st_mask, (edges_src, edges_tgt, edges_type, edges_pos), output_all_encoded_layers=False)
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
#            x = torch.cat((graph_output[:,0],sequence_output[:,0]),-1)
#            x = graph_output
#
    
#            x = self.dropout(graph_output)
#            tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
#            x = self.dropout(sequence_output)
#            tok_logits.append(self.tok_outputs2(self.dropout(torch.tanh(self.tok_dense2(x[:,0])))).squeeze(-1))
#            x = self.dropout(graph_output)
#            tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
            
#            tok_logits.append(self.tok_outputs(self.dropout(x)).squeeze(-1))
            
            
#            tok_logits.append(self.tok_outputs2(self.dropout(torch.tanh(self.tok_dense2(x)))).squeeze(-1))

            for index,lab in enumerate(label):
                if lab == 1:
                    res_labels.append(index)
        

        # token
#        print(tok_logits)
        tok_logits = torch.stack(tok_logits)
        res_labels = torch.tensor(res_labels,dtype=torch.long).to(tok_logits.device)
#        print(label)
#        print(tok_logits,res_labels)
#        print(res_labels)
        
        for index,score in enumerate(tok_logits):
            self.ALL+=1
            self.model_choice = torch.argmax(score)
            self.ground_answer = res_labels
            if torch.argmax(score) == res_labels:
                self.ACC+=1

#        print(self.ALL,self.ACC)
#        print("ACC:{}".format(self.ACC/self.ALL))
#        print(tok_logits.shape,res_labels.shape)
#        print(tok_logits)
#        print(res_labels)
        tok_label_loss = loss_fct(tok_logits, res_labels)
        loss.append(tok_label_loss)

        if labels is not None:
            return torch.sum(torch.stack(loss))
        else:
            return tok_logits