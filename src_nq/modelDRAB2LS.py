import torch
import torch.nn as nn

#from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

from modules.graph_encoderABDUG4LS2VOUSPK import NodeType, NodePosition, EdgeType, Encoder,GraphEncoder
from transformers import AutoTokenizer, AutoModelWithLMHead,AutoModel,AlbertModel,AlbertConfig,RobertaModel,RobertaConfig

#  elgeish/cs224n-squad2.0-albert-large-v2
#  albert-large-v2

class NqModel(nn.Module):
    def __init__(self, my_config,args):
        super(NqModel, self).__init__()
        #albert_base_configuration = AlbertConfig(vocab_size=30000,hidden_size=768,num_attention_heads=12,intermediate_size=3072,
        #                                        attention_probs_dropout_prob=0)
        self.my_mask = None
        self.args = args
        #mfeb/albert-xxlarge-v2-squad2
        self.bert_config = AlbertConfig.from_pretrained("albert-xxlarge-v2")
        # self.bert_config.gradient_checkpointing = True
        # self.bert_config.Extgradient_checkpointing = True
        self.bert =  AlbertModel.from_pretrained("albert-xxlarge-v2",config = self.bert_config)
#        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        my_config.hidden_size = self.bert.config.hidden_size

        self.right = 0
        self.all = 0
        #self.bert =  AlbertModel(albert_base_configuration)
        
        #self.bert2 = BertModel(bert_config)

        #self.bert = BertModel(BertConfig())
        
        
        #self.bert =  RobertaModel(RobertaConfig(max_position_embeddings=514,vocab_size=50265))

        #print(my_config,bert_config)
#        self.tok_dense = nn.Linear(my_config.hidden_size, my_config.hidden_size)
        self.tok_dense = nn.Linear(my_config.hidden_size*2, my_config.hidden_size*2)

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
        self.encoder = Encoder(my_config)
#        self.encoder2 = Encoder(my_config)
        
        self.my_config = my_config
        
        self.model_choice = None
        self.ground_answer = None

        self.ACC = 0
        self.ALL = 0
        
        self.ErrId = []
        
        #self.apply(self.init_bert_weights)

    def forward(self, input_idss, attention_masks, token_type_idss, st_masks=None, edgess=None, labels=None,all_sens=None):

#model(batch.input_ids, batch.input_mask, batch.segment_ids, batch.st_mask, batch.st_index,
#                                 (batch.edges_src, batch.edges_tgt, batch.edges_type, batch.edges_pos),
#                                 batch.start_positions, batch.end_positions, batch.answer_type)
        loss = []
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        tok_logits = []
        res_labels = []
#        print(input_idss.shape)
        if edgess is not None:
            edges_srcs, edges_tgts, edges_types, edges_poss = edgess
            for input_ids, attention_mask, token_type_ids, st_mask, label,edges_src, edges_tgt, edges_type, edges_pos,all_sen in zip(input_idss, attention_masks, token_type_idss, st_masks, labels,edges_srcs, edges_tgts, edges_types, edges_poss,all_sens):
    
                if self.args.run_og:
                    sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids) 
    #                .requires_grad_()
    
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
                        x = graph_output
                    
    #                    x = self.dropout(sequence_output[:,0])
    #                    print(x)
    #                    x = self.dropout(graph_output)
                        # tok_logits.append(self.tok_outputs(x).squeeze(-1))
                        # x = self.dropout(graph_output)
                        tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
    
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
    
    
                for index,lab in enumerate(label):
                    if lab == 1:
                        res_labels.append(index)
        else:
            for input_ids, attention_mask, token_type_ids, label, all_sen in zip(input_idss, attention_masks, token_type_idss, labels, all_sens):
                sequence_output,_ = self.bert(input_ids,  attention_mask,token_type_ids) 
                graph_output = self.encoder(sequence_output, None, None, None, None, None, all_sen,output_all_encoded_layers=False)
                x = self.dropout(graph_output)
                # x = graph_output
            
#                    x = self.dropout(sequence_output[:,0])
#                    print(x)
#                    x = self.dropout(graph_output)
                # tok_logits.append(self.tok_outputs(x).squeeze(-1))
                # x = self.dropout(graph_output)
                tok_logits.append(self.tok_outputs(self.dropout(torch.tanh(self.tok_dense(x)))).squeeze(-1))
                res_labels.append(label[0])

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