"""Module containing models"""
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpt_config=AutoConfig.from_pretrained('gpt2')
gpt_config.pad_token_id=gpt_config.eos_token_id
BART_config=AutoConfig.from_pretrained('facebook/bart-base')
BART_config.pad_token_id=BART_config.eos_token_id
num_labels = 5

def bert_base_uncased():    
    mymodel = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return mymodel


class multi_sentence_bert(nn.Module):     
    
    def __init__(self, total_sentences):
        super(multi_sentence_bert, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.num_sentences=total_sentences
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768*self.num_sentences, num_labels)
                
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):
        #print(input_ids.shape)
        #print(attention_mask.shape)
        #print(token_type_ids.shape)
        
      

        #change to reshape if issue
        outputs1 = self.bert(input_ids[:,0:self.num_sentences, :].view(input_ids.shape[0]*self.num_sentences,-1), 
                            attention_mask=attention_mask[:,0:self.num_sentences,:].view(input_ids.shape[0]*self.num_sentences,-1),
                            token_type_ids=token_type_ids[:,0:self.num_sentences,:].view(input_ids.shape[0]*self.num_sentences,-1))
        
        # You write you new head here
        bertOutput=outputs1[0][:,0,:].view(input_ids.shape[0],self.num_sentences,-1)
        # Xin: there was an error here so I revise for you.
        # we input 10 sentence pairs and first view them as 20 sentences
        # after extract CLS, we view it back to 10 sentence pairs
        # at last we seperate them to 1 and 2
        #print(bertOutput.shape)
        bertOutputArray = []
        
        for i in range(0,self.num_sentences):
            bertOutputArray.append(bertOutput[:,i,:])
            

        #print(bertOutput.shape)
       
        outputs=torch.cat(bertOutputArray,1)
        #print(outputs.shape)
        outputs = self.dropout(outputs)
        #print(outputs.shape)
        
        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            

            
        return SequenceClassifierOutput(loss=loss,logits=logits)
    

#GPT Model

def GPT_model():
    myGPTmodel = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
    myGPTmodel.config.pad_token_id=myGPTmodel.config.eos_token_id
    return myGPTmodel


#BERT+GPT Model
class BERT_plus_GPT(nn.Module):   
    
    def __init__(self):
        super(BERT_plus_GPT, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.gpt = AutoModel.from_pretrained('gpt2')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768*2, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):
        #print(input_ids[:,0,:].shape)
        
        outputs1 = self.bert(input_ids[:,0,:].view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask[:,0,:].view(input_ids.shape[0],-1),
                            token_type_ids=token_type_ids.view(input_ids.shape[0],-1))
        
        outputs2 = self.gpt(input_ids[:,1,:].view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask[:,1,:].view(input_ids.shape[0],-1))
        # You write you new head here
        bertOutput=outputs1[0][:,0,:]
        #print(bertOutput.shape)
        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,1,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)
        #print(sequence_lengths)
        
        gptOutput=hidden_states[torch.arange(batch_size), sequence_lengths,:]
        
        
        outputs=torch.cat((bertOutput,gptOutput),1)
        outputs = self.dropout(outputs)
        #print(outputs.shape)
        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss,logits=logits)
    


#BART
class BART_model(nn.Module):    
    
    def __init__(self):
        super(BART_model, self).__init__()
        self.gpt = AutoModel.from_pretrained('facebook/bart-base')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):
        #print(input_ids.shape)
               
        outputs2 = self.gpt(input_ids.view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask.view(input_ids.shape[0],-1))

        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]
        #print(sequence_length)
        sequence_lengths = (torch.eq(input_ids, BART_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)
        
       
        
        gptOutput=hidden_states[torch.arange(batch_size), sequence_lengths,:]
        
        
        outputs=gptOutput
        outputs = self.dropout(outputs)
        #print(outputs.shape)
        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss,logits=logits)     
    


#GBLS

class GBLS(nn.Module):    

    def __init__(self):
        super(GBLS, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.gpt = AutoModel.from_pretrained('gpt2')
        self.mapper=nn.Linear(768,768)
        self.tanh = nn.Tanh()
        self.mapper2 = nn.Linear(768,768)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768*2, num_labels)
        
        
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):
        #print(input_ids.shape)
        #print(attention_mask.shape)
        #print(token_type_ids.shape)
        
        bert_input=input_ids[:,1, :]
        if self.training:
            r_mask=(torch.rand((bert_input.shape[0],bert_input.shape[1]))>0.1).to(device)
            r_mask = torch.logical_or(r_mask,torch.eq(bert_input, 101))
            r_mask = torch.logical_or(r_mask,torch.eq(bert_input, 102))
            bert_input=torch.mul(bert_input,r_mask)

        
        outputs1 = self.bert(bert_input.view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask[:,1,:].view(input_ids.shape[0],-1),
                            token_type_ids=token_type_ids[:,1,:].view(input_ids.shape[0],-1))
        
        outputs2 = self.gpt(input_ids[:,2:4,:].reshape(input_ids.shape[0]*2,-1), 
                            attention_mask=attention_mask[:,2:4,:].reshape(input_ids.shape[0]*2,-1))
        # You write you new head here
        bertOutput=outputs1[0][:,0,:]
        #print(bertOutput.shape)
        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,2:4,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)
        #print(sequence_lengths)
        
        gptOutput=hidden_states[torch.arange(2*batch_size), sequence_lengths,:].view(input_ids.shape[0],2,-1)
        
        #print(gptOutput.shape)
        BertPredicted=self.mapper2(self.tanh(self.mapper(gptOutput)))
        #BertPredicted=self.mapper(gptOutput)
        BertPredict=BertPredicted[:,0,:]
        
        #print(BertPredicted.shape)
        
        #print(bertOutput.shape)
        #print(BertPredicted[:,1,:].shape)
        outputs=torch.cat((bertOutput,BertPredicted[:,1,:]),1)
        #print(outputs.shape)
        outputs = self.dropout(outputs)
        #print(outputs.shape)
        
        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
            if(self.training):
                Mapper_loss_fct = nn.MSELoss()
                #print(bertOutput.shape)
                #print(BertPredicted[:,0,:].shape)
                Mapper_loss = Mapper_loss_fct(bertOutput, BertPredict)
                loss=loss+Mapper_loss*0.05
            
        return SequenceClassifierOutput(loss=loss,logits=logits)  # (loss), scores, (hidden_states), (attentions)    


#GBAS


import numpy, copy
bert_config=AutoConfig.from_pretrained("bert-base-uncased")
num_labels = 5

class GBAS(nn.Module):
    
    def __init__(self):
        super(GBAS, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.gpt2 = AutoModel.from_pretrained('gpt2')
        self.atten = nn.MultiheadAttention(bert_config.hidden_size, 8)
        self.layernorm = nn.LayerNorm(bert_config.hidden_size)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size*2, num_labels)
        self.mapper_weight=.5
        self.current_epoch=0
        
    def set_mapper_weight(self,):        
        self.current_epoch=self.current_epoch+1
#        self.mapper_weight=self.mapper_weight*.9
#        print(self.mapper_weight)
        
    def forward(self,         
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):        
        #print(input_ids[:,0:2,:].reshape(input_ids.shape[0]*2,-1).shape,token_type_ids.shape)
        #randomlize the input for bert so that a memorization of the mapping can not be formed.
        bert_input=input_ids[:,0:2,:]
        if self.training:
            r_mask=(torch.rand((bert_input.shape[0],bert_input.shape[1],bert_input.shape[2]))>0.1).to(device)
            r_mask = torch.logical_or(r_mask,torch.eq(bert_input, 101))
            r_mask = torch.logical_or(r_mask,torch.eq(bert_input, 102))
            bert_input=torch.mul(bert_input,r_mask)

        bert_outputs = self.bert(
            bert_input.reshape(input_ids.shape[0]*2,-1), 
            attention_mask=attention_mask[:,0:2,:].reshape(input_ids.shape[0]*2,-1),
            token_type_ids=token_type_ids.view(input_ids.shape[0]*2,-1),
        )
        bert_outputs = bert_outputs[0][:,0,:].view(input_ids.shape[0],2,-1)
        #print(bert_outputs.shape)
        gpt_outputs = self.gpt2(
            input_ids[:,2:4,:].reshape(input_ids.shape[0]*2,-1), 
            attention_mask=attention_mask[:,2:4,:].reshape(input_ids.shape[0]*2,-1),
        )        
        hidden_states = gpt_outputs[0]
        #print(hidden_states.shape)
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,2:4,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)
        #print(sequence_lengths.shape)
        
        gpt_outputs=hidden_states[torch.arange(batch_size*2), sequence_lengths,:].view(input_ids.shape[0],2,-1)
        #print(gpt_outputs.shape)
        # You write you new head here
        
        bert_output2=bert_outputs[:,0,:]#bert sk-2 
        bert_output1=bert_outputs[:,1,:]#bert sk-1
        
#        gpt_trans=self.atten(gpt_outputs,gpt_outputs,gpt_outputs, need_weights=False)
        gpt_outputs2=gpt_outputs[:,0,:]#gpt sk-2 represent sk-1
        gpt_outputs1=gpt_outputs[:,1,:]#gpt sk-1 represent sk
        #gpt sk-2 -> bert sk-1
        #gpt sk-2, bert k-2   -> bert sk-1 
        #gpt sk-1 -> unknown bert sk
        #gpt sk-1, bert k-1   -> unknown bert sk 
#        gpt_trans,_=self.atten(gpt_outputs1,bert_output1,bert_output1)#qkv (kv from encoder)
#        gpt_trans,_=self.atten(gpt_outputs1,gpt_outputs1,bert_output1)#qkv (qk from encoder)
        gpt_trans,_=self.atten(gpt_outputs1,gpt_outputs2,bert_output1)#qkv (q gpt(sk-1), k gpt(sk-2),v bert(sk-1))
        #only use the atten not updating it
#        temp_atten = copy.deepcopy(self.atten)
#        for param in temp_atten.parameters():
#            param.requires_grad=False
#        gpt_trans0,_=self.atten(gpt_outputs1,sequence_output1,sequence_output1)#qkv
#        gpt_trans1=gpt_trans[:,0,:]#sk-2 represent sk-1
#        gpt_trans0=gpt_trans[:,1,:]#sk-1 represent sk
        
        gpt_trans = self.dropout(gpt_trans)
        gpt_trans=gpt_trans+gpt_outputs1
        gpt_trans = self.layernorm(gpt_trans)
#        sequence_output=gpt_trans
        sequence_output=torch.cat((bert_output1,gpt_trans),1)
        #print(sequence_output.shape)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
#            loss_gblm = nn.MSELoss()
#            loss_gblmvalue = loss_gblm(sequence_output1, gpt_trans1)*self.mapper_weight
#            loss=loss+loss_gblmvalue

        return SequenceClassifierOutput(loss=loss,logits=logits)  # (loss), scores, (hidden_states), (attentions)