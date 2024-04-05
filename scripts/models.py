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
    """Returns BERT model""" 
    mymodel = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return mymodel


class multi_sentence_bert(nn.Module):     
    """Model which applies BERT to each of multiple sentences individually"""
    def __init__(self, total_sentences):
        super(multi_sentence_bert, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.num_sentences=total_sentences
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768*self.num_sentences, num_labels)
                
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):

        outputs1 = self.bert(input_ids[:,0:self.num_sentences, :].view(input_ids.shape[0]*self.num_sentences,-1), 
                            attention_mask=attention_mask[:,0:self.num_sentences,:].view(input_ids.shape[0]*self.num_sentences,-1),
                            token_type_ids=token_type_ids[:,0:self.num_sentences,:].view(input_ids.shape[0]*self.num_sentences,-1))
        
        #Extract last hidden state of [CLS] token
        bertOutput=outputs1[0][:,0,:].view(input_ids.shape[0],self.num_sentences,-1)

        bertOutputArray = []
        
    
        #concatinates last hidden states of [CLS] token of each sentence together before classification
        for i in range(0,self.num_sentences):
            bertOutputArray.append(bertOutput[:,i,:])
        outputs=torch.cat(bertOutputArray,1)

        outputs = self.dropout(outputs)

        
        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss,logits=logits)
    

def GPT_model():
    """Returns GPT model"""
    myGPTmodel = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
    myGPTmodel.config.pad_token_id=myGPTmodel.config.eos_token_id
    return myGPTmodel


class BERT_plus_GPT(nn.Module):   
    """Model which processes text both with BERT and GPT, then appends outputs together before classification"""
    def __init__(self):
        super(BERT_plus_GPT, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.gpt = AutoModel.from_pretrained('gpt2')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768*2, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):

        
        outputs1 = self.bert(input_ids[:,0,:].view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask[:,0,:].view(input_ids.shape[0],-1),
                            token_type_ids=token_type_ids.view(input_ids.shape[0],-1))
        
        outputs2 = self.gpt(input_ids[:,1,:].view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask[:,1,:].view(input_ids.shape[0],-1))

        #Extract last hidden state of [CLS] token of BERT embedding
        bertOutput=outputs1[0][:,0,:]

        #Extract last hidden state of last token of GPT embedding
        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,1,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)

        gptOutput=hidden_states[torch.arange(batch_size), sequence_lengths,:]
        
        #Concatenates outputs together for classification
        outputs=torch.cat((bertOutput,gptOutput),1)

        outputs = self.dropout(outputs)

        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss,logits=logits)
    



class BART_model(nn.Module):    
    """BART model"""
    def __init__(self):
        super(BART_model, self).__init__()
        self.gpt = AutoModel.from_pretrained('facebook/bart-base')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,labels = None):

        outputs2 = self.gpt(input_ids.view(input_ids.shape[0],-1), 
                            attention_mask=attention_mask.view(input_ids.shape[0],-1))

        #Extracts last hidden state of last token of BART embedding
        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]

        sequence_lengths = (torch.eq(input_ids, BART_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)

        gptOutput=hidden_states[torch.arange(batch_size), sequence_lengths,:]
         
        #Classification
        outputs=gptOutput
        outputs = self.dropout(outputs)

        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss,logits=logits)     
    

class GBLS(nn.Module):    
    """Loss stitching model -- uses 2 layer mapper to map GPT representations to BERT representations before classification"""
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

        #Extract last hidden state of [CLS] token of BERT embedding of sk-1
        bertOutput=outputs1[0][:,0,:]

        
        #Extract last hidden state of last token of GPT embedding of sk-2 and sk-1
        hidden_states = outputs2[0]
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,2:4,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)
    
        gptOutput=hidden_states[torch.arange(2*batch_size), sequence_lengths,:].view(input_ids.shape[0],2,-1)
        
        #Map GPT embeddings to BERT embeddings using mapper
        #Mapper attempts to map GPT embeddings of sn to BERT embeddings of sn+1
        #GPT embedding of sk-2 mapped to BERT embedding of sk-1
        #GPT embedding of sk-1 mapped to BERT embedding of sk
        BertPredicted=self.mapper2(self.tanh(self.mapper(gptOutput)))
        
        #Stores mapper's attempt to map GPT embedding of sk-2 for later
        BertPredict=BertPredicted[:,0,:]
      
        #Concatinates BERT embedding and the GPT embedding of sk-1 together for classification
        outputs=torch.cat((bertOutput,BertPredicted[:,1,:]),1)

        outputs = self.dropout(outputs)

        logits = self.linear(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
            if(self.training):
                Mapper_loss_fct = nn.MSELoss()
                
                #Compares mapper's attempt to map GPT embedding of sk-2 to BERT embedding with BERT embedding of sk-1
                #Includes difference in loss
                Mapper_loss = Mapper_loss_fct(bertOutput, BertPredict)
                loss=loss+Mapper_loss*0.05
            
        return SequenceClassifierOutput(loss=loss,logits=logits)  # (loss), scores, (hidden_states), (attentions)    


import numpy, copy
bert_config=AutoConfig.from_pretrained("bert-base-uncased")
num_labels = 5

class GBAS(nn.Module):
    """Attention stitching model -- uses multi-head attention to map GPT representations to BERT representations before classification"""
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

        
    def forward(self,         
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):        
       
        #Masking the input for bert so that a memorization of the mapping can not be formed.
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
        #Extract last hidden state of [CLS] token of BERT embedding of sk-1
        bert_outputs = bert_outputs[0][:,0,:].view(input_ids.shape[0],2,-1)

        gpt_outputs = self.gpt2(
            input_ids[:,2:4,:].reshape(input_ids.shape[0]*2,-1), 
            attention_mask=attention_mask[:,2:4,:].reshape(input_ids.shape[0]*2,-1),
        )    

        #Extract last hidden state of last token of GPT embedding of sk-2 and sk-1    
        hidden_states = gpt_outputs[0]
 
        batch_size, sequence_length = input_ids.shape[:2]
        sequence_lengths = (torch.eq(input_ids[:,2:4,:], gpt_config.pad_token_id).long().argmax(-1) - 1).reshape(-1)

        gpt_outputs=hidden_states[torch.arange(batch_size*2), sequence_lengths,:].view(input_ids.shape[0],2,-1)

        
        bert_output2=bert_outputs[:,0,:]#bert sk-2 
        bert_output1=bert_outputs[:,1,:]#bert sk-1
              
        gpt_outputs2=gpt_outputs[:,0,:]#gpt sk-2 represent sk-1
        gpt_outputs1=gpt_outputs[:,1,:]#gpt sk-1 represent sk

        #Use multi-head attention to map gpt sk-1 to a BERT embedding of sk
        gpt_trans,_=self.atten(gpt_outputs1,gpt_outputs2,bert_output1)
     
        
        gpt_trans = self.dropout(gpt_trans)
        gpt_trans=gpt_trans+gpt_outputs1
        gpt_trans = self.layernorm(gpt_trans)

        #Concatinate attention output and BERT embedding of sk-1 for classification
        sequence_output=torch.cat((bert_output1,gpt_trans),1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))


        return SequenceClassifierOutput(loss=loss,logits=logits)  