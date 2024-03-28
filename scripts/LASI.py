import torch
import datasets
datasets.disable_progress_bar()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for debug
from transformers import set_seed
set_seed(12)
import random
random.seed(12)


import dataset_processing
import data_tokenizers
import model_training
import models


#Batch size as input
#Put check to ensure combination of input_data, model_type, evaluation_method valid
#If Code theoretically can handle it, then ok
#Function with noise and without noise
#Maybe collapse noise and without noise
def run_process(input_data, model_type,evaluation_method, batch_size):

    #input:
    if evaluation_method=="No Noise":
        if input_data=="s_{k}":#s_{k} Current Sentence
            ProcessedData = dataset_processing.process_data_Pubmed([0],0,0)
        elif input_data=="s_{k}s_{k-1}":#s_{k-1} Previous Sentence
            ProcessedData = dataset_processing.process_data_Pubmed([-1,0],1,0)
        elif input_data=="s_{k+1}s_{k}s_{k-1}":#s_{k+1} Previous Sentence
            ProcessedData = dataset_processing.process_data_Pubmed([-1,0,1],1,1)    
        elif input_data=="s_{k-1}":
            ProcessedData = dataset_processing.process_data_Pubmed([-1],1,0) 
        elif input_data=="s_{k-1}s_{k-2}":
            ProcessedData = dataset_processing.process_data_Pubmed([-2,-1],2,0)
    elif evaluation_method=="Noise":
        if input_data=="s_{k-1}":
            ProcessedData = dataset_processing.process_data_Pubmed_noise([-1],1,0)
        elif input_data=="s_{k-1}s_{k-2}":
            ProcessedData = dataset_processing.process_data_Pubmed_noise([-2,-1],2,0)

    #model      
    if model_type=="BERT" and (input_data=="s_{k}" or input_data=="s_{k-1}"):
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_bert, batched=False, batch_size=None)
        myModel = models.bert_base_uncased()
    elif model_type=="BERT" and (input_data=="s_{k}s_{k-1}" or input_data=="s_{k-1}s_{k-2}"):
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_bert_multisentence, batched=False, batch_size=None)
        myModel = models.multi_sentence_bert(2)
    elif model_type=="BERT" and input_data=="s_{k+1}s_{k}s_{k-1}":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_bert_multisentence, batched=False, batch_size=None)
        myModel = models.multi_sentence_bert(3)
    elif model_type=="GPT":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_GPT, batched=False, batch_size=None)
        myModel = models.GPT_model()
    elif model_type=="BERT[GPT]":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_BERT_GPT_, batched=False, batch_size=None)
        myModel = models.bert_base_uncased()
    elif model_type=="BERT+GPT":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_BERT_plus_GPT, batched=False, batch_size=None)
        myModel = models.BERT_plus_GPT()
    elif model_type=="BART":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_BART, batched=False, batch_size=None)
        myModel = models.BART_model()
    elif model_type=="GBLS":
        encoded_data = ProcessedData.map(data_tokenizers.tokenize_GBLS, batched=False, batch_size=None)
        myModel = models.GBLS()
    elif model_type=="GBAS":
        encoded_data=ProcessedData.map(data_tokenizers.tokenize_GBAS, batched=False, batch_size=None)
        myModel=models.GBAS()
        
    #training
    myModel=myModel.to(device)
    trainedModel = model_training.fine_tuning(myModel, encoded_data, batch_size)
    
    #testing
    if evaluation_method=="No Noise":
        model_training.test(trainedModel,encoded_data)
    elif evaluation_method=="Noise":
        model_training.test_with_noise(trainedModel,encoded_data)
        
def main(experiment):
    # we organize the code by experiments. But we recommend run one process in one round. 
    # Since some model's parameters may be affected by previous process run. 
    if experiment=="SI":
        #1 done#run_process("s_{k}", "BERT", "No Noise",15)
        #2 done#run_process("s_{k}s_{k-1}", "BERT", "No Noise",20)
        run_process("s_{k+1}s_{k}s_{k-1}", "BERT", "No Noise",10)
    elif experiment=="LASI_Baseline": #Switch to ELIF later
        #if the above three are correct, no need to test these two
        run_process("s_{k-1}", "BERT", "No Noise",10) # Maybe doesn't work. Code runs to completion, however, spits out notification. Pretty sure that notification is just because I am using less data, code used in paper gives same notification when run on smaller dataset.
        run_process("s_{k-1}s_{k-2}", "BERT", "No Noise",10)#Same situation for this one
    elif experiment=="LASI":
        #4 done#run_process("s_{k-1}", "GPT", "No Noise",10) #Same issue
        #5 too slow to test#run_process("s_{k-1}", "BERT[GPT]", "No Noise",10)
        #6 done#run_process("s_{k-1}", "BERT+GPT", "No Noise",10)
        #7 done#run_process("s_{k-1}", "BART", "No Noise",10)
        #8 done#run_process("s_{k-1}s_{k-2}", "GBLS", "No Noise",10)
        #----------------#Up to here
        run_process("s_{k-1}s_{k-2}", "GBAS", "No Noise",10)
        pass
    elif experiment=="LASI_tweaked":
        #no need to test if the noise generated correctly
        run_process("s_{k-1}", "BERT", "Noise",10)
        run_process("s_{k-1}", "BART", "Noise",10)
        run_process("s_{k-1}s_{k-2}", "GBLS", "Noise",10)
        run_process("s_{k-1}s_{k-2}", "GBAS", "Noise",10)


if __name__ == '__main__':
    main("LASI")

