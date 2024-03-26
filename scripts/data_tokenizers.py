
""" Module containing functions to tokenize text



"""
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer,AutoModelForCausalLM
    
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#Bert Model
def tokenize_bert(batch):
    """ Tokenizes dataset with Bert
    """
    return bert_tokenizer(batch["sentences"][0], padding="max_length", max_length=100, truncation=True)
    

def tokenize_bert_multisentence(batch):
    BertTokenized=bert_tokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    return BertTokenized


#GPT Tokenization
GPTtokenizer = AutoTokenizer.from_pretrained('gpt2')
GPTtokenizer.pad_token=GPTtokenizer.eos_token

def tokenize_GPT(batch):
    return GPTtokenizer(batch["sentences"][0], padding="max_length", max_length=100, truncation=True)


#BERT[GPT] Tokenization
def tokenize_BERT_GPT_(batch):
    GPTmymodel = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    GPTmymodel.config.pad_token_id=GPTmymodel.config.eos_token_id

    tokenized=GPTtokenizer(batch["sentences"][0], return_tensors="pt")["input_ids"].to(device)
    generated=GPTmymodel.generate(tokenized, max_new_tokens=50)
    sentence=GPTtokenizer.decode(generated[0])
    return bert_tokenizer(batch["sentences"][0], sentence[tokenized.size(-1):], padding="max_length", max_length=100, truncation=True)


#BERT+GPT
def tokenize_BERT_plus_GPT(batch):
    BertTokenized=bert_tokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True)
    GPTTokenized=GPTtokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True)
    BertTokenized["input_ids"]=torch.stack((BertTokenized["input_ids"],GPTTokenized["input_ids"]))
    BertTokenized["attention_mask"]=torch.stack((BertTokenized["attention_mask"],GPTTokenized["attention_mask"]))
    return BertTokenized


#BART
BARTtokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
BARTtokenizer.pad_token=BARTtokenizer.eos_token

def tokenize_BART(batch):
    BARTTokenized=BARTtokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    return BARTTokenized


#GBLS (Maybe fuse with BERT+GPT)
def tokenize_GBLS(batch):
    BertTokenized=bert_tokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    GPTTokenized=GPTtokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    BertTokenized["input_ids"]=torch.cat((BertTokenized["input_ids"],GPTTokenized["input_ids"]),0)
    BertTokenized["attention_mask"]=torch.cat((BertTokenized["attention_mask"],GPTTokenized["attention_mask"]),0)
    return BertTokenized


#GBAS
def tokenize_GBAS(batch):
    tonenized_inputs=bert_tokenizer(batch["sentences"] , padding="max_length", max_length=100,truncation=True,return_tensors="pt")
    tonenized_inputs2=GPTtokenizer(batch["sentences"] , padding="max_length", max_length=100,truncation=True,return_tensors="pt")
    #print(tonenized_inputs2)
    tonenized_inputs["input_ids"]=torch.cat((tonenized_inputs["input_ids"],tonenized_inputs2["input_ids"]),0)
    #bart does not have token_type_ids
    tonenized_inputs["attention_mask"]=torch.cat((tonenized_inputs["attention_mask"],tonenized_inputs2["attention_mask"]),0)
    #print(tonenized_inputs2)
    return tonenized_inputs
