
""" Module containing functions to tokenize text



"""
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer,AutoModelForCausalLM
    
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_bert(batch):
    """ Tokenizes dataset with BERT
    For models which associate one sentence with one label
    """
    return bert_tokenizer(batch["sentences"][0], padding="max_length", max_length=100, truncation=True)
    
def tokenize_bert_multisentence(batch):
    """ Tokenizes dataset with BERT
    For models which associate multiple sentences with one label
    """
    BertTokenized=bert_tokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    return BertTokenized


GPTtokenizer = AutoTokenizer.from_pretrained('gpt2')
GPTtokenizer.pad_token=GPTtokenizer.eos_token

def tokenize_GPT(batch):
    """Tokenizes dataset with GPT"""
    return GPTtokenizer(batch["sentences"][0], padding="max_length", max_length=100, truncation=True)

def tokenize_BERT_GPT_(batch):
    """Tokenizes dataset for BERT[GPT]"""
    GPTmymodel = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    GPTmymodel.config.pad_token_id=GPTmymodel.config.eos_token_id

    #Generates at most 50 tokens with GPT
    tokenized=GPTtokenizer(batch["sentences"][0], return_tensors="pt")["input_ids"].to(device)
    generated=GPTmymodel.generate(tokenized, max_new_tokens=50)

    sentence=GPTtokenizer.decode(generated[0])
    return bert_tokenizer(batch["sentences"][0], sentence[tokenized.size(-1):], padding="max_length", max_length=100, truncation=True)

def tokenize_BERT_plus_GPT(batch):
    """Tokenizes dataset for BERT+GPT"""
    BertTokenized=bert_tokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True)
    GPTTokenized=GPTtokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True)
    #Appends BERT and GPT tokenizations together
    BertTokenized["input_ids"]=torch.stack((BertTokenized["input_ids"],GPTTokenized["input_ids"]))
    BertTokenized["attention_mask"]=torch.stack((BertTokenized["attention_mask"],GPTTokenized["attention_mask"]))
    return BertTokenized


BARTtokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
BARTtokenizer.pad_token=BARTtokenizer.eos_token

def tokenize_BART(batch):
    """Tokenizes dataset with BART"""
    BARTTokenized=BARTtokenizer(batch["sentences"][0], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    return BARTTokenized

def tokenize_GBLS(batch):
    """Tokenizes dataset for GBLS"""
    BertTokenized=bert_tokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    GPTTokenized=GPTtokenizer(batch["sentences"], return_tensors="pt", padding="max_length", max_length=100, truncation=True).to(device)
    BertTokenized["input_ids"]=torch.cat((BertTokenized["input_ids"],GPTTokenized["input_ids"]),0)
    BertTokenized["attention_mask"]=torch.cat((BertTokenized["attention_mask"],GPTTokenized["attention_mask"]),0)
    return BertTokenized

def tokenize_GBAS(batch):
    """Tokenizes dataset for GBAS"""
    tokenized_inputs=bert_tokenizer(batch["sentences"] , padding="max_length", max_length=100,truncation=True,return_tensors="pt")
    tokenized_inputs2=GPTtokenizer(batch["sentences"] , padding="max_length", max_length=100,truncation=True,return_tensors="pt")

    tokenized_inputs["input_ids"]=torch.cat((tokenized_inputs["input_ids"],tokenized_inputs2["input_ids"]),0)

    tokenized_inputs["attention_mask"]=torch.cat((tokenized_inputs["attention_mask"],tokenized_inputs2["attention_mask"]),0)

    return tokenized_inputs