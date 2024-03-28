
""" Module containing functions to process text data into forms suitable for LASI



"""

import torch
import pandas as pd
import datasets
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = datasets.load_dataset('armanc/pubmed-rct20k')

# for debug
dataset["train"]=dataset["train"][0:2500]
dataset["validation"]=dataset["validation"][0:2500]
dataset["test"]=dataset["test"][0:2500]


  


def process_data_Pubmed(filter_array, low, high):
    """Processes data in accordance with method given to make it suitable for training
    
    Keyword arguments:
    datset -- Pubmed or other similarly structured data
    filter_array -- Array of integers
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
       
     
    """
    
    #Modifies pubmed data into more easy to use format
    new_train_text=modify_pubmed_format(dataset["train"]["text"],dataset["train"]["sentence_id"])
    new_train_label=modify_pubmed_format(dataset["train"]["label"],dataset["train"]["sentence_id"])

    new_test_text=modify_pubmed_format(dataset["test"]["text"],dataset["test"]["sentence_id"])
    new_test_label=modify_pubmed_format(dataset["test"]["label"],dataset["test"]["sentence_id"])

    new_validation_text=modify_pubmed_format(dataset["validation"]["text"],dataset["validation"]["sentence_id"])
    new_validation_label=modify_pubmed_format(dataset["validation"]["label"],dataset["validation"]["sentence_id"])

    #Processes Pubmed data
    new_train=associate_sentences_with_labels_pubmed(new_train_text, new_train_label,filter_array,low,high)
    new_test=associate_sentences_with_labels_pubmed(new_test_text, new_test_label,filter_array,low,high)
    new_validation=associate_sentences_with_labels_pubmed(new_validation_text, new_validation_label,filter_array,low,high)
    
    new_train=datasets.Dataset.from_pandas(pd.DataFrame(data=new_train))
    new_validation=datasets.Dataset.from_pandas(pd.DataFrame(data=new_validation))
    new_test=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test))

    new_dataset=datasets.DatasetDict({"train":new_train,"validation":new_validation,"test":new_test})
    
    return new_dataset

def modify_pubmed_format(text_array, index_array):
    """Changes the format of Pubmed-formatted data
    
    Keyword arguments:
    text_array -- array of either labels or sentences to be grouped into sections
    index_array -- array of numbers which act as deminators between different data sources
    """
    return_array=[]
    append_array=[]
    i = 1
    append_array.append(text_array[0])
    while(i<len(text_array)):
        if(index_array[i]>index_array[i-1]):
            append_array.append(text_array[i])
        else:
            return_array.append(append_array)
            append_array=[]
            append_array.append(text_array[i])
        i=i+1
    return_array.append(append_array)
    return return_array

def associate_sentences_with_labels_pubmed(sentences, labels , filter_array, low, high):
    """Converts Pubmed data into array of dictionaries where each dictionary associates several sentences to a label
    
    Keyword arguments:
    sentences -- Pubmed training, testing, or verification data
    labels -- Pubmed labels
    filter_array -- Array of integers which dictates how labels grouped with sentences
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
    """
    #note to self: Changed function call to dictionary
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
    
    
    return_array = []
    for i in range(0,len(labels)):
        for j in range(low,len(labels[i])-high):
                
            temp={}
                
            sentence_array=[]
            for k in range(0,len(filter_array)):
                sentence_array.append(sentences[i][j+filter_array[k]])
        
            temp["sentences"]=sentence_array
        
            temp["labels"]=numericalize[labels[i][j]]
        
            return_array.append(temp)
            
    return return_array



def process_data_Pubmed_noise(filter_array, low, high):
    """Processes data in accordance with method given to make it suitable for training
    
#     Keyword arguments:
#     datset -- Pubmed or other similarly structured data
#     filter_array -- Array of integers
#         Dictates which sentences should be grouped with which labels
#         A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
       
     
#     """
    
    #Modifies pubmed data into more easy to use format
    newTrainText=modify_pubmed_format(dataset["train"]["text"],dataset["train"]["sentence_id"])
    newTrainLabel=modify_pubmed_format(dataset["train"]["label"],dataset["train"]["sentence_id"])

    newTestText=modify_pubmed_format(dataset["test"]["text"],dataset["test"]["sentence_id"])
    newTestLabel=modify_pubmed_format(dataset["test"]["label"],dataset["test"]["sentence_id"])

    newValidationText=modify_pubmed_format(dataset["validation"]["text"],dataset["validation"]["sentence_id"])
    newValidationLabel=modify_pubmed_format(dataset["validation"]["label"],dataset["validation"]["sentence_id"])

    newTrain={}
    newValidation={}
    newTest={}

    newTrain["sentences"]=newTrainText
    newValidation["sentences"]=newValidationText
    newTest["sentences"]=newTestText

    newTrain["labels"]=newTrainLabel
    newValidation["labels"]=newValidationLabel
    newTest["labels"]=newTestLabel

    newTrain=datasets.Dataset.from_pandas(pd.DataFrame(data=newTrain))
    newValidation=datasets.Dataset.from_pandas(pd.DataFrame(data=newValidation))
    newTest=datasets.Dataset.from_pandas(pd.DataFrame(data=newTest))

    Data=datasets.DatasetDict({"train":newTrain,"validation":newValidation,"test":newTest})

    #Processes CSAbstract data
    #reuse associate_sentences_with_labels to be efficent
    newTrain=associate_sentences_with_labels_pubmed(Data["train"]["sentences"],Data["train"]["labels"],filter_array,low,high)
    newValidation=associate_sentences_with_labels_pubmed(Data["validation"]["sentences"],Data["validation"]["labels"],filter_array,low,high)
    if filter_array==[-1]:
        newTestdel1=sentenceCombineLASI1TestDelete(Data["test"],1)
        newTestdel2=sentenceCombineLASI1TestDelete(Data["test"],2)
    elif filter_array==[-2,-1]:
        newTestdel1=sentenceCombineLASI2TestDelete(Data["test"],1)
        newTestdel2=sentenceCombineLASI2TestDelete(Data["test"],2)
    newTestadd1=sentenceCombineLASI1TestAdd(Data["test"],filter_array,low,high,1)
    newTestadd2=sentenceCombineLASI1TestAdd(Data["test"],filter_array,low,high,2)
  
    newTrain=datasets.Dataset.from_pandas(pd.DataFrame(data=newTrain))
    newValidation=datasets.Dataset.from_pandas(pd.DataFrame(data=newValidation))
    newTestdel1=datasets.Dataset.from_pandas(pd.DataFrame(data=newTestdel1))
    newTestdel2=datasets.Dataset.from_pandas(pd.DataFrame(data=newTestdel2))
    newTestadd1=datasets.Dataset.from_pandas(pd.DataFrame(data=newTestadd1))
    newTestadd2=datasets.Dataset.from_pandas(pd.DataFrame(data=newTestadd2))

    newDataSet=datasets.DatasetDict({"train":newTrain,"validation":newValidation,"testdel1":newTestdel1,"testdel2":newTestdel2,"testadd1":newTestadd1,"testadd2":newTestadd2})
    
    return newDataSet



def sentenceCombineLASI1TestDelete(dataset, numDelete):
    #3/11/2024 Changed numericalization to be in sentence association step to be consistent
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}

    returnArray = []
    for i in range(0,len(dataset["sentences"])):
        for j in range(1,len(dataset["sentences"][i])):
            
            temp={}
            
            sentenceArray=[] 
            
            wordArray = re.split("\ ", dataset["sentences"][i][j-1])
            for k in range(0, numDelete):
                if(len(wordArray)>0):
                    deletenum = random.randint(0,len(wordArray)-1)
                    wordArray.pop(deletenum)
            
            sentenceArray.append(" ".join(wordArray))

            temp["sentences"]=sentenceArray
            temp["labels"]=numericalize[dataset["labels"][i][j]]
            returnArray.append(temp)
        
    return returnArray

def sentenceCombineLASI2TestDelete(dataset, numDelete):
    #3/11/2024 Changed numericalization to be in sentence association step to be consistent
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
    returnArray=[]
    for i in range(0, len(dataset["sentences"])):
        for j in range(2, len(dataset["sentences"][i])):
            temp={}
            
            sentenceArray=[]
            
            wordArray1 = re.split("\ ", dataset["sentences"][i][j-2])
            wordArray2 = re.split("\ ", dataset["sentences"][i][j-1])
            for k in range(0, numDelete):
                arrayNum=random.randint(0,1)
                if(arrayNum==0):
                    if(len(wordArray1)>0):
                        deletenum = random.randint(0,len(wordArray1)-1)
                        wordArray1.pop(deletenum)
                else:
                    if(len(wordArray2)>0):
                        deletenum = random.randint(0,len(wordArray2)-1)
                        wordArray2.pop(deletenum)
            
            sentenceArray.append(" ".join(wordArray1))
            sentenceArray.append(" ".join(wordArray2))
            temp["sentences"]=sentenceArray
            temp["labels"]=numericalize[dataset["labels"][i][j]]
            returnArray.append(temp)
    return returnArray

def sentenceCombineLASI1TestAdd(dataset, filter_array, low, high, numAdd):
    
    #3/11/2024 Changed numericalization to be in sentence association step to be consistent
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
   
    returnArray = []
    for i in range(0,len(dataset["sentences"])):
        for j in range(low,len(dataset["sentences"][i])-high):
            temp={}
            
            sentenceArray=[]
           
            sentence=dataset["sentences"][i][j]
            wordArray = re.split("\ ", dataset["sentences"][i][j], maxsplit=3)[0:2]
            combinedSentence=""
            for k in range(0, numAdd):
                if(len(wordArray)>k):
                    combinedSentence+=" "+wordArray[k]
                
            
            for k in range(0,len(filter_array)):
                if(k!=len(filter_array)-1):
                    sentenceArray.append(dataset["sentences"][i][j+filter_array[k]])
                else:
                    sentenceArray.append(dataset["sentences"][i][j+filter_array[k]]+combinedSentence)
            
            temp["sentences"]=sentenceArray
           
            temp["labels"]=numericalize[dataset["labels"][i][j]]
            returnArray.append(temp)
        
    return returnArray