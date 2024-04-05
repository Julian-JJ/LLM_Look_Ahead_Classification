
""" Module containing functions to process text data into forms suitable for LASI



"""

import torch
import pandas as pd
import datasets
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = datasets.load_dataset('armanc/pubmed-rct20k')

debug = True

if(debug):
    # for debug
    dataset["train"]=dataset["train"][0:2500]
    dataset["validation"]=dataset["validation"][0:2500]
    dataset["test"]=dataset["test"][0:2500]


  


def process_data_Pubmed(filter_array, low, high):
    """Processes data in accordance with method given to make it suitable for training
    
    Keyword arguments:
    filter_array -- Array of integers
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
    low -- Integer which dictates first sentence to be considered (to avoid accessing invalid index)
    high -- Integer which dictates last sentence to be considered (to avoid accessing invalid index)
    """
    
    #Modifies pubmed data into more easy to use format
    new_train_text=modify_pubmed_format(dataset["train"]["text"],dataset["train"]["sentence_id"])
    new_train_label=modify_pubmed_format(dataset["train"]["label"],dataset["train"]["sentence_id"])

    new_test_text=modify_pubmed_format(dataset["test"]["text"],dataset["test"]["sentence_id"])
    new_test_label=modify_pubmed_format(dataset["test"]["label"],dataset["test"]["sentence_id"])

    new_validation_text=modify_pubmed_format(dataset["validation"]["text"],dataset["validation"]["sentence_id"])
    new_validation_label=modify_pubmed_format(dataset["validation"]["label"],dataset["validation"]["sentence_id"])

    #Associates labels with pubmed data in accordace with filter_array
    new_train=associate_sentences_with_labels_pubmed(new_train_text, new_train_label,filter_array,low,high)
    new_test=associate_sentences_with_labels_pubmed(new_test_text, new_test_label,filter_array,low,high)
    new_validation=associate_sentences_with_labels_pubmed(new_validation_text, new_validation_label,filter_array,low,high)
    
    #Converts data to correct format
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
        #Loops through text
        #Whenever index_array decreases, start new group of sentences
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
    """Associates sentences to a label under method dictated by filter_array
    
    Keyword arguments:
    sentences -- Pubmed training, testing, or verification data
    labels -- Pubmed labels
    filter_array -- Array of integers which dictates how labels grouped with sentences
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
    low -- Integer which dictates first sentence to be considered (to avoid accessing invalid index)
    high -- Integer which dictates last sentence to be considered (to avoid accessing invalid index)
    """
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
    
    
    return_array = []
    for i in range(0,len(labels)):
        for j in range(low,len(labels[i])-high):

            temp={}

            #Find all sentences and associates them to label as dictated by filter_array
            sentence_array=[]
            for k in range(0,len(filter_array)):
                sentence_array.append(sentences[i][j+filter_array[k]])
        
            temp["sentences"]=sentence_array

            temp["labels"]=numericalize[labels[i][j]]
        
            return_array.append(temp)
            
    return return_array



def process_data_Pubmed_noise(filter_array, low, high):
    """Processes data in accordance with method given to make it suitable for training
    Automatically adds noise to testing data
    
    Keyword arguments:
    filter_array -- Array of integers
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
    low -- Integer which dictates first sentence to be considered (to avoid accessing invalid index)
    high -- Integer which dictates last sentence to be considered (to avoid accessing invalid index)
    """
    
    #Modifies pubmed data into more easy to use format
    new_train_text=modify_pubmed_format(dataset["train"]["text"],dataset["train"]["sentence_id"])
    new_train_label=modify_pubmed_format(dataset["train"]["label"],dataset["train"]["sentence_id"])

    new_test_text=modify_pubmed_format(dataset["test"]["text"],dataset["test"]["sentence_id"])
    new_test_label=modify_pubmed_format(dataset["test"]["label"],dataset["test"]["sentence_id"])

    new_validation_text=modify_pubmed_format(dataset["validation"]["text"],dataset["validation"]["sentence_id"])
    new_validation_label=modify_pubmed_format(dataset["validation"]["label"],dataset["validation"]["sentence_id"])

    #Packages data together in one dataset
    new_train={}
    new_validation={}
    new_test={}

    new_train["sentences"]=new_train_text
    new_validation["sentences"]=new_validation_text
    new_test["sentences"]=new_test_text

    new_train["labels"]=new_train_label
    new_validation["labels"]=new_validation_label
    new_test["labels"]=new_test_label

    new_train=datasets.Dataset.from_pandas(pd.DataFrame(data=new_train))
    new_validation=datasets.Dataset.from_pandas(pd.DataFrame(data=new_validation))
    new_test=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test))

    Data=datasets.DatasetDict({"train":new_train,"validation":new_validation,"test":new_test})

    #Associates labels with pubmed data in accordace with filter_array
    new_train=associate_sentences_with_labels_pubmed(Data["train"]["sentences"],Data["train"]["labels"],filter_array,low,high)
    new_validation=associate_sentences_with_labels_pubmed(Data["validation"]["sentences"],Data["validation"]["labels"],filter_array,low,high)
    #Randomly deletes words from test data to add noise
    if filter_array==[-1]:
        new_test_del1=associate_sentences_with_labels_pubmed_test_delete_1(Data["test"],1)
        new_test_del2=associate_sentences_with_labels_pubmed_test_delete_1(Data["test"],2)
    elif filter_array==[-2,-1]:
        new_test_del1=associate_sentences_with_labels_pubmed_test_delete_2(Data["test"],1)
        new_test_del2=associate_sentences_with_labels_pubmed_test_delete_2(Data["test"],2)
    #adds words to end of sentence to add noise
    new_test_add1=associate_sentences_with_labels_pubmed_test_add(Data["test"],filter_array,low,high,1)
    new_test_add2=associate_sentences_with_labels_pubmed_test_add(Data["test"],filter_array,low,high,2)
  
    #Converts data to correct format
    new_train=datasets.Dataset.from_pandas(pd.DataFrame(data=new_train))
    new_validation=datasets.Dataset.from_pandas(pd.DataFrame(data=new_validation))
    new_test_del1=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test_del1))
    new_test_del2=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test_del2))
    new_test_add1=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test_add1))
    new_test_add2=datasets.Dataset.from_pandas(pd.DataFrame(data=new_test_add2))

    new_dataset=datasets.DatasetDict({"train":new_train,"validation":new_validation,"testdel1":new_test_del1,"testdel2":new_test_del2,"testadd1":new_test_add1,"testadd2":new_test_add2})
    
    return new_dataset



def associate_sentences_with_labels_pubmed_test_delete_1(dataset, num_delete):
    """Method which associates previous sentence with label, then randomly deletes num_delete words

    dataset -- Testing data to be processed
    num_delete -- number of words to be deleted
    """
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}

    returnArray = []
    for i in range(0,len(dataset["sentences"])):
        for j in range(1,len(dataset["sentences"][i])):
            
            temp={}
            
            sentence_array=[] 
            
            #Splits previous sentence into words
            word_array = re.split("\ ", dataset["sentences"][i][j-1])
            for k in range(0, num_delete):
                #Makes sure array not empty before deleting word
                if(len(word_array)>0):
                    delete_num = random.randint(0,len(word_array)-1)
                    word_array.pop(delete_num)
            
            #rejoins words together into one sentence
            sentence_array.append(" ".join(word_array))

            temp["sentences"]=sentence_array
            temp["labels"]=numericalize[dataset["labels"][i][j]]
            returnArray.append(temp)
        
    return returnArray

def associate_sentences_with_labels_pubmed_test_delete_2(dataset, num_delete):
    """Method which associates previous two sentences with label, then randomly deletes num_delete words in total

    dataset -- Testing data to be processed
    num_delete -- number of words to be deleted
    """
    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
    returnArray=[]

    for i in range(0, len(dataset["sentences"])):
        for j in range(2, len(dataset["sentences"][i])):
            temp={}
            
            sentence_array=[]
            
            #Gets past two sentences, splits both into words
            word_array1 = re.split("\ ", dataset["sentences"][i][j-2])
            word_array2 = re.split("\ ", dataset["sentences"][i][j-1])
            for k in range(0, num_delete):
                #Picks random sentence to delete from
                arrayNum=random.randint(0,1)
                if(arrayNum==0):
                    #Makes sure array not empty before deleting word
                    if(len(word_array1)>0):
                        deletenum = random.randint(0,len(word_array1)-1)
                        word_array1.pop(deletenum)
                else:
                    #Makes sure array not empty before deleting word
                    if(len(word_array2)>0):
                        deletenum = random.randint(0,len(word_array2)-1)
                        word_array2.pop(deletenum)
            
            #Joins together words back into sentence
            sentence_array.append(" ".join(word_array1))
            sentence_array.append(" ".join(word_array2))
            temp["sentences"]=sentence_array
            temp["labels"]=numericalize[dataset["labels"][i][j]]

            returnArray.append(temp)
    return returnArray

def associate_sentences_with_labels_pubmed_test_add(dataset, filter_array, low, high, num_add):
    """Method which associates sentences with labels according to filter_array, then adds num_add words from next sentence to end of last sentence

    dataset -- Properly formatted Pubmed testing data
    filter_array -- Array of integers which dictates how labels grouped with sentences
        Dictates which sentences should be grouped with which labels
        A filterarray of [-1,0], means that the sentences before and at some position are associated with the label of the sentence in that position
    low -- Integer which dictates first sentence to be considered (to avoid accessing invalid index)
    high -- Integer which dictates last sentence to be considered (to avoid accessing invalid index)
    num_add -- 1 or 2. Number of words to add to the end of the last sentence
    """

    numericalize={'background':0, 'objective':1, 'methods':2,  'results':3, 'conclusions':4}
   
    returnArray = []
    for i in range(0,len(dataset["sentences"])):
        for j in range(low,len(dataset["sentences"][i])-high):
            temp={}
            
            sentence_array=[]
           
            sentence=dataset["sentences"][i][j]
            #Splits next sentence into words
            word_array = re.split("\ ", dataset["sentences"][i][j], maxsplit=3)[0:2]
            combined_sentence=""
            #Collects first num_add words together
            for k in range(0, num_add):
                if(len(word_array)>k):
                    combined_sentence+=" "+word_array[k]
                
            #Associates sentences together as dictated by filter_array
            for k in range(0,len(filter_array)):
                if(k!=len(filter_array)-1):
                    sentence_array.append(dataset["sentences"][i][j+filter_array[k]])
                else:
                    #If last sentence, add on extra words
                    sentence_array.append(dataset["sentences"][i][j+filter_array[k]]+combined_sentence)
            
            temp["sentences"]=sentence_array
            temp["labels"]=numericalize[dataset["labels"][i][j]]
            returnArray.append(temp)
        
    return returnArray