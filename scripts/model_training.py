
"""Module which contains different model training methods

"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def __perf_metrics(pred):
    """Defines how metrics to be computed"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1 score': f1, 'precision': prec, 'recall': rec}


def fine_tuning(mymodel, dataset_encoded , batch_size):
    """Method to fine tune model on tokenized dataset
    
    Keyword Arguments:
    myModel -- the model to be trained
    dataset_encoded -- tokenized dataset to fine tune model on
    batch_size -- batch size
    
    """
    #Controls how training is to be done
    batch_size = batch_size
    logging_steps = len(dataset_encoded['train'])
    model_name = 'Custom Text Classifier'
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=10,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy='epoch',
                                    disable_tqdm=True,
                                    logging_steps=logging_steps,
                                    log_level='error',
                                    save_strategy="epoch",
                                    load_best_model_at_end=True,
                                    metric_for_best_model="accuracy"
                                    )

    #Trains model
    trainer = Trainer(model=mymodel,
                    args=training_args,
                    compute_metrics=__perf_metrics,
                    train_dataset=dataset_encoded['train'],
                    eval_dataset=dataset_encoded['validation'])
    trainer.train()

    return trainer

def test(trainer, dataset_encoded):
    """Method which tests trained model on testing data
    
    Keyword arguments:
    trainer -- instance of Trainer class
    dataset_encoded -- tokenized dataset to test model effectiveness on
    """
    print("Validation results:")
    trainer.evaluate(dataset_encoded['validation'])
    print("Testing results:")
    trainer.evaluate(dataset_encoded['test'])
    
def test_with_noise(trainer, dataset_encoded):
    """Method which tests trained model on testing data with noise
    
    Keyword arguments:
    trainer -- instance of Trainer class
    dataset_encoded -- tokenized dataset to test model effectiveness on (Must include multiple versions of test with different kinds of noise)
    """
    print("Validation:")
    trainer.evaluate(dataset_encoded['validation'])
    print("test delete 1:")
    trainer.evaluate(dataset_encoded['testdel1'])
    print("test delete 2:")
    trainer.evaluate(dataset_encoded['testdel2'])
    print("test add 1:")
    trainer.evaluate(dataset_encoded['testadd1'])
    print("test add 2:")
    trainer.evaluate(dataset_encoded['testadd2'])
    

    

    

