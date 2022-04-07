"""
This script is created to test our models
"""

import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import os
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import random as python_random
import utils
import argparse



def create_arg_parser():

    '''Returns a map with commandline parameters taken from the user'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-ts",
        "--testset",
        default="24",
        type=str,
        help="define the test set. By default it uses "
             "the 24th meeting as test set. Input "
             " '25' to use the 25th meeting as test set."
        )

    args = parser.parse_args()
    return args
    

def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""

    data_frame_test = pd.read_csv(dir+'/SICK-NL.txt', sep='\t')
    data_frame_test['entailment_label'] = data_frame_test['entailment_label'].map({'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2})
    print(data_frame_test['entailment_label'] )
    return data_frame_test['sentence_A'], data_frame_test['sentence_B'], data_frame_test['entailment_label']

def save_output(Y_test, Y_pred, model_name):

    """save models prediction as csv file"""


    df = pd.DataFrame()
    print(Y_pred)
    print(Y_test)
    df['Test'] = Y_test
    df['Predict'] = Y_pred
    print("Writing output")

    #save output in directory
    try:
        os.mkdir(utils.OUTPUT_DIR)
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
   

def test(X_test_a, X_test_b, Y_test, config, model_name):

    """Return models prediction"""

    #set random seed to make results reproducible 
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
   
    if config["model"].upper() =='LONG':
        lm = "allenai/longformer-base-4096"
    elif config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
    elif config["model"].upper() =='ERNIE':
        lm = 'nghuyong/ernie-2.0-en'
    elif config["model"].upper() =='XLNET':
        lm = 'xlnet-base-cased'
        
    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    
    X_test = []
    for i,j in zip(X_test_a,X_test_b):
        d = []
        d.append(""+i)
        d.append(""+j)
        X_test.append(d)
    print(X_test)

    #transform raw texts into model input 
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    #change the data type of model inputs to int32 
    if config["model"] =='LONG':
        tokens_test = utils.change_dtype(tokens_test)

    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(utils.MODEL_DIR+model_name+"_1234")

    #get model's prediction
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    Y_pred = np.argmax(Y_pred, axis=1)
    #Y_test = np.argmax(Y_test, axis=1)
    
    return Y_test, Y_pred

def set_log(model_name):

    #create log file
    try:
        os.mkdir(utils.LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
    
        log.setLevel(logging.INFO)
    
    #create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #create file handler which logs info
    fh = logging.FileHandler(utils.LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    #get parameters for experiments
    config, model_name = utils.get_config()

    #set log settings
    set_log(model_name)

    args = create_arg_parser()

    #load data from train-test-dev folder
    X_train_a, X_train_b, Y_train = load_data(utils.DATA_DIR,args.testset)
    Y_test, Y_pred = test(X_train_a, X_train_b, Y_train, config, model_name)
    
    #save output in directory
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()

