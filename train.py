"""
This script is utilised to train or models
"""
import logging
# get TF logger for pre-trained transformer model
# envornment = tf_m1  conda activate tf_m1
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
#print = log.info
import random as python_random
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.losses import CategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback
import csv
from sklearn.preprocessing import LabelBinarizer
import utils
from pprint import pprint
import pandas as pd

    
def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""
    training_set = config["training-set"]

    if training_set.lower() == "train":
        data_frame = pd.read_csv(dir+'/train2.txt', sep='\t')
    data_frame_dev = pd.read_csv(dir+'/dev.txt', sep='\t')
    
    data_frame = filter(data_frame, config)
    data_frame_dev = filter(data_frame_dev, config)

    
    return data_frame['premise_nl'], data_frame['hypothesis_nl'], data_frame['label'], data_frame_dev['premise_nl'], data_frame_dev['hypothesis_nl'], data_frame_dev['label']
    
def filter(df, config):
     quality_estimation = config["quality-estimation"]
     translation = config["translation"]
     threshold = config["threshold"]
     
     #df = df[f'{quality_estimation}_{translation}' > threshold]
     query = f"{quality_estimation}_{translation} < {threshold}"
     df = df.query(query)
     
     return df


   

def classifier(X_train_p, X_train_h, X_dev_p, X_dev_h, Y_train, Y_dev, config, model_name):

    """Train and Save model for test and evaluation"""

    # set random seed to make results reproducible
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    # set model parameters
    max_length  =  config['max_length']
    learning_rate =  config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]

    if config["loss"].upper() == "CATEGORICAL":
        loss_function = CategoricalCrossentropy(from_logits=True)
#
    if config['optimizer'].upper() == "ADAM":
        optim = Adam(learning_rate=learning_rate)
    elif config['optimizer'].upper() == "SGD":
        optim = SGD(learning_rate=learning_rate)
        
    #nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # set tokenizer according to pre-trained model


    # get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    X_train = []
    X_dev = []
    #.encode('ascii', 'ignore').decode('ascii')
    for i,j in zip(X_train_p,X_train_h):
        d = []
        d.append(""+i)
        d.append(""+j)
        X_train.append(d)
    for i,j in zip(X_dev_p, X_dev_h):
        d = []
        d.append(""+i)
        d.append(""+j)
        X_dev.append(d)
    
    

    # transform raw texts into model input
    tokens_train = tokenizer(X_train,padding=True,truncation=True, return_tensors="np").data
    tokens_dev =  tokenizer(X_dev, padding=True,truncation=True, return_tensors="np").data

    pprint(f'tokens_train: {tokens_train}')
    pprint(f'tokens_train: {tokens_train.keys()}')


    #convert Y into one hot encoding
    Y_train = tf.one_hot(Y_train.tolist(),depth=2)
    Y_dev = tf.one_hot(Y_dev.tolist(),depth=2)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    # callbacks for ealry stopping and saving model history
    es = EarlyStopping(monitor="val_loss", patience=patience,
        restore_best_weights=True, mode='max')
    history_logger = CSVLogger(utils.LOG_DIR+model_name+"_HISTORY.csv",
        separator=",", append=True)

    # train models
    model.fit(tokens_train, Y_train, verbose=1, epochs=epochs,
        batch_size= batch_size, validation_data=(tokens_dev, Y_dev),
        callbacks=[es, history_logger, TqdmCallback(verbose=2)])

    # save models in directory
    model.save_pretrained(save_directory=utils.MODEL_DIR+model_name)


def set_log(model_name):

    #Create Log file to save info
    try:
        os.mkdir(utils.LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
        log.setLevel(logging.INFO)

    
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs info
    fh = logging.FileHandler(utils.LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device 
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))



    #get parameters for experiments
    config, model_name = utils.get_config()
    
    
    if config['training-set'] != 'trial':
        model_name = model_name+"_"+str(config['seed'])

    set_log(model_name)

    X_train_p, X_train_h, Y_train, X_dev_p, X_dev_h, Y_dev = load_data(utils.DATA_DIR, config)
    #run model

    classifier(X_train_p, X_train_h, X_dev_p, X_dev_h, Y_train, Y_dev, config, model_name)

  

if __name__ == "__main__":
    main()



