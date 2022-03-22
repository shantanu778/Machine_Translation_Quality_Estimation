"""
This script is utilised to train or models
"""

#print = log.info
import random as python_random
import numpy as np
import os
import pandas as pd
import csv
import utils
from pprint import pprint
import pandas as pd

    
def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""
    training_set = config["training-set"]

    if training_set.lower() == "train":
        data_frame = pd.read_csv(dir+'/train.txt', sep='\t')
    data_frame_dev = pd.read_csv(dir+'/dev.txt', sep='\t')
    
    return data_frame, data_frame_dev
    
def filter(df, config):
     quality_estimation = config["quality-estimation"]
     translation = config["translation"]
     threshold = config["threshold"]
     
     query = f"{quality_estimation}_{translation} > {threshold}"
     df = df.query(query)
     
     return df



def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device 



    #get parameters for experiments
    config, model_name = utils.get_config()
    

    df, df_dev = load_data(utils.DATA_DIR, config)
    #run model
    print(df[["da_premise","mqm_premise", "da_hypothesis","mqm_hypothesis"]].describe())
    print(df.groupby("label").count())



  

if __name__ == "__main__":
    main()


