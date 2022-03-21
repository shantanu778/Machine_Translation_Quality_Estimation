import json


DATA_DIR = 'Data'
MODEL_DIR = "Saved_Models/"
OUTPUT_DIR = "Output/"
LOG_DIR = "Logs/"

def get_config():

    """Return model name and paramters after reading it from json file"""
    try:
        location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v).upper() for v in configs.values()]
            model_name = "_".join(vals[:-1])
        return configs, model_name
    except FileNotFoundError as error:
        print(error)

def change_dtype(tokens):

    """Return model inputs after changing data type to int32"""
    
    tokens['input_ids'] = tokens['input_ids'].astype('int32')
    tokens['input_ids'] = tokens['input_ids'].astype('int32')

    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    tokens['attention_mask'] = tokens['attention_mask'].astype('int32')
    
    return tokens

def read_data(file):
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(int(tokens[-1]))
    return sentences, labels
