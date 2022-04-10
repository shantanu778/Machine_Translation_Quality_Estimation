# Natural Language Processing
This is the repository for the final project of our NLP course, which involves assess how the estimated quality of translated instances affects the performance of models trained on them. To test this we created several experiments, which involves training BERT using a serie of different train sets.

<!-- GETTING STARTED -->
## Getting Started

Before you can start creating your models we firstly need to make sure the datasets we utilised for training our models are present in the Data folder.

### Prerequisites 

Our train set and dev set are the huggingface train set and dev set taken from [this page](https://huggingface.co/datasets/GroNLP/ik-nlp-22_transqe) and our test set are the SICK.txt and the SICK_NL.txt. The train and dev set needs to be saved as train.txt and dev.txt in the Data folder, while the SICK datasets can maintain their original name  


### Training your model (SubTask-A)
If you want to run the train sets from implementation 1 and 3, first go to the Models/LM_BERT_Balanced_labels/ directory.

* step 1: open config.json
* for implentation 1: set "training-set" value to "label_template_balanced_train" and set "model" value to "BERT"
* for implentation 3: set "training-set" value to "custom" and set "model" value to "ERNIE"
* step 2:run 
 ```
  train.py
  ```
  * step 3:run
 ```
  test.py
  ```
  * step 4:run
 ```
  evaluate.py
  ```
 
 If you want to run the train sets from implementation 2, first go to the Models/LM/src directory.

* step 1:run 
 ```
  train.py config_11.json
  ```
  * step 3:run
 ```
  test.py config_11.json
  ```
  * step 4:run
 ```
  evaluate.py config_11.json
  ```
  
  ### Training your model (SubTask-B)
To create your Regression model, follow these steps

First go to the Models/REG/ directory.
* step 1: open config.json
* for BERT (full dataset): set "model" value to "BERT"
* for ERNIE (full dataset): set "model" value to "ERNIE"
* step 2:run
 ```
  train_reg.py
  ```
  * step 3:run
 ```
  test.py
  ```
  Go to Output directory
  * step 4:run
 ```
  rho.py
  ```
To run the third BERT model (using only train set), first go to the Models/LM/src directory.
* step 1:run
 ```
  train_reg.py config_11_reg.json
  ```
  * step 3:run
 ```
  test_reg.py config_11_reg.json
  ```
  Go to Output directory
  * step 4:run
 ```
  rho.py 

