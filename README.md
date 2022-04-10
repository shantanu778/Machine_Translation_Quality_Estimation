# Natural Language Processing
This is the repository for the final project of our NLP course, which involves assess how the estimated quality of translated instances affects the performance of models trained on them. To test this we created several experiments, which involves training BERT using a serie of different train sets.

<!-- GETTING STARTED -->
## Getting Started

Before you can start creating your models we firstly need to make sure the datasets we utilised for training our models are present in the Data folder.

### Prerequisites 

Our train set and dev set are the huggingface train set and dev set taken from [this page](https://huggingface.co/datasets/GroNLP/ik-nlp-22_transqe) and our test set are the SICK.txt and the SICK_NL.txt. The train and dev set needs to be saved as train.txt and dev.txt in the Data folder, while the SICK datasets can maintain their original name  


### Effects of using different QE intervals on
model performance
If you want to run the an experiment follow these steps:

* step 1: open config.json
* The DA interval can be set by adjusting"threshold" value, which is the upper limit of the DA score range
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
 
 To create a baseline 

* step 1: open config.json
* Set the value of translation to english
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
  
  ### Trade-off between data quantity and data quality

If you want to run the an experiment follow these steps testing a training model using train sets of different sizes, follow these steps:

* step 1: open config_quantity.json
*  use the keys "train_size" and "dev_size" to set the sizes of the train and dev set
* step 2:run
 ```
  train_quantity.py
  ```
  * step 3:run
 ```
  test.py
  ```

  * step 4:run
 ```
  evaluate.py
  ```
To use differen quality threshold, follow these steps:
* step 1: open config.json
* set "task" to 2
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

