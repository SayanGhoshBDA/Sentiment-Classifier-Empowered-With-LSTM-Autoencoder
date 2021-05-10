# Sentiment-Classifier-Empowered-With-LSTM-Autoencoder
This repository has been created as a part of my term project.  The motto of this work is to build an RNN classifier combined with an autoencoder such that the task of reconstruction via autoencoder helps in better classification.


### Contents:
1.  README.md
2.  Presentation.pdf
3.  Report.pdf
4.  'code' folder containing all relevent python and shell scripts


### Requirements:
1.  Python (version=3.6.9)
2.  Pytorch (cuda version >= 1.7.0)
3.  torchtext
4.  gdown


### Steps to train the model:
1.	Clone this repository: <br />
    ```git clone https://github.com/SayanGhoshBDA/Sentiment-Classifier-Empowered-With-LSTM-Autoencoder.git```
2.	Enter into the directory named 'code': <br />
    ```cd Sentiment-Classifier-Empowered-With-LSTM-Autoencoder/code/```
3.	Run the shell script named 'Apriori.sh': <br />
    ```bash Apriori.sh```
4.	Finally run the 'Train_and_Validate.py' to train the model and save the model weights: <br />
    ```python Train_and_Validate.py```
5.  Next run the 'Evaluate.py' to check the performance of the model on test dataset: <br />
    ```python Evaluate.py```

<ins>Note</ins>: Step 3 works in linux system.  For windows system, please manually run each of the commands inside the 'Apriori.sh' file.

