# MLOPS project description - Natural Language Processing with Disaster Tweets

Niels Tiben s203131 \
Denis Ghiletki s210714 \
Jakob Schneider s202432 \
Kristin Anett Remmelgas s203129 \
Wybren Meinte Oppedijk s203130
 
### Overall goal of the project
The goal of the project is to use natural language processing to solve a classification task of predicting whether a given tweet is about a real disaster or not.
### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
Since we chose a natural language processing problem, we plan to use the Transformers framework (https://github.com/huggingface/transformers).
### How to you intend to include the framework into your project
We plan on utilizing one of the strengths of the Transformers framework which is that it provides thousands of pretrained models to perform different tasks. We intend to start with using some of the pretrained models on our data and then see where to go from there. 
### What data are you going to run on (initially, may change)
We are using the Kaggle dataset Natural Language Processing with Disaster Tweets (https://www.kaggle.com/c/nlp-getting-started/overview). Each sample in the train and test set has the following information: a unique identifier, the text of a tweet, a keyword from that tweet (although this may be blank) and the location the tweet was sent from (may also be blank) and the training set also has a target value whether a tweet is about a real disaster (1) or not (0). The dataset was chosen because it is quite simple and straightforward which makes it a great dataset for getting started with Natural Language Processing. It also seems feasible to implement in such a short timeframe.
### What deep learning models do you expect to use
We intend to use pre-trained models due to limited time, and also train the model(s) additionally on our dataset.
