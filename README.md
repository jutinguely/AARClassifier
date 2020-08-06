# Android App Review Classification

## Description
> In this task we want to build a model which gives the best accuracy in predicting the score of a given Google App review. 
> The score (or number of stars) of a review goes from bad to amazing and is represented by a value from [1,2,3,4,5] (1 star being bad and 5 stars amazing).

The documented implementation can be found in `main.py` and the hyper-parameters used for the experiment in `roberta_android.json`.

#### Dataset

> The AAR dataset contains 752'936 entries. Each entries has a `review`, a summary, a `score`, and some unused meta-information.

The dataset is not *balanced* (see figure): there are 44k reviews with a score of 2 and 380k with a score of 5. 

It contains cased characters (A-Z). As the user probably take the task of writing reviews a bit more seriously than posting on social medias,
the reviews are already pretty *cleaned* and contain few slang words.

![](coarse_analysis.png)

#### Trivial and unfair solution

A trivial solution would be to assign always 5 stars. That would give an accuracy of `51%`. Our goal is to be better than this 
and at the same time we want to build a model which is consistent. Therefore we want to have a balanced dataset in case a user
wants mostly to predict reviews with a score of 2. (Note: in this case the prediction would be really bad if we don't balance the dataset)

#### Model and Methods 

##### RoBERTa

RoBERTa or Robustly optimized BERT approach is a framework from Facebook that aims at improving performance of BERT based models 
by offering a new architecture on top of the model.  
It carefully looks at the impact of the parameters and the training data size. 
It uses its own classifier on top of the model (RobertaClassificationHead), which is composed of two Dense linear layers with a tanh activation function. 
Over Bert and XLNet, it claims that the performance can be improved by: 


- training the model longer,
- having bigger batch sizes over more data,
- removing the next sentence prediction,
- training on longer sequences, and
- dynamically changing the masking patterns applied to the dataset.


I augmented the large pretrained model from RoBERTa (roberta-large) with the AAR datasets. 
Each class has **44k** reviews. The training rate is 0.9 (198k) and validation rate 0.1 (22k).
I used the default RobertaTokenizer to encode the dataset and ran the training with a batch size of 8 learning rate of 10^{-6}, and 1-3 epoch.


##### Preventing overfitting

> - In theory: high bias -> low variance -> *underfitting* vs. low bias -> high variance -> *overfitting*
> - In practice: I used early stopping and took reasonable batch sizes (bs=8).
> - As the validation computes the accuracy e.g. the variation with real value `sum(expected_i == predicted_i)/len(validation)`,
> then if the validation accuracy is good, we can pretend that the trained bias is the best for our hyper-parameters. 


#### Result
> With transfer learning on **roberta-large**, I achieved an accuracy of: `` 80 ``

#### Further improvements
> - Tuning the model by finding the best hyper-parameters with a CVGridSearch,
> - More training time often implies a better accuracy,
> - Testing out XLnet or some other state-of-art models,
> - Include correlation such as between lengths of reviews and scores (bad reviews tend to have longer lengths).

## Environment and Commands

> - simulations done on a *Nvidia GeForce GTX 1080 Ti GPU* (cuda 10.1 and python 3.7.6)
> - installation: `pip install -r requirements.txt`
> - run `python main.py roberta_android.json`
> - run `python analysis.py` (generate )


### Links
- [RoBERTa paper](https://arxiv.org/pdf/1907.11692.pdf)
- [RoBERTa Github](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
- [dataset source](https://www.google.com/url?sa=D&q=http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz&ust=1596803940000000&usg=AOvVaw04CuHBOYoLzp-Xsq9tGL3S&hl=en)
