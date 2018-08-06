# UOI-P18-2012

[![Travis](https://travis-ci.org/PoWWoP/UOI-P18-2012.svg)](https://travis-ci.org/PoWWoP/UOI-P18-2012)

## Introduction

The repository is an _**UN**OFFICIAL_ Keras implementation of the paper: [_Named Entity Recognition With Parallel Recurrent Neural Networks_](http://aclweb.org/anthology/P18-2012).

Several parallel Bi-LSTMs are used to extract features from [embeddings of words and characters](https://github.com/PoWWoP/keras_word_char_embd). The recurrent weights for calculating cell state vectors in LSTMs are compared and added to the final loss to make sure they extract different information from each other.

## Demo

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
python demo.py
```
