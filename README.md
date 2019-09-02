# Stance Detection
This repository contains the source code associated to the stance detection dissertation submitted as part of the MSc Data Science with Machine Learning course at UCL.

## Structure

The repository contains key scripts for encoding inputs with BERT (`bert.py`) and additional utilities for training and evaluation (`utils.py`).

Two tutorial notebooks for data management (`data.examples.py`) and BERT encoding (`bert_examples.py`) are provided.

The remaining notebooks (`*.ipynb`) contain all that is necessary to reproduce the training and evaluation procedures detailed in the dissertation.

The data is contained in the `data/` repository; when new data is to be added, the associated interface should be added in `datasets.py`

## Tutorial

A self-contained tutorial is available at https://colab.research.google.com/drive/1EWKYyducE-TWPYk0z2fk7br4TZ18fttD.

## Notes

Huggingface's implementation of BERT is used, hence `pytorch_pretrained_bert` needs to be installed.
