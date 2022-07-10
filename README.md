# Wine Quality Detection
Authors: Samuele Lo Truglio, Mario Gabriele Palmeri

## What is it?
This project is a part of the Machine Learning and Pattern Recognition course at the Politecnico of Torino.

The goal of this project is to predict the quality (good or bad) of wine based on its properties.

## Dataset
The dataset used is a simplified version of [wine quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). 

The original dataset consists of 10 classes (quality 1 to 10).

For the project, the dataset has been binarized, collecting all
wines with low quality (lower than 6) into class 0, and good quality (greater than 6) into class 1.

Wines with quality 6 have been discarded to simplify the task.

The dataset contains both red and white wines (originally separated, they have been merged).

There are 11 features, that represent physical properties of the
wine.

The dataset has been split into Train and Evaluation (Test) data. The first is used to train and validate the model, while the second is used to evaluate the model.

## Citation
*P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.*
