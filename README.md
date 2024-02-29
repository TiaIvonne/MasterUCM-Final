## README


## How to use this project?

It's possible to explore jupyter notebook using this link: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TiaIvonne/MasterUCM-Final/HEAD)


Welcome to my final project to get a Master in Big Data and Business Analytics of the Universidad Complutense de Madrid. This journey begins with a deep study in database administration, statistics, programming languages (Python, R) and machine learning.

At the end of this journey I would be capable to combine different techniques of machine learning to solve a problem and offer a solution for a real world problem.

## Overview

A company named 'Supermercado S.A' has been developing a strong process of advertising and marketing due to the upcoming christmas season.

The main goal is to obtain a prediction which can indicate if the client will take a positive decision about an offer. Also we would like to get insights about collected data to elaborate a portfolio of customer’s profiles: this outcome can be used by the sales team for example to create email campaigns.

Dataset used for this project: https://www.kaggle.com/datasets/ahsan81/superstore-marketing-campaign-dataset

## Steps

The dataset had to be cleaned before moving on to subsequent steps.
New columns were created based on existing ones to check or test if these new features influenced the algorithms.


Different notebooks were created to run step by step according to the following itinerary:

* Data Cleaning: Analyzing EDA, data cleaning, removing duplicates, handling missing data, analyzing outliers and getting interesting insights from the dataset.
* Modeling: Once my dataset is ready to work we apply machine learning models for its first time. This is achieved by running a function which quickly compares different algorithms.

Then each algorithm is analyzed in detail with its respective tunning process:
* Logistic regression
* Random forest classifier
* Decision tree
* AdaBoost
* CatBoost
* LightGBM
* XGBoost
* Gradient Boosting

Each machine learning algorithm was measured in execution time, CPU usage and algorithm performance.
Finally a comparison table is created with the measurements of the different algorithms and a graph showing the features that have the most impact on the winning model.

For additonal investigation a few special notebooks are created to test automatic variable selection methods and improvements using the winning model, data standardization techniques, and the use of AutoML to try different machine learning techniques.

* Relevant Models: A special notebook is created with the results of the winning model.
* Standarization Techniques: A special notebook is created with the results of the data standardization techniques.
* Automatic Variable Selection: A special notebook is created with the results of the automatic variable selection methods.
* In addition, a python script was created for help with visualization of the results.(Generate graphs, tables, charts and confusion matrix)

## Challenges


It is crucial to find the best model to predict the sales of the Superstore. In this context a ‘model’ is a machine learning algorithm.

What makes a model better? we can compare different models and choose the one that performs better. My criteria was to choose a model that has a high accuracy and low execution time.

The dataset is highly imbalanced which represent a problem or a challenge. There are a lot of techniques for balancing data artificially. Since this is a a project with educational purposes I did not implement any technique to balance the data.

I wanted to understand different tradeoffs between multiple algorithms using their own features for dealing with this kind of dataset.

