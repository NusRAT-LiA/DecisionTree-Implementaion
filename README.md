# Decision Tree Implementation from Scratch and with scikit-learn

This repository contains two implementations of the Decision Tree algorithm:
1. **Custom Decision Tree** built from scratch.
2. **scikit-learn Decision Tree** implementation for comparison.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Custom Decision Tree Class](#custom-decision-tree-class)
  - [Features](#features)
  - [Usage](#usage)
- [scikit-learn Decision Tree](#scikit-learn-decision-tree)
- [Comparison](#comparison)

## Introduction

This project demonstrates the implementation of a Decision Tree classifier using:
- A **custom implementation** from scratch, providing insight into how the algorithm works at a low level.
- The **scikit-learn** library, one of the most popular machine learning libraries, for ease of use and comparison.

Both implementations are tested on the **Iris dataset** to classify different species of Iris flowers based on their sepal and petal dimensions.

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `collections`

## Getting Started

### Clone this repository:

```bash
git clone https://github.com/NusRAT-LiA/DecisionTree-Implementaion.git
cd DecisionTree-Implementaion
pip install -r requirements.txt
python decisionTree1.py
python decisionTree2.py
```

## Dataset

The Iris dataset is used in this project. This dataset contains 150 samples of iris flowers, with 50 samples from each of three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each sample contains the following features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is the species of the iris flower.

## Custom Decision Tree Class

### Features

The custom decision tree implementation includes:

- Building a decision tree from scratch without using any external libraries.
- Handling continuous data by finding the optimal split points.
- Using Entropy and Information Gain for splitting the data.

### Usage

You can run the custom decision tree implementation by executing the following command:

```bash
python decisionTree1.py
```
The script includes:

    Training the decision tree on a training set.
    Predicting the outcomes for the test set.
    Printing out the confusion matrix and accuracy of the model.
    
## scikit-learn Decision Tree

For comparison, the scikit-learn implementation of a Decision Tree classifier is also provided. You can run this version by executing:

```bash
python decisionTree2.py
```
This version leverages the DecisionTreeClassifier from scikit-learn to build a classifier with minimal code.

## Comparison

Both implementations achieve almost similar results when tested on the Iris dataset. The custom implementation showcases how the Decision Tree algorithm works under the hood, while the scikit-learn version is easier to implement and maintain.

### Key differences:

- The custom implementation allows you to understand how a Decision Tree is built from scratch, including how splits are made and how nodes are created.
- The scikit-learn implementation is more optimized for performance and ease of use.

| Implementation   | Accuracy | Confusion Matrix                  |
|------------------|----------|-----------------------------------|
| **Custom**       | 93.33 %  | [[18, 0, 0], [0, 21, 2], [0, 2, 17]]|
| **scikit-learn** | 100.00%  | [[10, 0, 0], [0, 9, 0], [0, 0, 11]]|






