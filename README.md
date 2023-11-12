# Like Count Prediction with XGBoost

This repository contains the code for predicting the like count of Dcard posts using the [XGBoost](https://github.com/dmlc/xgboost) machine learning library. This project was a part of the Dcard Data Intern homework.

## Dependencies
The following libraries are required to run this project:

- xgboost
- mlxtend
- jieba
- textblob

To install these dependencies, use the following command:

```sh
pip install xgboost mlxtend jieba textblob
```

## About XGBoost
![XGBoost Logo](https://xgboost.ai/images/logo/xgboost-logo.svg)

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solves many data science problems in a fast and accurate way.

## Usage

To use this project, run the following command in your terminal:

```sh
python Dcard_HW.py
```

Make sure your data is in the same directory as the script.

## Options

The script accepts several options:

- `--train_Set` or `-t`: Name of the training set data file. Default is `intern_homework_train_dataset.csv`.
- `--public_test_set` or `-pu`: Name of the public test set data file. Default is `intern_homework_public_test_dataset.csv`.
- `--private_test_set` or `-pr`: Name of the private test set data file. Default is `intern_homework_private_test_dataset.csv`.
- `--max_depth` or `-m`: XGBoost parameter for max_depth. Default is `9`.
- `--eta` or `-e`: XGBoost parameter for learning rate. Default is `0.05`.
- `--eval_metric` or `-ev`: XGBoost parameter for evaluation metric. Default is `mape`.
- `--num_round` or `-n`: XGBoost parameter for number of iterations. Default is `300`.

## Warning

Please note that the matrices used in this project are quite large. Make sure to have sufficient memory available.
