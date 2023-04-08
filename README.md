# like_count-prediction
This repository is for Dcard Data itern homework based on [XGBoost](https://github.com/dmlc/xgboost).

### Dependencies
-------
- xgboost
- mlxtend
- jieba
- textblob

You should ` pip install ` about those dependencies.

### XGBoost
-------
<img src="https://xgboost.ai/images/logo/xgboost-logo.svg" width=135/> 

XGBoost is an optimized distributed gradient boosting library designed to be highly ***efficient***, ***flexible*** and ***portable***.
It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.

### How to use?
-------
you can use `python Dcard_HW.py` on CMD

you just need to put your data in the same stage.
### Options
-------

Use `--train_Set` or `-t` means train_Set data name. 

default : intern_homework_train_dataset.csv

Use `--public_test_set` or `-pu` means Public_Test_Set data name.

default : intern_homework_public_test_dataset.csv

Use `--private_test_set` or `-pr` means Private_Test_Set data name.

default : intern_homework_private_test_dataset.csv

Use `--max_depth` or `-m` means XGBoost parameter about max_depth.

default : 7

Use `--eta` or `-e` means XGBoost parameter about learning rate.

default : 0.05

Use `--eval_metric` or `-ev` means you XGBoost parameter about evaluation metric.

default : mape

Use `--num_round` or `-n` means XGBoost parameter about iteration number.

default : 100

### Warning
-------
The Matrix is huge，you should have a lot of memory.
