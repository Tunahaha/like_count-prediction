# -*- coding: utf-8 -*-
"""
##import some packages
------------------------------------------------------------------------------
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import date, timedelta
import calendar
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb
import jieba
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import argparse
"""##argparse
------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser(prog='Dcard_HW.py', description='Dcard HW code')
parser.add_argument('--train_Set', '-t', default='intern_homework_train_dataset.csv', type=str,  help='train_Set data name')
parser.add_argument('--public_test_set', '-pu', default='intern_homework_public_test_dataset.csv', type=str,  help='Public_Test_Set data name')
parser.add_argument('--private_test_set', '-pr', default='intern_homework_private_test_dataset.csv', type=str,  help='Private_Test_Set data name')
parser.add_argument('--max_depth', '-m', default='9', type=int,  help='XGBoost parameter about max_depth')
parser.add_argument('--eta', '-e', default='0.05', type=float,  help='XGBoost parameter about learning rate')
parser.add_argument('--eval_metric', '-ev', default='mape', type=str,  help='XGBoost parameter about evaluation metric')
parser.add_argument('--num_round', '-n', default='300', type=int,  help='XGBoost parameter about iteration number')


"""##Hyperparameters
------------------------------------------------------------------------------
"""
args = parser.parse_args()
config={
    'Train_Set':args.train_Set,
    'Public_Test_Set':args.public_test_set,
    'Private_Test_Set':args.private_test_set,
}

param = {
    'max_depth': args.max_depth,
    'eta': args.eta,
    'objective': 'reg:squarederror',
    'eval_metric': args.eval_metric,
}

num_round = args.num_round
"""## Load Data"""

train=pd.read_csv(config['Train_Set'])
test=pd.read_csv(config['Public_Test_Set'])
Private_Test_set=pd.read_csv(config['Private_Test_Set'])

"""## function"""
def join_df(train, test,Private_set):
  df = pd.concat([train, test,Private_set], axis=0).reset_index(drop = True)
  features = [c for c in df.columns if c not in [TARGET_COL]]
  df[num_cols + ['like_count_24h']] = df[num_cols + ['like_count_24h']].apply(lambda x: np.log1p(x))
  return df, features
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True):
    seg_list =jieba.cut(text)
    text = " ".join(seg_list)
    return text

def calculate_correlation(row):
    like_counts = [row[f'like_count_{i}h'] for i in range(1, 7)]
    comment_counts = [row[f'comment_count_{i}h'] for i in range(1, 7)]
    return np.corrcoef(like_counts, comment_counts)[0, 1]
def cal_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
"""# Preprocess"""

TARGET_COL ='like_count_24h'
num_cols = ['like_count_1h','like_count_2h','like_count_3h','like_count_4h','like_count_5h','like_count_6h','comment_count_1h','comment_count_2h','comment_count_3h','comment_count_4h','comment_count_5h','comment_count_6h','forum_stats']
cat_cols = ['forum_id','author_id']
text_cols = ['title']
date_cols = ['created_at','date_only','time_only']

"""--------------------------------------------------"""
df, features = join_df(train,test,Private_Test_set)
train['set'] = 'train'
test['set']='test'
Private_Test_set['set'] = 'Private_Test_set'
df = pd.concat([train, test,Private_Test_set])

"""categories"""
print("----categories start----")
df = pd.get_dummies(df, columns=cat_cols)
df = df.fillna(0)
df[num_cols + ['like_count_24h']] = df[num_cols + ['like_count_24h']].apply(lambda x: np.log1p(x))

"""# Feature engineering"""
"""## time"""
print("----time start----")
df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
df['created_at_since_start'] = (df['created_at'] - df['created_at'].min()).dt.days
df['created_at_of_week'] = df['created_at'].dt.dayofweek
df['created_at_year'] = df['created_at'].dt.year
df['created_at_year_month'] = df['created_at'].dt.month
df['created_at_day'] = pd.to_datetime(df['created_at']).dt.day
df['created_at_hour'] = pd.to_datetime(df['created_at']).dt.hour




"""## title"""
print("----title start----")
df['title_len'] = df['title'].apply(lambda x: len(x))
df["clean_title"] = df["title"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, ))
df['clean_title_word_count'] = df["clean_title"].apply(lambda x: len(str(x).split(" ")))
df['clean_title_char_count'] = df["clean_title"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
df['clean_title_avg_word_length'] = df['clean_title_char_count'] / df['clean_title_word_count']
df["clean_title_sentiment"] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)


vectorizer = TfidfVectorizer(tokenizer=utils_preprocess_text, min_df=5, max_df=0.8, ngram_range=(1, 2), max_features=500)
title_tfidf_matrix = vectorizer.fit_transform(df['title'])
title_tfidf_df = pd.DataFrame(title_tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), title_tfidf_df], axis=1)


"""## like && comment"""
print("----like && comment start----")
for i in range(1, 6):
    df[f'like_count_diff_{i}_{i+1}h'] = df[f'like_count_{i+1}h'] - df[f'like_count_{i}h']
    df[f'comment_count_diff_{i}_{i+1}h'] = df[f'comment_count_{i+1}h'] - df[f'comment_count_{i}h']
for i in range(1, 7):
    df[f'like_count_ratio_{i}h'] = df[f'like_count_{i}h'] / (df[f'like_count_{i}h'].sum() + 1e-8)
    df[f'comment_count_ratio_{i}h'] = df[f'comment_count_{i}h'] / (df[f'comment_count_{i}h'].sum() + 1e-8)
for i in range(1, 7):
    df[f'like_comment_ratio_{i}h'] = df[f'like_count_{i}h'] / (df[f'comment_count_{i}h'] + 1e-8)
    df[f'like_comment_diff_{i}h'] = df[f'like_count_{i}h'] - df[f'comment_count_{i}h']
df['like_comment_corr'] = df.apply(calculate_correlation, axis=1)

"""split"""
train = df[df['set']=='train']
test = df[df['set']=='test']
train = train.drop('set', 1)
test = test.drop('set', 1)
Private_Test_set = df[df['set']=='Private_Test_set']
Private_Test_set = Private_Test_set.drop('set', 1)
features = [c for c in train.columns if c not in [TARGET_COL]]
cat_num_cols = [c for c in features if c not in ['title','created_at', 'clean_title']]
X = train[features]
y = train[TARGET_COL]
Private_Test_set=Private_Test_set.drop(TARGET_COL, 1)
X_train, X_val, y_train, y_val = train_test_split(train[cat_num_cols],y, test_size=0.2, random_state = 23)


"""outbound"""
y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
sample_weights = np.ones_like(y_train, dtype=np.float32)
#std2_outliers = (np.abs(y_train - y_train_mean) >= 2 * y_train_std) & (np.abs(y_train - y_train_mean) < 3 * y_train_std)
std2_outliers = (np.abs(y_train - y_train_mean) >= 2 * y_train_std)
#std3_outliers = np.abs(y_train - y_train_mean) >= 3 * y_train_std
sample_weights[std2_outliers] = 0.7
#sample_weights[std3_outliers] = 0.1


"""XGBoost"""
print("----training start----")
dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
dtest = xgb.DMatrix(X_val, label=y_val)
bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=5)
tes_matrix=xgb.DMatrix(test[cat_num_cols])
test_pre=bst.predict(tes_matrix)
test_pre = np.round(np.expm1(test_pre)).astype(int).ravel()
test_true=np.round(np.expm1(test[TARGET_COL])).astype(int)
#print(cal_mape(test_true,test_pre))

"""## Private_Test_set save"""

pri_tes_matrix=xgb.DMatrix(Private_Test_set[cat_num_cols])
pri_xgboost=bst.predict(pri_tes_matrix)
pri_xgboost = np.round(np.expm1(pri_xgboost)).astype(int).ravel()
Private_Test_pred = pd.DataFrame({'like_count_24h': pri_xgboost})
Private_Test_pred.to_csv('result.csv', index=False)
print("===over===")