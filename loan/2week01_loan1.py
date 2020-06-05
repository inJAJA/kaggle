#1. 데이터
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse


dataset = pd.read_csv('D:/Practice/kaggle/Loan_data.csv')

print(dataset.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500 entries, 0 to 499
Data columns (total 11 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   Loan_ID         500 non-null    object    # 대출한 고객의 고유 ID                              # index_col
 1   loan_status     500 non-null    object    # 이번 분석의 타겟 변수, 상환 여부를 나타냄            # y
 2   Principal       500 non-null    int64     # 고객이 대출받은 금액
 3   terms           500 non-null    int64     # 대출금 지급까지 걸린 시간
 4   effective_date  500 non-null    object    # 실제 계약 효과가 발휘하기 시작한 날짜
 5   due_date        500 non-null    object    # 대출금 납부 기한 날짜
 6   paid_off_time   400 non-null    object    # 고객이 은행에 모두 상환한 날짜, 시간                 # 100
 7   past_due_days   200 non-null    float64   # 고객이 은행에 대출금을 모두 상환하는데 걸린 기간      # 300
 8   age             500 non-null    int64     # 고객의 나이
 9   education       500 non-null    object    # 고객의 교육 수준
 10  Gender          500 non-null    object    # 고객의 성별
dtypes: float64(1), int64(3), object(7)
'''
'''
loan_status
0. PAIDOFF : 기한 내에 대출금 모두 상환
1. COLLECTION : data 수집 당시까지 미납 (연체)
2. COLLECTION_PAIDOFF : 기한은 지났지만 대출금 모두 상환
'''
# loan_status
print(dataset.loan_status.value_counts(dropna= False))
# PAIDOFF               300
# COLLECTION            100
# COLLECTION_PAIDOFF    100
# Name: loan_status, dtype: int64

print(type(dataset))               # <class 'pandas.core.frame.DataFrame'>
print(type([dataset]))             # <class 'list'>

                                     # AttributeError: 'str' object has no attribute 'loc', 
for dataset in [dataset]:            # dataframe을 list로 바꿔 줘야 한다.
    dataset.loc[(dataset['loan_status'] == 'PAIDOFF'), 'loan_status']= 0
    dataset.loc[(dataset['loan_status'] == 'COLLECTION'), 'loan_status'] = 1
    dataset.loc[(dataset['loan_status'] == 'COLLECTION_PAIDOFF'), 'loan_status'] = 2
    
print(dataset.loan_status.value_counts())
# 0    300
# 2    100
# 1    100
# Name: loan_status, dtype: int64

from keras.utils.np_utils import to_categorical
y = dataset['loan_status']
y = to_categorical(y)                 # 3중 분류



# Principal




# education
print(dataset.education.value_counts(dropna = False))
# college                 220       # 2년제 학사
# High School or Below    209       # 고등학교 또는 그 아래
# Bechalor                 67       # 4년제 학사
# Master or Above           4       # 석사 또는 그 이상
# Name: education, dtype: int64
print(type(dataset.iloc[0, 2]))       
for dataset in [dataset]:
    dataset.loc[(dataset['education'] == 'High School or Below'), 'education'] =  0
    dataset.loc[(dataset['education'] == 'college'), 'education'] =  1
    dataset.loc[(dataset['education'] == 'Bechalor'), 'education'] =  2
    dataset.loc[(dataset['education'] == 'Master or Above'), 'education'] =  3


# terms
print(dataset.terms.value_counts(dropna = False))
# 30    272
# 15    207
# 7      21
# Name: terms, dtype: int64
for dataset in [dataset]:
    dataset.loc[(dataset['terms'] == 7), 'terms'] = 0 
    dataset.loc[(dataset['terms'] == 15), 'terms'] = 1
    dataset.loc[(dataset['terms'] == 30), 'terms'] = 2


# gender
# one_hot_encoding필요

# effective_date
print(dataset.effective_date.value_counts(dropna=False))
# Name: terms, dtype: int64
# 9/11/2016    231
# 9/12/2016    148
# 9/10/2016     46
# 9/14/2016     33
# 9/13/2016     23
# 9/9/2016      15
# 9/8/2016       4
# Name: effective_date, dtype: int64


# 모두 날짜형으로 바꾸기
for i in range(len(dataset.values)):
    dataset.iloc[i, 4] = parse(dataset.iloc[i, 4])  # effective_date
    dataset.iloc[i, 5] = parse(dataset.iloc[i, 5])  # due_date
    
print(dataset['effective_date'].head())
print(dataset['due_date'].head())


# paid off time
dataset.loc['paid_off_time'] = dataset.loc['paid_off_time'].fillna(values = '0')

for i in range(len(dataset.values)):
    dataset.iloc[i, 6] = parse(dataset.iloc[i, 6])

print(dataset['paid_off_time'].head())




