import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train = pd.read_csv('D:/data/kaggle/titanic/train.csv')
test = pd.read_csv('D:/data/kaggle/titanic/test.csv')

print(train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
'''
print(test.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  418 non-null    int64
 1   Pclass       418 non-null    int64
 2   Name         418 non-null    object
 3   Sex          418 non-null    object
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64
 6   Parch        418 non-null    int64
 7   Ticket       418 non-null    object
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object
 10  Embarked     418 non-null    object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
'''

# PassengerId data로 사용 X 
# Survived = y값
# Plcass  :  ticket class
# Name 통일화    : 원핫 인코딩 필요
# Sex  : 남, 여 : 원핫 인코딩 필요
# age : 결칙치 존재
# Sibsp, parch와 관계
# ticket : 영향이 있을까?, 뒤에 숫자만 빼보자
# fare : 운임료
# cabin: 결측치가 너무 많다.
# Embarked : 탑승장소

dataset = [train, test]

# Name
for data in dataset:                                         # i = [train, test]
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.')   # (A-Z, a-z)이 이어지고 .으로 끝나는 문자
print(pd.crosstab(train['Title'], train['Sex']))
'''
None
Sex       female  male
Title
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
'''
print(pd.crosstab(test['Title'], test['Sex']))   # pd.crosstab(index, columns) : 교차표 형성
'''
Sex     female  male
Title
Col          0     2
Dona         1     0
Dr           0     1
Master       0    21
Miss        78     0
Mr           0   240
Mrs         72     0
Ms           1     0
Rev          0     2
'''


for dataset in dataset:                                         # test값
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Countess','Don','Dona','Dr',
                                                'Jonkheer','Lady','Major','Rev','Sir'],'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mr')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    


# Age
print(train['Age'].mean())   # 29.69911764705882

for data in dataset:
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())  # trian'Age', test'Age'의 평균
    dataset['Age'] = dataset['Age'].astype(int)                    # train, test의 'Age'를 int형으로 



# SibSp
# Parch

train['Family'] = train['Parch'] + train['SibSp']
train['Family'] = train['Family'].astype(int)

test['Family'] = test['Parch'] + test['SibSp']
test['Family'] = test['Family'].astype(int)
   

# tickect
train['Ticket'] = train['Ticket'].astype(str)
train['Ticket_num'] = train.Ticket.str.extract(r'(\d{4,6})')
train['Ticket_num'] = train['Ticket_num'].astype(float)

test['Ticket'] = test['Ticket'].astype(str)
test['Ticket_num'] = test.Ticket.str.extract(r'(\d{4,6})')
train['Ticket_num'] = train['Ticket_num'].astype(float)

print(train['Ticket_num'])

# cabin
train_cabin = train['cabin']
a = train['cabin'].dropna().copy()
b = train['cabin'].isnull().copy()
