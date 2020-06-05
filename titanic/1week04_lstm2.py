# 데이터 불러오기 
import numpy as np
import pandas as pd

x_train = pd.read_csv('E:/Bit Camp/titanic/train.csv')
x_pred = pd.read_csv('E:/Bit Camp/titanic/test.csv') 

print(x_train.shape)       # (891, 12)
print(x_pred.shape)        # (418, 11)

print(x_train.info())
"""
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
 5   Age          714 non-null    float64    # 177개 data 빠짐
 6   SibSp        891 non-null    int64 

 7   Parch        891 non-null    int64 

 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object     # 687개 data 빠짐
 11  Embarked     889 non-null    object     #   2개 data 빠짐
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
"""
print(x_pred.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  418 non-null    int64
 1   Pclass       418 non-null    int64
 2   Name         418 non-null    object
 3   Sex          418 non-null    object
 4   Age          332 non-null    float64     # 86개 data 빠짐
 5   SibSp        418 non-null    int64
 6   Parch        418 non-null    int64
 7   Ticket       418 non-null    object
 8   Fare         417 non-null    float64     #   1개 data 빠짐
 9   Cabin        91 non-null     object      # 327개 data 빠짐
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
"""

'''
# 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Pie chart 함수 만들기
def pie_chart(feature):
    feature_ratio = x_train[feature].value_counts(sort = False)   # 정렬 기준 없음
    print(feature , 'ratio: ',feature_ratio)    
    feature_size = feature_ratio.size
    print(feature, 'size: ', feature_size)      
    feature_index = feature_ratio.index
    print(feature, 'index: ',feature_index)    
    survived = x_train[x_train['Survived']==1][feature].value_counts()
    dead = x_train[x_train['Survived']==0][feature].value_counts()

    plt.plot(aspect = 'equal')
    plt.pie(feature_ratio, labels = feature_index, autopct = '%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    print('------')

    for i , index in enumerate(feature_index):   # enumerate : (인덱스 번호, index)튜플 형태로 반환
        print(i,"",feature_size)
        plt.subplot(1, feature_size, i+1, aspect = 'equal')
        plt.pie([survived[index], dead[index]], labels = ['Survived','Dead'], autopct = '%1.1f%%')
        plt.title(feature_index[i])
        
    plt.show()


pie_chart('Sex')
# Sex ratio:  male  577, female  314 
# Name: Sex, dtype: int64
# Sex size:  2
# Sex index:  Index(['male', 'female'], dtype='object') 
# ------
# 0  2
# 1  2
pie_chart('Pclass')
# Pclass ratio:  1    216 , 2    184 , 3    491
# Name: Pclass, dtype: int64
# Pclass size:  3
# Pclass index:  Int64Index([1, 2, 3], dtype='int64')
# ------
# 0  3
# 1  3
# 2  3
pie_chart('Embarked')
# Embarked ratio:  S    644, Q     77, C    168
# Name: Embarked, dtype: int64
# Embarked size:  3
# Embarked index:  Index(['S', 'Q', 'C'], dtype='object')
# ------
# 0  3
# 1  3
# 2  3


# Bar chart 함수 만들기
# SibSp = # of siblings and space
# Parch = # of parents and children
def bar_chart(feature):
    survived = x_train[x_train['Survived']==1][feature].value_counts()
    dead = x_train[x_train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead]) 
    df.index = ['Survived','Dead']
    df.plot( kind = 'bar', stacked = True, figsize = (10, 5))
    plt.show()

bar_chart('SibSp')
bar_chart('Parch')
'''


# x 데이터 전처리를 위해 합쳐준다.
x = [x_train, x_pred]              
print(type(x))                          # list형태

"""
선택한 특성 : Name, Sex, Embarked, Age, SibSp, Parch, Fare, Pclass
제외 : Ticket, Cabin  - 아직 의미를 찾지 못함
"""
# Name feature 
for dataset in x:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
# ’ ([A-Za-z]+).‘는 정규표현식
# 공백으로 시작하고, .으로 끝나는 문자열을 추출하는데 사용
print(pd.crosstab(x_train['Title'], x_train['Sex']))
# Sex       female  male
# Title
# Capt           0     1
# Col            0     2
# Countess       1     0
# Don            0     1
# Dr             1     6
# Jonkheer       0     1
# Lady           1     0
# Major          0     2
# Master         0    40
# Miss         182     0
# Mlle           2     0
# Mme            1     0
# Mr             0   517
# Mrs          125     0
# Ms             1     0
# Rev            0     6
# Sir            0     1

# 흔하지 않는 Title은 Other로 대체하고 중복 표현은 통일 시킨다.
for dataset in x:                                                 # test값
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Countess','Don','Dona','Dr',
                                                'Jonkheer','Lady','Major','Rev','Sir'],'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mr')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

print(x_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# 추출한 Title을 데이터 String Data로 변형
for dataset in x:
    dataset['Title'] = dataset['Title'].astype(str)


# Sex feature
# male과 female로 나눠진 데이터를 String data로 변형
for dataset in x:
    dataset['Sex'] = dataset['Sex'].astype(str)


# Embarked Feature
print(x_train.Embarked.value_counts(dropna=False))
# S      644
# C      168
# Q       77
# NaN      2
# Name: Embarked, dtype: int64
# train데이터에서 NaN인 값이 존재한다.

# 대부분의 데이터가 S임으로 빠진 NaN값을 S로 채워주고 String Data로 변형시켜 준다.
for dataset in x:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)


# Age feature 
# Age feature에도 NaN값이 존재하는데 나머지 모든 승객의 평균 나이를 집어 넣겠다.
for dataset in x:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace =True)    # NaN값에 평균 넣음
    dataset['Age'] = dataset['Age'].astype(int) 
    x_train['AgeBand'] =pd.cut(x_train['Age'],5) 

print(x_train[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean())   # AgeBand의 생존률
#          AgeBand  Survived
# 0  (-0.08, 16.0]  0.550000
# 1   (16.0, 32.0]  0.344762
# 2   (32.0, 48.0]  0.403226
# 3   (48.0, 64.0]  0.434783
# 4   (64.0, 80.0]  0.090909
 
# Age에 들어있는 값을 위해서 구한 구간에 속하도록 바꿔준다.
for dataset in x:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <=48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <=64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64 , 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0:'Child', 1:'Young', 2:'Middle', 3:'Prime', 4:'Old'}).astype(str)


# Fare feature
# test데이터 중에 NaN값이 하나 존재한다. 
# Pclass와 Fare가 어느정도 연관성이 있는거 같아 
# 빠진 Fare데이터에 Pclass를 가진 사람들의 평균 Fare값을 넣어 주었다.
print(x_train[['Pclass','Fare']].groupby(['Pclass'], as_index=False).mean())
print(" ")
print(x_pred[x_pred['Fare'].isnull()]['Pclass'])
#    Pclass       Fare
# 0       1  84.154687
# 1       2  20.662183
# 2       3  13.675550
# " "
# 152    3
# Name: Pclass, dtype: int64

# 누락된 데이터의 Pclass는 3이고, 
# train데이터에서 Pclasss가 3인 사람들의 평균 Fare가 13.675550임으로 이 값을 넣어주자
for dataset in x:
    dataset["Fare"] = dataset['Fare'].fillna(13.676)


# SibSp & Parch feature (Family)
# 형제, 자매, 배우자, 부모님, 자녀의 수가 많을 수록 생존하는 경우가 많았다. 
# 두개의 feature를 합쳐서 Family라는 feature를 만들자
for dataset in x:
    dataset['Family'] = dataset['Parch'] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)

print(dataset.columns)
# Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',       
# 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'Family'],dtype='object')


# 특성 추출 및 나머지 전처리
features_drop = ['Name', 'Ticket','Cabin','SibSp','Parch']       # Drop시킬 column들
x_train = x_train.drop(features_drop, axis =1)                       # train data에서 drop
x_pred = x_pred.drop(features_drop, axis =1)                         # test data에서 drop
x_train = x_train.drop(['PassengerId', 'AgeBand'], axis = 1)
x_pred1 = x_pred.drop(['PassengerId'], axis =1)
print(x_train.shape)     # (891, 8)
print(x_pred.shape)      # (418, 7)


# One_Hot_Encoding
# get_dummies를 사용하면 문자열 특성만 인코딩되며 숫자 특성은 바뀌지 않음
# series를 str형태로 변형후 get_dummies 적용
# df_dummies = pd.get_dummies(df, columns=['숫자 특성', 'factor형 특성']) # get_dummies 적용시 columns 지정
x_train = pd.get_dummies(x_train)
x_pred1 = pd.get_dummies(x_pred1)

y_train = x_train['Survived']
x_train = x_train.drop(['Survived'], axis =1)
print(x_train.shape)      # (891, 18)
print(x_pred1.shape)       # (418, 18)
print(y_train.shape)      # (891, )


from sklearn.preprocessing import StandardScaler # MinMax보다 나은듯
scaler = StandardScaler()
scaler.fit(x_train + x_pred1)
x_train = scaler.transform(x_train)
x_pred1 = scaler.transform(x_pred1)


# x_train = x_train.values.reshape(891, 18, 1)
# x_pred1 = x_pred1.values.reshape(418,18, 1)
x_train = x_train.reshape(891, 18, 1)
x_pred1 = x_pred1.reshape(418,18, 1)
print(x_train.shape)      
print(x_pred1.shape)      


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.8,
                                                   random_state =30)


# model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(500, input_shape = (18,1 ), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(180, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(160, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(120, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
# model.add(Dense(60, activation = 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))

# earlystopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, verbose =1)

# compile, fit
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs =200, batch_size = 32,
                validation_split = 0.2, verbose = 2,
                callbacks = [es])

# evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 30)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_pred1)

print(y_pred.shape)                               # (418, 1)
y_pred = y_pred.reshape(418, )

submission = pd.DataFrame({
    "PassengerId": x_pred["PassengerId"],
    "Survived": y_pred
})

submission.to_csv('submission_lstm2.csv', index = False)
'''
loss :  0.5556808080753135
acc :  0.8044692873954773
'''