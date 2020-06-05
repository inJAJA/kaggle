"""kaggle datasets download <데이터 명> -p <"경로">"""
import pandas as pd
import numpy as np

# train = pd.read_csv('C:/Users/bitcamp/.kaggle/titanic/train.csv',engine='python',encoding='euc-kr')
# test = pd.read_csv('C:/Users/bitcamp/.kaggle/titanic/test.csv')

train = pd.read_csv('D:/kaggle/titanic/train.csv')
test = pd.read_csv('D:/kaggle/titanic/test.csv')


print(train.head())         # default= 5행 가져온다.

print('train data shape: ', train.shape)    # (891, 12)
print('test data shape: ', test.shape)      # (418, 11)
print('-----------[train information]-------------')
print(train.info())
'''
 -----------[train information]-------------
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
print('-----------[test information]-------------')
print(test.info())
'''
-----------[test information]-------------
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
print(train)                                # [891 rows x 12 columns] : header = 0,    index_col = None
                                            # [892 rows x 12 columns] : header = None, index_col = None
                                            # [891 rows x 11 columns] : header = 0,    index_col = 0 



# 데이터 분포
import matplotlib.pyplot as plt
import seaborn as sns                        # 향상된 데이터 시각화를 위해 만들어진 python라이브러리
sns.set()                                    # setting seaborn default for plots

# pie_chart 만드는 함수
def pie_chart(feature):                                     # .unique() : 유일한 값 찾기
    feature_ratio = train[feature].value_counts(sort=False) # .value_counts(): 유일한 값별 개수
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    
    plt.plot(aspect ='auto')
    plt.pie(feature_ratio, labels = feature_index, autopct='%1.1f%%')  # autopct : 파이조각 전체 대비 백분율
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    
    for i, index in enumerate(feature_index):                      # enumerate : 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
        plt.subplot(1, feature_size + 1, i+1, aspect='equal')      # aspect = 'eaual' : # x와 y갑의 크기을 동일하게 잡는다.
        plt.pie([survived[index],dead[index]], labels = ['Survivied', 'Dead'], autopct ='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()
        
# 성별에 따른 생존률        
pie_chart('Sex')
# 사회적 등수에 따른 생존률
pie_chart('Pclass')
# 탑승장소에 따른 생존률
pie_chart('Embarked')

"""bar chart"""
#SibSp = # of siblings and space
#Parch = # of parents and children

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize=(10,5))
    plt.show()

bar_chart("SibSp")

bar_chart("Parch")


"""
선택한 특성 : Name, Sex, Embarked, Age, SibSp, Parch, Fare, Pclass
제외 : Ticket, Cabin  - 아직 의미를 찾지 못함
"""
# 데이터 전처리 : 전체 데이터로 한번에 처리하기 위해서 합쳐줌
train_and_test = [train, test]


# Name feature 
# 이름에서 제공되는 승객들의 이름에 존재하는 Title('Hekkinen','Miss','Lania')로 
# 성별, 나이대, 결혼 유무를 알 수 있음
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')
# ’ ([A-Za-z]+).‘는 정규표현식
# 공백으로 시작하고, .으로 끝나는 문자열을 추출하는데 사용

print(train.head(5))                     # (5, 13)


# 추출한 타이틀을 가진 사람은 몇명인지 성별과 함께 표현
print(pd.crosstab(train['Title'], train['Sex']))
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
for dataset in train_and_test:                                         # test값
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Countess','Don','Dona','Dr',
                                                'Jonkheer','Lady','Major','Rev','Sir'],'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mr')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()) 
#     Title  Survived           # .groupby() : 집단, 그룹별로 데이터 요약 집계
# 0  Master  0.575000
# 1    Miss  0.702703
# 2      Mr  0.158301
# 3     Mrs  0.792000
# 4   Other  0.347826

# 추출한 Title을 데이터 학습하기 알맞게 String Data로 변형
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)


# Sex feature
# male과 female로 나눠진 데이터를 String data로 변형
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)


# Embarked Feature
# train데이터에서 NaN인 값이 존재한다.
# .value_counts(dropna = False) : NaN을 포함한 개수의 총합을 세줌
print(train.Embarked.value_counts(dropna=False))
# S      644
# C      168
# Q       77
# NaN      2
# Name: Embarked, dtype: int64

# 대부분의 데이터가 S임으로 빠진 NaN값을 S로 채워주고 String Data로 변형시켜 준다.
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)


# Age feature 
# Age feature에도 NaN값이 존재하는데 나머지 모든 승객의 평균 나이를 집어 넣겠다.
# 연속적인 numeric data를 처리하는 방법이 여러개 존재하는데, 그중 Binning을 사용하겠다.
# Binning : 여러 종류의 데이터에 대해 범위를 지정해주거나 , 
#           카테고리를 통해 이전보다 작은 수의 그룹으로 만드는 것
#          -> 단일성 분포 왜곡은 막을 수 있다.
#          -> But, 이산화를 통한 데이터의 손실이라는 단점 존재

# pd.cut()을 통해 같은 길이의 구간을 가지는 다섯개의 그룹을 만들겠다.
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace =True)    # NaN값에 평균 넣음
    dataset['Age'] = dataset['Age'].astype(int)                    # data type을 int로 지정
    train['AgeBand'] =pd.cut(train['Age'],5)                       # 같은 구간을 가지는 다섯개의 그룹으로 자름

print(train[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean())   # AgeBand의 생존률
#          AgeBand  Survived
# 0  (-0.08, 16.0]  0.550000
# 1   (16.0, 32.0]  0.344762
# 2   (32.0, 48.0]  0.403226
# 3   (48.0, 64.0]  0.434783
# 4   (64.0, 80.0]  0.090909


# Age에 들어있는 값을 위해서 구한 구간에 속하도록 바꿔준다.
for dataset in train_and_test:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <=48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <=64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64 , 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0:'Child', 1:'Young', 2:'Middle', 3:'Prime', 4:'Old'}).astype(str)
# 여기서 Age를 numeric이 아닌 String형식으로 넣어주었는데 숫자에 대한 경향성을 가지고 싶지 않아서 그럼


# Fare feature
# test데이터 중에 NaN값이 하나 존재한다. 
# Pclass와 Fare가 어느정도 연관성이 있는거 같아 
# 빠진 Fare데이터에 Pclass를 가진 사람들의 평균 Fare값을 넣어 주었다.
print(train[['Pclass','Fare']].groupby(['Pclass'], as_index=False).mean())
print(" ")
print(test[test['Fare'].isnull()]['Pclass'])
#    Pclass       Fare
# 0       1  84.154687
# 1       2  20.662183
# 2       3  13.675550
# " "
# 152    3
# Name: Pclass, dtype: int64

# 누락된 데이터의 Pclass는 3이고, 
# train데이터에서 Pclasss가 3인 사람들의 평균 Fare가 13.675550임으로 이 값을 넣어주자
for dataset in train_and_test:
    dataset["Fare"] = dataset['Fare'].fillna(13.676)

print(dataset['Fare'])

# Age에서 했던 것 처럼 Fare도 Binning해주자
for dataset in train_and_test:
    train['Fare'] = train['Fare'].astype(int)
    train['FareBand'] =pd.cut(train['Fare'],5)                       # 같은 구간을 가지는 다섯개의 그룹으로 자름

print(train[['FareBand','Survived']].groupby(['FareBand'], as_index = False).mean())   # AgeBand의 생존률

for dataset in train_and_test:
    dataset.loc[dataset['Fare'] <= 7.845, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.845)&(dataset['Fare']<= 10.5), 'Fare'] =1
    dataset.loc[(dataset['Fare'] > 10.5)&(dataset['Fare']<= 21.679), 'Fare'] =2
    dataset.loc[(dataset['Fare'] > 21.679)&(dataset['Fare']<= 39.688), 'Fare'] =3
    dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

# SibSp & Parch feature (Family)
# 형제, 자매, 배우자, 부모님, 자녀의 수가 많을 수록 생존하는 경우가 많았다. 
# 두개의 feature를 합쳐서 Family라는 feature를 만들자
for dataset in train_and_test:
    dataset['Family'] = dataset['Parch'] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)


# 특성 추출 및 나머지 전처리
# 이용할 Feature에 대해서는 전치리가 되었으니 학습시킬 때 제외시킬 fsature들을 Drop시키자
features_drop = ['Name', 'Ticket','Cabin','SibSp','Parch']       # Drop시킬 column들
train = train.drop(features_drop, axis =1)                       # train data에서 drop
test = test.drop(features_drop, axis =1)                         # test data에서 drop
train = train.drop(['PassengerId', 'AgeBand','FareBand'], axis = 1)
print('--------train.head---------')
print(train.head())
#    Survived  Pclass  ... Title Family        
# 0         0       3  ...    Mr      1        
# 1         1       1  ...   Mrs      1        
# 2         1       3  ...  Miss      0        
# 3         1       1  ...   Mrs      1        
# 4         0       3  ...    Mr      0  
# [5 rows x 8 columns]      
print(test.head())
#    PassengerId  Pclass  ... Title Family     
# 0          892       3  ...    Mr      0     
# 1          893       3  ...   Mrs      1     
# 2          894       2  ...    Mr      0     
# 3          895       3  ...    Mr      0     
# 4          896       3  ...   Mrs      2     
# [5 rows x 8 columns]


# One_Hot_Encoding
train =pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis = 1)
test_data = test.drop('PassengerId', axis =1 ).copy()

print(train_data.head())

print(train_data.shape)                           # (891, 18)
print(test_data.shape)                            # (418, 18)


# model구성
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle

# train_data 섞기
train_data, train_label = shuffle(train_data, train_label, random_state = 5)

# 함수 사용하여 fit, predict하기
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label)*100, 2 )
    print('Accuracy : ', accuracy, "%")
    return prediction


# Logistic Regression
# : 선형 or 이진 분류을 위한 단순하고 강력한 모델
# : 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측
log_pred = train_and_test(LogisticRegression())

# SVM
# : 서포트 벡터 머신(SVM: Support Vector Machine)
# : 결정 경계(Decision Boundary), 즉 분류를 위한 기준 선을 정의하는 모델
# : 분류과제에 사용
svm_pred = train_and_test(SVC())

# kNN
# : K-최근접 이웃 알고리즘 (K - Nearest Neighbors)
# : 특정공간내에서 입력과 제일 근접한 k개의 요소를 찾아,
# : 더 많이 일치하는 것으로 분류하는 알고리즘입니다.
kNN_pred = train_and_test(KNeighborsClassifier())

# Random Forest
# : ensemble(앙상블) machine learning 모델
# : decision tree를 형성하고 새로운 데이터 포인트를 각 트리에 동시에 통과시키며, 
# : 각 트리가 분류한 결과에서 투표를 실시하여 가장 많이 득표한 결과를 최종 분류 결과로 선택
rf_pred = train_and_test(RandomForestClassifier())

# Navie Bayes
nb_pred = train_and_test(GaussianNB())

"""결과값
Accuracy :  82.94 %
Accuracy :  83.5 %
Accuracy :  85.3 %
Accuracy :  88.66 %
Accuracy :  79.8 %
"""
### submission 하기 ###
# 제일 높은 결과값인 Random Forest 모델 선택
submission = pd.DataFrame({
    "PassngerId": test["PassengerId"],
    "Survived": rf_pred
})

submission.to_csv('submission_rf.csv', index = False)
# 여기서 만든 csv파일을 kaggle에 업로드하면 결과가 나타난다.
