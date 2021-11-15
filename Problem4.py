from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("park.csv", encoding = "cp949")
#df = pd.read_csv("park.csv")

#df.head()


x = df[['일반이용자', '운동시설', '자전거']]

#광나루=>1 잠실 =>2 뚝섬=>3 잠원=>4 반포=>5 이촌=>6 여의도=>7 양화=>8 망원=>9 난지=>10 강서=>11
y = df[['구분']]


#데이터 학습시키기
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train.values, y_train.values)

#예측하기 (shell에서 my_predict입력 시 약 5.9의 값, 즉 이촌에 가까운 값이 나옴)
my_park = [[1500000, 20000, 500000]]
my_predict = mlr.predict(my_park)
