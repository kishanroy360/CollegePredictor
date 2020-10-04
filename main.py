import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('Resources/Admission_Predict.csv')
# print(df.describe())
df.drop(labels='Serial No.', axis=1, inplace=True)
# print(df.describe())

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
# print(X)
# Gre Toefl Uni SOP LOR CGPA Research
# print(Y)

X, Y = shuffle(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, Y_train)
pickle.dump(model, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

Y_prediction = model.predict(X_test)
# print(X_test)
print(r2_score(Y_test, Y_prediction))

ip = {'GRE Score': [300], 'TOEFL Score': [100], 'University Rating': [3], 'SOP': [3.0], 'LOR': [3.0], 'CGPA': [9.00], 'Research': [0]}
ip_df = pd.DataFrame(data=ip)
print('Your Chances of getting an Admit are', model.predict(ip_df))
