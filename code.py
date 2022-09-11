import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd

import tensorflow as tf

customer_data = pd.read_csv('Churn_Modelling.csv')
#customer_data = pd.read_csv(URL)
# Review the top rows of what is left of the data frame
customer_data.head()

x = customer_data.iloc[:, 3:-1].values
y = customer_data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
customer_data = LabelEncoder()
x[:, 2] = customer_data.fit_transform(x[:, 2])
print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(column.fit_transform(x))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
x_train = ssc.fit_transform(x_train)
x_test = ssc.transform(x_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(x_train, y_train, batch_size = 36, epochs = 119)

print(ann.predict(ssc.transform([[1, 0, 0, 500, 1, 40, 3, 70000, 2, 1, 1, 60000]])) > 0.5)

predictions = ann.predict(x_test)
predictions = (predictions > 0.5)

print(np.concatenate((predictions.reshape(len(predictions),1), y_test.reshape(len(y_test),1)),1))
#Finding accuracy of model
from sklearn.metrics import  accuracy_score

accuracy_score(y_test, predictions)
