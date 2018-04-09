import pandas as pd
from data_preprocessing import preprocess

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("bank.csv")
X_train, X_test, y_train, y_test = preprocess(dataset)

classifier = Sequential()

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
# convert from percentage to True/False to show results in confusion matrix
y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: {}".format(cm))
