import pandas as pd

from plot_and_pretty_print import pretty_print_cm
from plot_and_pretty_print import plot_classifier

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


dataset = pd.read_csv("ads.csv")
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
pretty_print_cm(cm)

f1_score = metrics.f1_score(y_test, y_pred)
print("F1-score: {}".format(f1_score))

plot_classifier(X_train, y_train, classifier)

plot_classifier(X_test, y_test, classifier, False)
