from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess(dataset):
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    label_encoder_X_1 = LabelEncoder()
    X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])
    label_encoder_X_2 = LabelEncoder()
    X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

    one_hot_encoder = OneHotEncoder(categorical_features=[1])
    X = one_hot_encoder.fit_transform(X).toarray()
    X = X[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, X_test, y_train, y_test
