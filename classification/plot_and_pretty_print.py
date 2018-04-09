import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def pretty_print_cm(cm, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    labels = ['Purchased', 'Not_Purchased']
    columnwidth = max([len(x) for x in labels] + [5])
    empty_cell = " " * columnwidth
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def plot_classifier(X, y, classifier, Train=True):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1,
                                   step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1,
                                   step=0.01))
    plt.contourf(X1, X2,
                 classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
                 .reshape(X1.shape),
                 alpha=0.75,
                 cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Kernel SVM ({type} set)'.format(type="Train" if Train else "Test"))
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
