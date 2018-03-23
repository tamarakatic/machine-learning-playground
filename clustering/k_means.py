import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")
X = data.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = kmeans.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=100, c='r', label='Careful')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=100, c='b', label='Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=100, c='g', label='Target')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=100, c='c', label='Careless')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=100, c='m', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='y', label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
