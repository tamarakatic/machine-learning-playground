import pandas as pd

from apyori import apriori

dataset = pd.read_csv("products.csv", header=None)
transaction = []

for i in range(0, 7501):
    transaction.append([str(dataset.values[i, j]) for j in range(0, 20)])

rules = apriori(transaction, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

list_of_rules = list(rules)
