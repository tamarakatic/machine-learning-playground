import pandas as pd
import matplotlib.pyplot as plt
import math


dataset = pd.read_csv('ads_optimisation.csv')

N = 10000
d = 10
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if(numbers_of_selections[i] > 0):
            avarage_reward = sums_of_rewards[i] / numbers_of_selections[i]
            confidence_interval = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = avarage_reward + confidence_interval
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_rewards += reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
