import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("AirPassengers.csv")

dataset.head()
dataset["Month"] = pd.to_datetime(dataset['Month'])
indexedDataset = dataset.set_index(['Month'])


from datetime import datetime
indexedDataset.head(5)

plt.xlabel("Date")
plt.ylabel("no. of passengers")
plt.plot(indexedDataset)

rolmean = indexedDataset.rolling(window=12).mean()
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

orig = plt.plot(indexedDataset, color='blue', label='Original')
mean = plt.plot(rolmean, color='red',label='Rolling Mean')
std = plt.plot(rolstd, color='black',label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
