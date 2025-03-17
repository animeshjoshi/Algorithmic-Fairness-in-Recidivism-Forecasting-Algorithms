
import pandas as pd
import numpy as np

k = 250

df = pd.read_csv('training_data.csv')
df = df.drop('Unnamed: 0', axis = 1)
df = df.sample(frac = 1)

observations = len(df)

size = len(df) // k

index = 0
folds = []
for fold in range(0, k):
    
    folds.append(df[index:index+size])

    index += size

index = 0

for x in folds:

    title = 'Folds/Fold ' + str(index) + '.csv'

    x.to_csv(title)

    index += 1


