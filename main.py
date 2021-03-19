import pandas as pd
import numpy as np

df = pd.read_csv('data/onionArticles.csv', dtype=str)
onionArticles = np.array(df)

uniqueWords = []

for article in onionArticles:
	for word in str(article).split():
		if word not in uniqueWords:
			uniqueWords.append(word)

print(len(uniqueWords))