import pandas as pd
import numpy as np
import hmm

df = pd.read_csv('data/onionArticles.csv', dtype=str, header=None)
onionArticles = np.array(df, dtype=str)

model = hmm.hmm()
model.fit(onionArticles)

for _ in range(100):
	print(model.generateFakeText())

prompt = input("Enter your prompt for predicting text: ")
print(model.predictText(prompt))