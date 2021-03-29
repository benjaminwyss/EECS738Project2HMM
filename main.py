import sys
import pandas as pd
import numpy as np
import hmm

#Check command line args
if len(sys.argv) < 2:
	print('Not enough arguments. Please use -g to generate fake text or -p to predict text')
	exit()

flag = sys.argv[1]
if flag[1] != 'g' and flag[1] != 'p':
	print('Bad flag specified. Please use -g to generate fake text or -p to predict text')
	exit()

#Read in text corpus
df = pd.read_csv('data/onionArticles.csv', dtype=str, header=None)
onionArticles = np.array(df, dtype=str)

#Train model
model = hmm.hmm()
model.fit(onionArticles)

#Based on flag, either generate or predict text
if flag[1] == 'g':
	lines = input('Enter how many lines of text would you like to generate: ')
	n = int(lines)
	for _ in range(n):
		print(model.generateFakeText())

if flag[1] == 'p':
	prompt = input('Enter your prompt for predicting text: ')
	print(model.predictText(prompt))