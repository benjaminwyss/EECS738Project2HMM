import pandas as pd

df = pd.read_csv('data/OnionOrNot.csv')

df = df[df.label == 1]

for index, row in df.iterrows():
	text = str(row['text'])
	cleanText = ''
	for c in text:
		if c in ["'", '"', ':', '‘', '’', '.', ',', '?', '!', '“', '”', ';', '(', ')', '[', ']', '`']:
			continue
		elif c in ['—', '-', '–', '/', '\\', '|']:
			cleanText += ' '
		elif c == '%':
			cleanText += ' percent'
		else:
			cleanText += c

	

	df.at[index, 'text'] = cleanText.lower()
	

print(df)

df.to_csv('data/onionArticles.csv', columns=['text'], index=False, header=False)